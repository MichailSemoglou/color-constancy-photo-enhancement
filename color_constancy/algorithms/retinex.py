"""Single-Scale Retinex (SSR) enhancement."""

import numpy as np
from scipy import ndimage

from .base import ColorConstancyAlgorithm

# Sigma applied before the log transform to suppress shot noise.
_PRE_SMOOTH_SIGMA: float = 0.5
# Mild gamma applied after percentile normalisation for tonal balance.
_GAMMA: float = 0.9
# Weight of the surround response subtracted from the log-domain signal.
_SURROUND_WEIGHT: float = 0.3
# Small offset added before log to avoid log(0).
_LOG_OFFSET: float = 0.04


class RetinexEnhancement(ColorConstancyAlgorithm):
    """Single-Scale Retinex (SSR) enhancement.

    Models the centre-surround processing of the human visual system.  For
    each channel, a log-domain image is computed and a Gaussian-smoothed
    surround (an estimate of the slowly-varying illumination) is subtracted.
    The result enhances local contrast while partially normalising the global
    illumination.

    .. note::
        This implementation is **Single-Scale Retinex (SSR)** using a single
        surround ``sigma``.  It is *not* Multi-Scale Retinex (MSR), which
        averages over several ``sigma`` values and is the more common choice in
        the literature (Jobson et al., 1997).  The pre-smoothing step
        (``sigma=0.5``) and the partial surround subtraction
        (``0.3 × surround``) are heuristic modifications for photographic use
        and are not part of the canonical SSR formula.

    Parameters
    ----------
    surround_sigma:
        Standard deviation of the Gaussian used to model the surround
        (illumination estimate).  Larger values produce broader smoothing
        and more aggressive illumination removal.  Default ``15.0``.
    blend_alpha:
        Weight of the Retinex output in the final linear blend with the
        original image.  ``1.0`` is pure Retinex; ``0.0`` returns the
        original unchanged.  Default ``0.6``.

    References
    ----------
    Land, E. H., & McCann, J. J. (1971). Lightness and retinex theory.
    *Journal of the Optical Society of America*, 61(1), 1–11.

    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex
    for bridging the gap between colour images and the human observation of
    scenes.  *IEEE Transactions on Image Processing*, 6(7), 965–976.
    """

    def __init__(
        self,
        surround_sigma: float = 15.0,
        blend_alpha: float = 0.6,
    ) -> None:
        self.surround_sigma = surround_sigma
        self.blend_alpha = blend_alpha

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Single-Scale Retinex enhancement.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Enhanced image, same shape and dtype.
        """
        enhanced = np.zeros_like(image)

        for c in range(3):
            channel = image[:, :, c]
            # Minor noise suppression before entering the log domain.
            smoothed = ndimage.gaussian_filter(channel, sigma=_PRE_SMOOTH_SIGMA)
            log_channel = np.log(smoothed + _LOG_OFFSET)
            # Broad Gaussian models the slowly-varying illumination component.
            surround = ndimage.gaussian_filter(log_channel, sigma=self.surround_sigma)
            enhanced[:, :, c] = log_channel - _SURROUND_WEIGHT * surround

        # Stretch to [0, 1] using robust percentile limits (ignores outliers).
        p_low = np.percentile(enhanced, 5)
        p_high = np.percentile(enhanced, 95)
        if p_high > p_low:
            enhanced = (enhanced - p_low) / (p_high - p_low)

        # Mild gamma for tonal balance.
        enhanced = np.power(np.clip(enhanced, 0.0, 1.0), _GAMMA)

        # Blend with the original to preserve colour fidelity.
        result = self.blend_alpha * enhanced + (1.0 - self.blend_alpha) * image
        return np.clip(result, 0.0, 1.0).astype(np.float32)
