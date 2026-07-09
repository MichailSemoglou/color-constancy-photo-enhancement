"""Single-Scale Retinex (SSR), Multi-Scale Retinex (MSR), and MSRCR."""

import numpy as np
from scipy import ndimage

from .base import ColorConstancyAlgorithm

# Sigma applied before the log transform to suppress shot noise.
_PRE_SMOOTH_SIGMA: float = 0.5
# Mild gamma applied after percentile normalization for tonal balance.
_GAMMA: float = 0.9
# Weight of the surround response subtracted from the log-domain signal.
_SURROUND_WEIGHT: float = 0.3
# Small offset added before log to avoid log(0).
_LOG_OFFSET: float = 0.04


def _ssr_channel(channel: np.ndarray, sigma: float, pre_smooth: float, offset: float) -> np.ndarray:
    """Core SSR computation for a single channel and a single scale."""
    smoothed = ndimage.gaussian_filter(channel, sigma=pre_smooth)
    log_ch = np.log(smoothed + offset)
    surround = ndimage.gaussian_filter(log_ch, sigma=sigma)
    return log_ch - _SURROUND_WEIGHT * surround


def _apply_ssr(
    image: np.ndarray,
    sigma: float,
    pre_smooth: float = _PRE_SMOOTH_SIGMA,
    offset: float = _LOG_OFFSET,
) -> np.ndarray:
    """Apply SSR across all three RGB channels."""
    out = np.stack(
        [_ssr_channel(image[:, :, c], sigma, pre_smooth, offset) for c in range(3)],
        axis=2,
    )
    return out


def _percentile_stretch(enhanced: np.ndarray) -> np.ndarray:
    """Stretch values to [0, 1] using 5th/95th percentiles and apply gamma."""
    p_low = np.percentile(enhanced, 5)
    p_high = np.percentile(enhanced, 95)
    if p_high > p_low:
        enhanced = (enhanced - p_low) / (p_high - p_low)
    return np.power(np.clip(enhanced, 0.0, 1.0), _GAMMA)


class RetinexEnhancement(ColorConstancyAlgorithm):
    """Single-Scale Retinex (SSR) enhancement.

    Models the center-surround processing of the human visual system.  For
    each channel, a log-domain image is computed and a Gaussian-smoothed
    surround (an estimate of the slowly-varying illumination) is subtracted.
    The result enhances local contrast while partially normalizing the global
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
    for bridging the gap between color images and the human observation of
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
        enhanced = _apply_ssr(image, sigma=self.surround_sigma)
        enhanced = _percentile_stretch(enhanced)
        result = self.blend_alpha * enhanced + (1.0 - self.blend_alpha) * image
        return np.clip(result, 0.0, 1.0).astype(np.float32)


class MultiScaleRetinex(ColorConstancyAlgorithm):
    """Multi-Scale Retinex (MSR) enhancement (Jobson et al., 1997).

    Averages SSR outputs at multiple surround scales — typically a small,
    medium, and large sigma — to balance dynamic range compression and tonal
    rendition.  Unlike SSR, which must trade off between local contrast and
    color fidelity depending on a single scale, MSR combines the benefits of
    all three.

    Parameters
    ----------
    sigmas:
        Sequence of surround sigma values.  Default ``(15.0, 80.0, 250.0)``,
        following the canonical small/medium/large triplet from Jobson et al.
    blend_alpha:
        Weight of the MSR output in the final linear blend with the original
        image.  Default ``0.7``.

    References
    ----------
    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex
    for bridging the gap between color images and the human observation of
    scenes.  *IEEE Transactions on Image Processing*, 6(7), 965–976.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
        blend_alpha: float = 0.7,
    ) -> None:
        self.sigmas = sigmas
        self.blend_alpha = blend_alpha

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Multi-Scale Retinex enhancement.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Enhanced image, same shape and dtype.
        """
        accumulated = np.zeros_like(image)
        for sigma in self.sigmas:
            accumulated += _apply_ssr(image, sigma=sigma)
        msr = accumulated / len(self.sigmas)

        msr = _percentile_stretch(msr)
        result = self.blend_alpha * msr + (1.0 - self.blend_alpha) * image
        return np.clip(result, 0.0, 1.0).astype(np.float32)


class MSRCR(ColorConstancyAlgorithm):
    """Multi-Scale Retinex with Color Restoration (MSRCR).

    Extends MSR with a per-channel color restoration step (Jobson et al.,
    1997) that compensates for the desaturation MSR/SSR can introduce.  After
    computing the MSR output, each channel is multiplied by a color
    restoration coefficient derived from the log-ratio of the original channel
    to the sum of all channels.

    Parameters
    ----------
    sigmas:
        Sequence of surround sigma values.  Default ``(15.0, 80.0, 250.0)``.
    blend_alpha:
        Weight of the MSRCR output in the final linear blend.  Default ``0.7``.
    cr_gain:
        Gain applied to the color restoration factor.  Higher values produce
        more vivid colors.  Default ``125.0`` (canonical from Jobson et al.).
    cr_bias:
        Offset added to the color-restored MSR before blending.  Default
        ``-46.0`` (canonical from Jobson et al.).

    References
    ----------
    Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex
    for bridging the gap between color images and the human observation of
    scenes.  *IEEE Transactions on Image Processing*, 6(7), 965–976.
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
        blend_alpha: float = 0.7,
        cr_gain: float = 125.0,
        cr_bias: float = -46.0,
    ) -> None:
        self.sigmas = sigmas
        self.blend_alpha = blend_alpha
        self.cr_gain = cr_gain
        self.cr_bias = cr_bias

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply MSRCR enhancement.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Enhanced image, same shape and dtype.
        """
        # --- MSR ---
        accumulated = np.zeros_like(image)
        for sigma in self.sigmas:
            accumulated += _apply_ssr(image, sigma=sigma)
        msr = accumulated / len(self.sigmas)

        # --- Color restoration ---
        # CR_c = beta * (log(alpha * I_c) - log(sum_j(I_j)))
        # where beta * (...) produces a per-channel weighting.
        # We absorb beta into cr_gain for simplicity.
        img_sum = image.sum(axis=2, keepdims=True) + 1e-6
        cr = self.cr_gain * (np.log(self.cr_gain * image + _LOG_OFFSET) - np.log(img_sum + _LOG_OFFSET))
        cr = np.clip(cr, -self.cr_gain, self.cr_gain)

        # Apply color restoration: multiply MSR by CR, then add bias
        msrcr = msr * cr + self.cr_bias

        msrcr = _percentile_stretch(msrcr)
        result = self.blend_alpha * msrcr + (1.0 - self.blend_alpha) * image
        return np.clip(result, 0.0, 1.0).astype(np.float32)
