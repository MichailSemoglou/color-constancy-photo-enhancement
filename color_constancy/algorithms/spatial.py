"""Spatially-varying local color correction."""

import numpy as np
from scipy import ndimage

from .base import ColorConstancyAlgorithm


class SpatialColorCorrection(ColorConstancyAlgorithm):
    """Spatially-varying local color correction.

    Estimates a smooth per-pixel local illuminant using a large Gaussian
    neighborhood mean (computed via ``scipy.ndimage.gaussian_filter`` for
    efficiency).  Each pixel is then corrected toward the global mean
    relative to its local neighborhood.

    The correction ratio is clipped to
    ``[1 - correction_strength, 1 + correction_strength]`` to prevent
    over-saturation in regions with very dark local means.

    This implementation replaces the original nested Python loop (which used
    ``np.maximum`` for patch blending, causing a systematic over-brightening
    bias) with a fully vectorised approach.  The Gaussian filter is applied
    once per channel, making the runtime ``O(H × W)`` rather than
    ``O(H × W × window_size²)``.

    Parameters
    ----------
    correction_strength:
        Half-width of the clipping range for per-pixel correction ratios.
        Default ``0.2`` gives corrections in ``[0.8, 1.2]``.
    """

    def __init__(self, correction_strength: float = 0.2) -> None:
        self.correction_strength = correction_strength

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply spatially-varying local color correction.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Corrected image, same shape and dtype.
        """
        height, width = image.shape[:2]
        # Neighborhood radius: one-eighth of the shortest spatial dimension,
        # with a minimum of 8 to keep the filter meaningful.
        window_size = max(8, min(height, width) // 8)
        sigma = window_size / 3.0

        global_mean = image.mean(axis=(0, 1))  # (3,)

        # Compute local mean per channel with a single Gaussian pass each —
        # fully vectorised, no Python loops over spatial coordinates.
        local_mean = np.stack(
            [ndimage.gaussian_filter(image[:, :, c], sigma=sigma) for c in range(3)],
            axis=2,
        )  # (H, W, 3)

        # Avoid division by near-zero local means.
        safe_local = np.where(local_mean > 1e-6, local_mean, 1e-6)
        raw_correction = global_mean / safe_local  # (H, W, 3)

        lo = 1.0 - self.correction_strength
        hi = 1.0 + self.correction_strength
        correction = np.clip(raw_correction, lo, hi)

        return np.clip(image * correction, 0.0, 1.0).astype(np.float32)
