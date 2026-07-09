"""Gray World illuminant estimation and correction."""

import numpy as np

from .base import ColorConstancyAlgorithm


class GrayWorldCorrection(ColorConstancyAlgorithm):
    """Gray World illuminant estimation (Buchsbaum, 1980).

    Assumes the spatial average of scene reflectances is achromatic: the mean
    of each color channel over the whole image should equal the global mean
    brightness.  The illuminant is estimated as the per-channel spatial mean,
    and each channel is scaled so that all three means become equal.

    This assumption breaks down for images dominated by a single hue (e.g.
    sunsets, grass fields, clear sky), where the gray-world prior does not hold.

    References
    ----------
    Buchsbaum, G. (1980). A spatial processor model for object color perception.
    *Journal of the Franklin Institute*, 310(1), 1–26.
    """

    def estimate_illuminant(self, image: np.ndarray) -> np.ndarray:
        """Return the per-channel spatial mean as the illuminant estimate.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Shape ``(3,)``, the estimated illuminant chromaticity.
        """
        return image.mean(axis=(0, 1)).astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Gray World correction.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Corrected image, same shape and dtype.
        """
        illuminant = self.estimate_illuminant(image)
        global_mean = illuminant.mean()
        # Guard against near-black channels to avoid division by zero.
        safe = np.where(illuminant > 1e-6, illuminant, 1e-6)
        correction = global_mean / safe
        return np.clip(image * correction, 0.0, 1.0).astype(np.float32)
