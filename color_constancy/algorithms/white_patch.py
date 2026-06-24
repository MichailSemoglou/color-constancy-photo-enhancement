"""White Patch / Max-RGB illuminant estimation and correction."""

import numpy as np

from .base import ColorConstancyAlgorithm


class WhitePatchCorrection(ColorConstancyAlgorithm):
    """White Patch / Max-RGB illuminant estimation (Land & McCann, 1971).

    Assumes that the brightest surface in the scene reflects the illuminant
    maximally across all wavelengths.  Each colour channel is normalised by
    its spatial maximum, mapping the brightest pixel to pure white.

    This assumption fails when the image contains specular highlights that are
    coloured (e.g. metallic objects) or when the brightest pixel is a
    single-channel saturated region rather than a white surface.

    References
    ----------
    Land, E. H., & McCann, J. J. (1971). Lightness and retinex theory.
    *Journal of the Optical Society of America*, 61(1), 1–11.
    """

    def estimate_illuminant(self, image: np.ndarray) -> np.ndarray:
        """Return the per-channel spatial maximum as the illuminant estimate.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Shape ``(3,)``, the estimated illuminant chromaticity.
        """
        return image.max(axis=(0, 1)).astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply White Patch correction.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Corrected image, same shape and dtype, clipped to ``[0, 1]``.
        """
        illuminant = self.estimate_illuminant(image)
        # Guard against near-black channels to avoid division by zero and
        # overflow, which previously caused silent uint8 wrap-around.
        safe = np.where(illuminant > 1e-6, illuminant, 1.0)
        return np.clip(image / safe, 0.0, 1.0).astype(np.float32)
