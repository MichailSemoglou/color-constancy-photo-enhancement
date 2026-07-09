"""Von Kries chromatic adaptation."""


import numpy as np

from .base import ColorConstancyAlgorithm
from .constants import NEUTRAL_ILLUMINANT


class VonKriesAdaptation(ColorConstancyAlgorithm):
    """Von Kries chromatic adaptation (von Kries, 1902).

    Applies a per-channel diagonal scaling to the cone (RGB channel) responses
    to normalise the estimated scene illuminant to a neutral reference.  The
    implementation blends a Grey World estimate with a specular-highlight
    (White Patch) estimate for robustness across scene types.

    The raw adaptation coefficients are clipped to ``clip_range`` and then
    blended toward the identity (1.0) via ``adaptation_strength``, producing a
    partial rather than full chromatic adaptation.  This avoids
    over-correction on naturally colored scenes.

    Parameters
    ----------
    adaptation_strength:
        Blending factor in ``[0, 1]``.  ``1.0`` applies the full diagonal
        adaptation; lower values produce gentler corrections.  Default ``0.6``.
    clip_range:
        Hard limits on the raw adaptation coefficients before blending.
        Prevents extreme corrections on unusual images.  Default ``(0.6, 1.7)``.
    gray_world_weight:
        Weight assigned to the Grey World component of the illuminant blend.
        The complementary weight ``(1 - gray_world_weight)`` goes to the
        specular-highlight estimate.  Default ``0.7``.

    References
    ----------
    von Kries, J. (1902). Chromatic adaptation. *Festschrift der Albrecht-Ludwigs-
    Universität*, 145–158.

    Foster, D. H. (2011). Color constancy. *Vision Research*, 51(7), 674–700.
    """

    def __init__(
        self,
        adaptation_strength: float = 0.6,
        clip_range: tuple = (0.6, 1.7),
        gray_world_weight: float = 0.7,
    ) -> None:
        self.adaptation_strength = adaptation_strength
        self.clip_range = clip_range
        self.gray_world_weight = gray_world_weight

    def estimate_illuminant(self, image: np.ndarray) -> np.ndarray:
        """Estimate the scene illuminant as a weighted blend of Grey World
        and specular-highlight (White Patch) estimates.

        Pixels in the top 5 % of mean luminance are used as the
        specular-highlight proxy.  If no such pixels exist the method falls
        back to the pure Grey World estimate.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Shape ``(3,)``, the estimated illuminant chromaticity.
        """
        gray_world = image.mean(axis=(0, 1))

        brightness = image.mean(axis=2)
        threshold = np.percentile(brightness, 95)
        bright_mask = brightness > threshold

        if bright_mask.any():
            specular = image[bright_mask].mean(axis=0)
            w = self.gray_world_weight
            return (w * gray_world + (1.0 - w) * specular).astype(np.float32)

        return gray_world.astype(np.float32)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Von Kries chromatic adaptation.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Adapted image, same shape and dtype.
        """
        illuminant = self.estimate_illuminant(image)

        # Avoid division by near-zero illuminant values.
        safe_illuminant = np.maximum(illuminant, 0.02)

        # Normalise to chromaticity (remove brightness) before computing the
        # adaptation ratio.  Without this step, a neutral grey scene produces
        # coefficients < 1 and dims the image even though no color cast exists,
        # because the raw illuminant mean is compared against the fixed
        # NEUTRAL_ILLUMINANT magnitude (1/3) instead of its chromaticity.
        illuminant_chroma = safe_illuminant / safe_illuminant.sum()
        raw_coefficients = NEUTRAL_ILLUMINANT / illuminant_chroma

        # Clip to prevent extreme per-channel scaling.
        clipped = np.clip(raw_coefficients, *self.clip_range)

        # Blend toward identity for gentler, more natural-looking output.
        coefficients = 1.0 + self.adaptation_strength * (clipped - 1.0)

        return np.clip(image * coefficients, 0.0, 1.0).astype(np.float32)
