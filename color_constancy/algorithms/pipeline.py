"""Sequential algorithm pipeline and factory helpers."""

from collections.abc import Sequence

import numpy as np

from .base import ColorConstancyAlgorithm
from .gray_world import GrayWorldCorrection
from .retinex import RetinexEnhancement
from .von_kries import VonKriesAdaptation


class AlgorithmPipeline(ColorConstancyAlgorithm):
    """Sequential pipeline that applies algorithms left-to-right.

    The output of each step becomes the input of the next.  An empty pipeline
    returns the image unchanged.

    Parameters
    ----------
    steps:
        Sequence of :class:`~color_constancy.algorithms.base.ColorConstancyAlgorithm`
        instances to apply in order.
    """

    def __init__(self, steps: Sequence[ColorConstancyAlgorithm]) -> None:
        self.steps = list(steps)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply each step in order.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Processed image, same shape and dtype.
        """
        result = image
        for step in self.steps:
            result = step.process(result)
        return result


def build_combined_pipeline() -> AlgorithmPipeline:
    """Return the default combined pipeline: Grey World → Von Kries → Retinex.

    The three stages complement each other:

    1. **Grey World** removes the gross global colour cast.
    2. **Von Kries** (gentler parameters) fine-tunes the illuminant adaptation
       without over-correcting naturally coloured scenes.
    3. **Retinex** enhances local contrast and residual tonal range.

    Returns
    -------
    AlgorithmPipeline
        A configured, ready-to-use pipeline instance.
    """
    return AlgorithmPipeline(
        [
            GrayWorldCorrection(),
            VonKriesAdaptation(adaptation_strength=0.5, clip_range=(0.7, 1.4)),
            RetinexEnhancement(),
        ]
    )
