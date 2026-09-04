"""Sequential algorithm pipeline and factory helpers."""

from collections.abc import Sequence

import numpy as np

from .base import ColorConstancyAlgorithm
from .gray_world import GrayWorldCorrection
from .retinex import MSRCR


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

    def __init__(self, steps: Sequence[ColorConstancyAlgorithm], _repr_name: str = "") -> None:
        self.steps = list(steps)
        self._repr_name = _repr_name

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
    """Return the default combined pipeline: Gray World → MSRCR.

    The two stages complement each other:

    1. **Gray World** removes the gross global color cast.
    2. **MSRCR** (Multi-Scale Retinex with Color Restoration) enhances local
       contrast and preserves color fidelity.

    .. note::
        Versions up to 1.3.0 included an intermediate Von Kries stage.  It
        was removed because, applied after Gray World, its measured effect
        was negligible (mean output change of about 0.003 in ``[0, 1]``).

    Returns
    -------
    AlgorithmPipeline
        A configured, ready-to-use pipeline instance.
    """
    return AlgorithmPipeline(
        [
            GrayWorldCorrection(),
            MSRCR(blend_alpha=0.7),
        ]
    )
