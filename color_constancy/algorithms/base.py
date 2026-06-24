"""Abstract base class for all color constancy algorithms."""

from abc import ABC, abstractmethod

import numpy as np


class ColorConstancyAlgorithm(ABC):
    """Abstract base for color constancy correction algorithms.

    Convention
    ----------
    All concrete subclasses must accept and return **float32** arrays with
    shape ``(H, W, 3)`` and pixel values in ``[0, 1]``.  This single-precision
    float contract eliminates repeated ``uint8 <-> float32`` conversions when
    algorithms are chained in a :class:`~color_constancy.algorithms.pipeline.AlgorithmPipeline`.

    The caller is responsible for converting ``uint8`` source images before
    invoking ``process`` and for converting the result back to ``uint8``
    afterwards (see :mod:`color_constancy.cli`).
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply color constancy correction.

        Parameters
        ----------
        image:
            Float32 RGB image, shape ``(H, W, 3)``, values in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            Corrected image, same shape and dtype, values clipped to ``[0, 1]``.
        """
