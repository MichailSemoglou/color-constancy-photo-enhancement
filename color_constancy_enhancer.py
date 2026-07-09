#!/usr/bin/env python3
"""Backward-compatibility entry point.

The implementation has been restructured into the ``color_constancy`` package.
This module re-exports the public API and provides a ``ColorConstancyEnhancer``
facade that preserves the original single-class interface.

For new code, import from ``color_constancy`` directly::

    from color_constancy import build_combined_pipeline, load_image, save_image
    import numpy as np

    image = load_image("photo.jpg")
    pipeline = build_combined_pipeline()
    enhanced_f = pipeline.process(image.astype("float32") / 255.0)
    enhanced = (enhanced_f * 255).astype("uint8")
    save_image(enhanced, "enhanced.jpg")
"""

from __future__ import annotations

import numpy as np

from color_constancy import (
    MSRCR,
    GrayWorldCorrection,
    MultiScaleRetinex,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
    build_combined_pipeline,
    color_statistics,
    display_comparison,
    load_image,
    save_image,
)
from color_constancy.cli import main  # re-exported for the entry-point

_SINGLE_METHODS = {
    "gray_world": GrayWorldCorrection,
    "white_patch": WhitePatchCorrection,
    "von_kries": VonKriesAdaptation,
    "retinex": RetinexEnhancement,
    "msr": MultiScaleRetinex,
    "msrcr": MSRCR,
    "spatial": SpatialColorCorrection,
}


class ColorConstancyEnhancer:
    """Facade that preserves the original single-class API.

    For new projects prefer the composable API in ``color_constancy``.
    """

    def __init__(self) -> None:
        self.original_image: np.ndarray | None = None
        self.enhanced_image: np.ndarray | None = None

    def enhance_image(
        self,
        image_path: str,
        method: str = "combined",
        output_path: str | None = None,
    ) -> np.ndarray:
        """Load *image_path*, apply *method*, optionally save, return enhanced uint8.

        Parameters
        ----------
        image_path:
            Path to the source image.
        method:
            One of ``'gray_world'``, ``'white_patch'``, ``'von_kries'``,
            ``'retinex'``, ``'spatial'``, ``'combined'``.
        output_path:
            If provided, save the enhanced image to this path.

        Returns
        -------
        np.ndarray
            Enhanced RGB image, dtype ``uint8``.
        """
        original = load_image(image_path)
        self.original_image = original

        img_float = original.astype(np.float32) / 255.0

        if method == "combined":
            algorithm = build_combined_pipeline()
        elif method in _SINGLE_METHODS:
            algorithm = _SINGLE_METHODS[method]()
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                f"Choose from {sorted(_SINGLE_METHODS) + ['combined']}."
            )

        enhanced_float = algorithm.process(img_float)
        enhanced = (enhanced_float * 255.0).astype(np.uint8)
        self.enhanced_image = enhanced

        if output_path:
            save_image(enhanced, output_path)

        return enhanced

    def display_results(self, save_comparison: str | None = None) -> None:
        """Show a side-by-side before/after comparison.

        Must be called after :meth:`enhance_image`.

        Parameters
        ----------
        save_comparison:
            If provided, save the comparison figure to this path.
        """
        if self.original_image is None or self.enhanced_image is None:
            raise RuntimeError("Call enhance_image() before display_results().")
        display_comparison(
            self.original_image,
            self.enhanced_image,
            save_path=save_comparison,
        )

    def analyze_color_statistics(self, image: np.ndarray) -> dict[str, float]:
        """Return per-channel statistics for *image* (uint8 input).

        Parameters
        ----------
        image:
            RGB image, dtype ``uint8``, shape ``(H, W, 3)``.

        Returns
        -------
        dict
            Keys: ``mean_r``, ``mean_g``, ``mean_b``, ``std_r``, ``std_g``,
            ``std_b``, ``red_cast``, ``green_cast``, ``blue_cast``.
        """
        return color_statistics(image.astype(np.float32) / 255.0)


if __name__ == "__main__":
    main()

