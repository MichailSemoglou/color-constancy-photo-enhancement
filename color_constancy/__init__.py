"""color_constancy — Photo enhancement using color constancy algorithms.

Quick start
-----------
>>> from color_constancy import build_combined_pipeline, load_image, save_image
>>> import numpy as np
>>> image = load_image("photo.jpg")                       # uint8 RGB
>>> pipeline = build_combined_pipeline()
>>> enhanced_f = pipeline.process(image.astype("float32") / 255.0)
>>> enhanced = (enhanced_f * 255).astype("uint8")
>>> save_image(enhanced, "enhanced.jpg")

For the original single-class API (backward-compatible) see
``color_constancy_enhancer.ColorConstancyEnhancer``.
"""

from .algorithms import (
    AlgorithmPipeline,
    GrayWorldCorrection,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
    build_combined_pipeline,
)
from .io import load_image, save_image
from .metrics import angular_error, color_statistics, psnr, ssim
from .visualization import display_comparison, visualize_illuminant

__version__ = "1.1.0"

__all__ = [
    # algorithms
    "GrayWorldCorrection",
    "WhitePatchCorrection",
    "VonKriesAdaptation",
    "RetinexEnhancement",
    "SpatialColorCorrection",
    "AlgorithmPipeline",
    "build_combined_pipeline",
    # I/O
    "load_image",
    "save_image",
    # metrics
    "angular_error",
    "color_statistics",
    "psnr",
    "ssim",
    # visualization
    "display_comparison",
    "visualize_illuminant",
    # metadata
    "__version__",
]
