"""Tests for the backward-compatible ColorConstancyEnhancer facade."""

from unittest.mock import patch

import numpy as np
import pytest

from color_constancy.io import save_image
from color_constancy_enhancer import ColorConstancyEnhancer


@pytest.fixture()
def png_path(tmp_path) -> str:
    """A small PNG with a mild red cast, so corrections have an effect."""
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    img[:, :, 0] = 200
    path = tmp_path / "input.png"
    save_image(img, str(path))
    return str(path)


@pytest.mark.parametrize(
    "method",
    ["gray_world", "white_patch", "von_kries", "retinex", "msr",
     "msrcr", "spatial", "sme", "combined"],
)
def test_enhance_image_all_methods(png_path, method):
    enhancer = ColorConstancyEnhancer()
    out = enhancer.enhance_image(png_path, method=method)
    assert out.dtype == np.uint8
    assert out.shape == (32, 32, 3)
    assert enhancer.original_image is not None
    assert enhancer.enhanced_image is not None


def test_enhance_image_saves_output(png_path, tmp_path):
    out_path = tmp_path / "out.png"
    enhancer = ColorConstancyEnhancer()
    returned = enhancer.enhance_image(png_path, method="gray_world", output_path=str(out_path))
    assert out_path.exists()
    assert np.array_equal(returned, enhancer.enhanced_image)


def test_enhance_image_unknown_method_raises(png_path):
    with pytest.raises(ValueError, match="Unknown method"):
        ColorConstancyEnhancer().enhance_image(png_path, method="does_not_exist")


def test_display_results_requires_enhance_first():
    with pytest.raises(RuntimeError, match="enhance_image"):
        ColorConstancyEnhancer().display_results()


def test_display_results_saves_comparison(png_path, tmp_path):
    enhancer = ColorConstancyEnhancer()
    enhancer.enhance_image(png_path, method="gray_world")
    cmp_path = tmp_path / "comparison.png"
    with patch("color_constancy.visualization.plt.show"):
        enhancer.display_results(save_comparison=str(cmp_path))
    assert cmp_path.exists()


def test_analyze_color_statistics_keys_and_range():
    stats = ColorConstancyEnhancer().analyze_color_statistics(
        np.full((8, 8, 3), 128, dtype=np.uint8)
    )
    assert set(stats) == {
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "red_cast", "green_cast", "blue_cast",
    }
    assert all(0.0 <= stats[f"mean_{c}"] <= 1.0 for c in "rgb")
