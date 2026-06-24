"""Tests for color_constancy.io (load_image, save_image, validation)."""

import numpy as np
import pytest
import cv2

from color_constancy.io import load_image, save_image


def _write_rgb_png(path, array: np.ndarray) -> None:
    """Helper: save an RGB uint8 array as PNG via OpenCV."""
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


def test_load_nonexistent_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(str(tmp_path / "no_such_image.png"))


def test_load_invalid_file_raises_value_error(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image file")
    with pytest.raises(ValueError, match="decode"):
        load_image(str(bad))


def test_load_small_image_raises_value_error(tmp_path):
    tiny = np.full((4, 4, 3), 128, dtype=np.uint8)
    path = tmp_path / "tiny.png"
    _write_rgb_png(path, tiny)
    with pytest.raises(ValueError, match="too small"):
        load_image(str(path))


def test_load_returns_rgb_uint8(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "img.png"
    _write_rgb_png(path, img)
    loaded = load_image(str(path))
    assert loaded.dtype == np.uint8
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_load_roundtrip_preserves_data(tmp_path):
    """Save with cv2 then load — pixel values must be identical (lossless PNG)."""
    img = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    path = tmp_path / "rt.png"
    _write_rgb_png(path, img)
    loaded = load_image(str(path))
    np.testing.assert_array_equal(loaded, img)


def test_load_grayscale_promoted_to_rgb(tmp_path):
    gray = np.full((32, 32), 128, dtype=np.uint8)
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), gray)
    loaded = load_image(str(path))
    assert loaded.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


def test_save_creates_output_file(tmp_path):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    out = tmp_path / "out.png"
    save_image(img, str(out))
    assert out.exists()


def test_save_creates_parent_directories(tmp_path):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    out = tmp_path / "a" / "b" / "c" / "out.png"
    save_image(img, str(out))
    assert out.exists()


def test_save_and_reload_roundtrip(tmp_path):
    img = np.random.default_rng(2).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = tmp_path / "roundtrip.png"
    save_image(img, str(out))
    reloaded = load_image(str(out))
    np.testing.assert_array_equal(reloaded, img)
