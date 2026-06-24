"""Tests for color_constancy.metrics."""

import math

import numpy as np
import pytest

from color_constancy.metrics import angular_error, color_statistics, psnr, ssim

# ---------------------------------------------------------------------------
# angular_error
# ---------------------------------------------------------------------------


def test_angular_error_identical_vectors():
    v = np.array([0.3, 0.4, 0.3])
    assert angular_error(v, v) == pytest.approx(0.0, abs=1e-6)


def test_angular_error_orthogonal_vectors():
    assert angular_error(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ) == pytest.approx(90.0, abs=1e-5)


def test_angular_error_antiparallel_vectors():
    assert angular_error(
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ) == pytest.approx(180.0, abs=1e-5)


def test_angular_error_symmetric():
    v1 = np.array([0.5, 0.3, 0.2])
    v2 = np.array([0.2, 0.5, 0.3])
    assert angular_error(v1, v2) == pytest.approx(angular_error(v2, v1), abs=1e-8)


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------


def test_psnr_identical_images():
    img = np.random.default_rng(0).random((32, 32, 3)).astype(np.float32)
    assert psnr(img, img) == math.inf


def test_psnr_positive_for_different_images():
    rng = np.random.default_rng(1)
    img1 = rng.random((32, 32, 3)).astype(np.float32)
    img2 = rng.random((32, 32, 3)).astype(np.float32)
    result = psnr(img1, img2)
    assert 0.0 < result < 100.0


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(2)
    ref = rng.random((32, 32, 3)).astype(np.float32)
    small_noise = (ref + 0.01 * rng.standard_normal(ref.shape)).astype(np.float32)
    large_noise = (ref + 0.10 * rng.standard_normal(ref.shape)).astype(np.float32)
    small_noise = np.clip(small_noise, 0.0, 1.0)
    large_noise = np.clip(large_noise, 0.0, 1.0)
    assert psnr(ref, small_noise) > psnr(ref, large_noise)


# ---------------------------------------------------------------------------
# ssim
# ---------------------------------------------------------------------------


def test_ssim_identical_images():
    img = np.random.default_rng(3).random((64, 64, 3)).astype(np.float32)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-4)


def test_ssim_less_than_one_for_different_images():
    rng = np.random.default_rng(4)
    img1 = rng.random((64, 64, 3)).astype(np.float32)
    img2 = rng.random((64, 64, 3)).astype(np.float32)
    assert ssim(img1, img2) < 1.0


def test_ssim_range():
    rng = np.random.default_rng(5)
    img1 = rng.random((64, 64, 3)).astype(np.float32)
    img2 = rng.random((64, 64, 3)).astype(np.float32)
    result = ssim(img1, img2)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# color_statistics
# ---------------------------------------------------------------------------


def test_color_statistics_keys(neutral_image):
    stats = color_statistics(neutral_image)
    expected = {
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "red_cast", "green_cast", "blue_cast",
    }
    assert set(stats.keys()) == expected


def test_color_statistics_neutral_image_zero_cast(neutral_image):
    stats = color_statistics(neutral_image)
    assert stats["red_cast"] == pytest.approx(0.0, abs=1e-6)
    assert stats["green_cast"] == pytest.approx(0.0, abs=1e-6)
    assert stats["blue_cast"] == pytest.approx(0.0, abs=1e-6)


def test_color_statistics_neutral_image_zero_std(neutral_image):
    stats = color_statistics(neutral_image)
    assert stats["std_r"] == pytest.approx(0.0, abs=1e-6)


def test_color_statistics_red_channel_has_positive_cast(red_cast_image):
    stats = color_statistics(red_cast_image)
    assert stats["red_cast"] > 0.0
    assert stats["green_cast"] < 0.0
    assert stats["blue_cast"] < 0.0
