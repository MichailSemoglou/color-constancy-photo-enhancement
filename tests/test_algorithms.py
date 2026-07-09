"""Tests for all color constancy algorithm classes.

Covers:
- Output shape, dtype, and value-range invariants (parameterized over all algorithms).
- Algorithm-specific correctness assertions.
- Pipeline composition.
- Multi-Scale Retinex (MSR) and MSRCR.
"""

import numpy as np
import pytest

from color_constancy.algorithms import (
    MSRCR,
    AlgorithmPipeline,
    GrayWorldCorrection,
    MultiScaleRetinex,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
    build_combined_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_ALGORITHMS = [
    GrayWorldCorrection(),
    WhitePatchCorrection(),
    VonKriesAdaptation(),
    RetinexEnhancement(),
    MultiScaleRetinex(),
    MSRCR(),
    SpatialColorCorrection(),
    build_combined_pipeline(),
]
_ALL_IDS = [type(a).__name__ for a in _ALL_ALGORITHMS]

# ---------------------------------------------------------------------------
# Universal invariants (shape / dtype / value range)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", _ALL_ALGORITHMS, ids=_ALL_IDS)
def test_output_shape_preserved(algo, random_image):
    assert algo.process(random_image).shape == random_image.shape


@pytest.mark.parametrize("algo", _ALL_ALGORITHMS, ids=_ALL_IDS)
def test_output_dtype_float32(algo, random_image):
    assert algo.process(random_image).dtype == np.float32


@pytest.mark.parametrize("algo", _ALL_ALGORITHMS, ids=_ALL_IDS)
def test_output_values_in_unit_range(algo, random_image):
    out = algo.process(random_image)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("algo", _ALL_ALGORITHMS, ids=_ALL_IDS)
def test_near_black_input_safe(algo, near_black_image):
    """No NaN / Inf and values remain in [0, 1] on near-black input."""
    out = algo.process(near_black_image)
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


# ---------------------------------------------------------------------------
# GrayWorldCorrection
# ---------------------------------------------------------------------------


def test_gray_world_neutral_image_preserved(neutral_image):
    """A neutral grey image should be returned essentially unchanged."""
    out = GrayWorldCorrection().process(neutral_image)
    np.testing.assert_allclose(out, neutral_image, atol=1e-5)


def test_gray_world_reduces_red_cast(red_cast_image):
    """After correction, the red channel cast should be smaller."""
    algo = GrayWorldCorrection()
    original_means = red_cast_image.mean(axis=(0, 1))
    original_cast = abs(original_means[0] - original_means.mean())

    out = algo.process(red_cast_image)
    corrected_means = out.mean(axis=(0, 1))
    corrected_cast = abs(corrected_means[0] - corrected_means.mean())

    assert corrected_cast < original_cast


def test_gray_world_illuminant_estimate_shape(random_image):
    illum = GrayWorldCorrection().estimate_illuminant(random_image)
    assert illum.shape == (3,)
    assert illum.dtype == np.float32


def test_gray_world_illuminant_is_channel_mean(random_image):
    illum = GrayWorldCorrection().estimate_illuminant(random_image)
    expected = random_image.mean(axis=(0, 1))
    np.testing.assert_allclose(illum, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# WhitePatchCorrection
# ---------------------------------------------------------------------------


def test_white_patch_max_approaches_one(random_image):
    """After correction the brightest pixel in each channel should be ~1."""
    out = WhitePatchCorrection().process(random_image)
    assert float(out.max()) > 0.95


def test_white_patch_no_overflow_on_near_black(near_black_image):
    """Near-black input should not overflow — the old code had this bug."""
    out = WhitePatchCorrection().process(near_black_image)
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= 0.0


def test_white_patch_illuminant_estimate_shape(random_image):
    illum = WhitePatchCorrection().estimate_illuminant(random_image)
    assert illum.shape == (3,)


def test_white_patch_illuminant_is_channel_max(random_image):
    illum = WhitePatchCorrection().estimate_illuminant(random_image)
    expected = random_image.max(axis=(0, 1))
    np.testing.assert_allclose(illum, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# VonKriesAdaptation
# ---------------------------------------------------------------------------


def test_von_kries_illuminant_estimate_shape(random_image):
    illum = VonKriesAdaptation().estimate_illuminant(random_image)
    assert illum.shape == (3,)
    assert illum.dtype == np.float32


def test_von_kries_neutral_image_preserved(neutral_image):
    """A neutral grey image has no color cast; Von Kries should leave it unchanged."""
    out = VonKriesAdaptation().process(neutral_image)
    np.testing.assert_allclose(out, neutral_image, atol=1e-5)


def test_von_kries_reduces_red_cast(red_cast_image):
    algo = VonKriesAdaptation()
    original_means = red_cast_image.mean(axis=(0, 1))
    original_cast = abs(original_means[0] - original_means.mean())

    out = algo.process(red_cast_image)
    corrected_means = out.mean(axis=(0, 1))
    corrected_cast = abs(corrected_means[0] - corrected_means.mean())

    assert corrected_cast < original_cast


# ---------------------------------------------------------------------------
# RetinexEnhancement
# ---------------------------------------------------------------------------


def test_retinex_blend_alpha_zero_returns_original(random_image):
    """blend_alpha=0 should return the original image unchanged."""
    out = RetinexEnhancement(blend_alpha=0.0).process(random_image)
    np.testing.assert_allclose(out, random_image, atol=1e-5)


# ---------------------------------------------------------------------------
# SpatialColorCorrection
# ---------------------------------------------------------------------------


def test_spatial_neutral_image_preserved(neutral_image):
    """Uniform grey image: local mean equals global mean, correction is 1.0."""
    out = SpatialColorCorrection().process(neutral_image)
    np.testing.assert_allclose(out, neutral_image, atol=1e-4)


# ---------------------------------------------------------------------------
# AlgorithmPipeline
# ---------------------------------------------------------------------------


def test_pipeline_empty_returns_input(random_image):
    """An empty pipeline should return the image unchanged."""
    out = AlgorithmPipeline([]).process(random_image)
    np.testing.assert_array_equal(out, random_image)


def test_pipeline_single_step_matches_direct(random_image):
    gw = GrayWorldCorrection()
    pipeline = AlgorithmPipeline([gw])
    np.testing.assert_allclose(
        pipeline.process(random_image),
        gw.process(random_image),
        atol=1e-6,
    )


def test_pipeline_two_steps_matches_sequential(random_image):
    gw = GrayWorldCorrection()
    wp = WhitePatchCorrection()
    pipeline = AlgorithmPipeline([gw, wp])
    expected = wp.process(gw.process(random_image))
    np.testing.assert_allclose(pipeline.process(random_image), expected, atol=1e-6)


def test_build_combined_pipeline_type_and_length():
    p = build_combined_pipeline()
    assert isinstance(p, AlgorithmPipeline)
    assert len(p.steps) == 3


# ---------------------------------------------------------------------------
# Multi-Scale Retinex (MSR)
# ---------------------------------------------------------------------------


def test_msr_blend_alpha_zero_returns_original(random_image):
    out = MultiScaleRetinex(blend_alpha=0.0).process(random_image)
    np.testing.assert_allclose(out, random_image, atol=1e-5)


def test_msr_output_preserves_invariants(random_image):
    out = MultiScaleRetinex().process(random_image)
    assert out.shape == random_image.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_msr_different_scales_produces_different_output(random_image):
    msr_small = MultiScaleRetinex(sigmas=(5.0, 10.0, 20.0))
    msr_large = MultiScaleRetinex(sigmas=(50.0, 150.0, 300.0))
    out_small = msr_small.process(random_image)
    out_large = msr_large.process(random_image)
    assert not np.allclose(out_small, out_large, atol=0.01)


# ---------------------------------------------------------------------------
# MSRCR
# ---------------------------------------------------------------------------


def test_msrcr_blend_alpha_zero_returns_original(random_image):
    out = MSRCR(blend_alpha=0.0).process(random_image)
    np.testing.assert_allclose(out, random_image, atol=1e-5)


def test_msrcr_output_in_unit_range(random_image):
    out = MSRCR().process(random_image)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert np.isfinite(out).all()


def test_msrcr_different_gains_produce_different_output(random_image):
    vivid = MSRCR(cr_gain=200.0, cr_bias=-60.0)
    muted = MSRCR(cr_gain=50.0, cr_bias=-20.0)
    out_vivid = vivid.process(random_image)
    out_muted = muted.process(random_image)
    assert not np.allclose(out_vivid, out_muted, atol=0.005)


def test_combined_pipeline_uses_msrcr(random_image):
    p = build_combined_pipeline()
    assert any("MSRCR" in type(s).__name__ for s in p.steps)
    out = p.process(random_image)
    assert out.shape == random_image.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
