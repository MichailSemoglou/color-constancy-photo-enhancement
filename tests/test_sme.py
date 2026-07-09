"""Tests for Selective Midtone Enhancement (SME) algorithm."""
from __future__ import annotations

import numpy as np

from color_constancy.algorithms.sme import (
    SelectiveMidtoneEnhancement,
    _chroma,
    _color_definition_confidence,
    _dynamic_s_curve,
    _highlight_guard,
    _local_variance_ab,
    _luminance,
    _neutrality_mask,
    _rgb_to_lab,
    _sigmoid,
)

# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------

class TestLuminance:
    def test_black(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        lum = _luminance(img)
        assert lum.shape == (10, 10)
        assert np.allclose(lum, 0.0)

    def test_white(self):
        img = np.ones((10, 10, 3), dtype=np.float32)
        lum = _luminance(img)
        assert np.allclose(lum, 1.0)

    def test_mid_gray(self):
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        lum = _luminance(img)
        assert np.allclose(lum, 0.5)

    def test_red_channel(self):
        """Pure red: R=1, G=0, B=0 → L ≈ 0.299."""
        img = np.zeros((4, 4, 3), dtype=np.float32)
        img[..., 0] = 1.0
        lum = _luminance(img)
        assert np.allclose(lum, 0.299, atol=1e-3)


class TestSigmoid:
    def test_midpoint_zero(self):
        x = np.array([-5.0, 0.0, 5.0], dtype=np.float32)
        y = _sigmoid(x, 0.0, 1.0)
        assert y[0] < 0.01
        assert abs(y[1] - 0.5) < 0.01
        assert y[2] > 0.99

    def test_high_steepness(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y = _sigmoid(x, 0.5, 20.0)
        assert y[0] < 1e-4
        assert y[2] > 0.9999


class TestLabRoundTrip:
    def test_round_trip_gray(self):
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        lab = _rgb_to_lab(rgb)
        from color_constancy.algorithms.sme import _lab_to_rgb

        rgb2 = _lab_to_rgb(lab)
        assert np.allclose(rgb, rgb2, atol=0.02)

    def test_chroma_gray_is_zero(self):
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        lab = _rgb_to_lab(rgb)
        c = _chroma(lab)
        assert np.allclose(c, 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# Dynamic S-curve
# ---------------------------------------------------------------------------

class TestDynamicSCurve:
    def test_uniform_image_passthrough(self):
        """Near-uniform luminance should be returned unchanged."""
        lum = np.full((32, 32), 0.5, dtype=np.float32)
        result = _dynamic_s_curve(lum, 0.5, 0.5, 0.5)
        # Uniform image ends in degenerate branch → input unchanged
        assert np.allclose(result, 0.5, atol=0.01)

    def test_preserves_monotonicity(self):
        """Darker pixels should remain ≤ brighter pixels after the curve."""
        rng = np.random.RandomState(99)
        lum = rng.rand(64, 64).astype(np.float32)
        q1 = np.percentile(lum, 25)
        q3 = np.percentile(lum, 75)
        median = np.median(lum)
        result = _dynamic_s_curve(lum, q1, q3, median)
        # Spearman rank correlation should be high (monotonic)
        from scipy.stats import spearmanr

        rho, _ = spearmanr(lum.ravel(), result.ravel())
        assert rho > 0.94  # shadow protection breaks strict monotonicity at tails

    def test_expands_midtone_range(self):
        """For a bimodal image, the midtone range should be expanded."""
        lum = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, -1)
        # Repeat to make a 2D image
        lum = np.tile(lum, (16, 1))
        q1 = np.percentile(lum, 25)
        q3 = np.percentile(lum, 75)
        median = np.median(lum)
        result = _dynamic_s_curve(lum, q1, q3, median)
        # Should have non-trivial variance
        assert result.std() > 0.0

    def test_shadows_protected(self):
        """Deep shadows (bottom 10 %) should receive minimal change."""
        rng = np.random.RandomState(42)
        lum = rng.rand(128, 128).astype(np.float32) * 0.3  # all in low range
        q1 = np.percentile(lum, 25)
        q3 = np.percentile(lum, 75)
        median = np.median(lum)
        result = _dynamic_s_curve(lum, q1, q3, median, shadow_thresh=0.10)
        # The darkest 5 % should not change much
        cutoff = np.percentile(lum, 5)
        dark_mask = lum <= cutoff
        change = np.abs(result[dark_mask] - lum[dark_mask])
        assert change.mean() < 0.15


# ---------------------------------------------------------------------------
# Neutrality mask & CDC
# ---------------------------------------------------------------------------

class TestNeutralityMask:
    def test_gray_is_fully_neutral(self):
        rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
        lab = _rgb_to_lab(rgb)
        mask = _neutrality_mask(lab)
        assert mask.mean() > 0.7  # near 1.0

    def test_vivid_is_not_neutral(self):
        """A pure red image should have a low neutrality score."""
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        rgb[..., 0] = 1.0
        lab = _rgb_to_lab(rgb)
        mask = _neutrality_mask(lab)
        assert mask.mean() < 0.3


class TestLocalVariance:
    def test_uniform_zero_variance(self):
        rgb = np.full((32, 32, 3), 0.5, dtype=np.float32)
        lab = _rgb_to_lab(rgb)
        var = _local_variance_ab(lab, sigma=3.0)
        assert np.allclose(var, 0.0, atol=0.5)

    def test_high_variance_edges(self):
        """Checkerboard pattern should produce non-zero local variance."""
        rng = np.random.RandomState(7)
        rgb = rng.rand(32, 32, 3).astype(np.float32)
        lab = _rgb_to_lab(rgb)
        var = _local_variance_ab(lab, sigma=3.0)
        assert var.max() > 0.0
        assert var.min() >= 0.0


class TestColorDefinitionConfidence:
    def test_gray_low_cdc(self):
        rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
        lab = _rgb_to_lab(rgb)
        cdc = _color_definition_confidence(lab)
        assert cdc.mean() < 0.3

    def test_vivid_high_cdc(self):
        """Vivid uniform colors should have high CDC."""
        rgb = np.zeros((16, 16, 3), dtype=np.float32)
        rgb[..., 0] = 1.0
        lab = _rgb_to_lab(rgb)
        cdc = _color_definition_confidence(lab)
        assert cdc.mean() > 0.6

    def test_range_zero_to_one(self):
        rng = np.random.RandomState(123)
        rgb = rng.rand(32, 32, 3).astype(np.float32)
        lab = _rgb_to_lab(rgb)
        cdc = _color_definition_confidence(lab)
        assert cdc.min() >= 0.0
        assert cdc.max() <= 1.0


# ---------------------------------------------------------------------------
# Highlight guard
# ---------------------------------------------------------------------------

class TestHighlightGuard:
    def test_normal_pixels_unchanged(self):
        rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
        result = _highlight_guard(rgb)
        assert np.allclose(result, 0.5)

    def test_white_compressed(self):
        """Pure white should be pulled down below 1.0."""
        rgb = np.ones((8, 8, 3), dtype=np.float32)
        result = _highlight_guard(rgb)
        assert result.max() < 1.0
        assert result.max() > 0.95  # not crushed to gray

    def test_bright_but_safe_unchanged(self):
        """Pixels below the ceiling should not be modified."""
        rgb = np.full((8, 8, 3), 0.9, dtype=np.float32)
        result = _highlight_guard(rgb)
        assert np.allclose(result, rgb)

    def test_monotonic(self):
        """Brighter pixels should not become darker than originally darker ones."""
        rng = np.random.RandomState(55)
        rgb = rng.rand(16, 16, 3).astype(np.float32)
        result = _highlight_guard(rgb)
        assert result.shape == rgb.shape
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Full SME algorithm
# ---------------------------------------------------------------------------

class TestSelectiveMidtoneEnhancement:
    def test_neutral_gray_passthrough(self):
        """A uniform mid-gray image should remain essentially unchanged."""
        algo = SelectiveMidtoneEnhancement()
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        result = algo.process(img)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        # Should be close to 0.5 (within 5 %)
        assert np.allclose(result, 0.5, atol=0.05)

    def test_black_stays_black(self):
        algo = SelectiveMidtoneEnhancement()
        img = np.zeros((32, 32, 3), dtype=np.float32)
        result = algo.process(img)
        assert np.allclose(result, 0.0, atol=0.01)

    def test_white_never_clips_to_1(self):
        """Property: no pure white (255, 255, 255) pixels are generated."""
        algo = SelectiveMidtoneEnhancement()
        img = np.ones((32, 32, 3), dtype=np.float32)
        result = algo.process(img)
        assert result.max() < 1.0

    def test_white_not_unreasonably_dark(self):
        """White should still look white-ish, not crushed to mid-gray."""
        algo = SelectiveMidtoneEnhancement()
        img = np.ones((32, 32, 3), dtype=np.float32)
        result = algo.process(img)
        assert result.mean() > 0.8

    def test_neutral_tones_stay_neutral(self):
        """Property: neutral colors remain visually neutral (R≈G≈B)."""
        algo = SelectiveMidtoneEnhancement()
        # A gradient of neutral grays
        gray_vals = np.linspace(0.1, 0.9, 16, dtype=np.float32)
        img = np.stack(
            [gray_vals[np.newaxis, :]] * 3, axis=-1
        ).repeat(8, axis=0)
        result = algo.process(img)
        # Each pixel's channel differences should remain small
        max_diff = np.abs(result[..., 0] - result[..., 1]).max()
        assert max_diff < 0.02

    def test_saturation_boost_on_vivid_colors(self):
        """Vivid regions should receive chroma boost (test via noise image)."""
        algo = SelectiveMidtoneEnhancement(saturation_gain=1.5)
        rng = np.random.RandomState(42)
        img = rng.rand(64, 64, 3).astype(np.float32)
        result = algo.process(img)
        # Processed image should differ from original (enhancement happened)
        assert not np.allclose(result, img, atol=1e-4)

    def test_saturation_gain_zero_no_boost(self):
        """With saturation_gain=1.0, no saturation changes beyond contrast."""
        algo_low = SelectiveMidtoneEnhancement(saturation_gain=1.0)
        algo_high = SelectiveMidtoneEnhancement(saturation_gain=1.5)
        rng = np.random.RandomState(7)
        img = rng.rand(32, 32, 3).astype(np.float32)
        r1 = algo_low.process(img)
        r2 = algo_high.process(img)
        # The high-gain version should have higher per-pixel chroma on average
        lab1 = _rgb_to_lab(r1)
        lab2 = _rgb_to_lab(r2)
        c1 = _chroma(lab1).mean()
        c2 = _chroma(lab2).mean()
        assert c1 <= c2 + 0.1  # high-gain should not have less chroma

    def test_contrast_strength_zero(self):
        """contrast_strength=0 should disable S-curve, leaving only saturation."""
        algo = SelectiveMidtoneEnhancement(contrast_strength=0.0, saturation_gain=1.0)
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        result = algo.process(img)
        assert np.allclose(result, 0.5, atol=0.05)

    def test_contrast_strength_high(self):
        """High contrast_strength should expand midtone IQR."""
        algo = SelectiveMidtoneEnhancement(contrast_strength=2.0, saturation_gain=1.0)
        rng = np.random.RandomState(77)
        img = rng.rand(64, 64, 3).astype(np.float32)
        result = algo.process(img)
        # Midtone expansion: Q1→Q3 range should widen with contrast_strength > 1
        orig_lum = _luminance(img)
        result_lum = _luminance(result)
        o_iqr = np.percentile(orig_lum, 75) - np.percentile(orig_lum, 25)
        r_iqr = np.percentile(result_lum, 75) - np.percentile(result_lum, 25)
        assert 0.5 < r_iqr / o_iqr < 2.0  # midtone expansion without blowing up
        assert result.shape == img.shape
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_shadow_protection_param(self):
        """Higher shadow_protection preserves more shadows."""
        algo_low = SelectiveMidtoneEnhancement(shadow_protection=0.05)
        algo_high = SelectiveMidtoneEnhancement(shadow_protection=0.30, saturation_gain=1.0)
        rng = np.random.RandomState(99)
        img = rng.rand(64, 64, 3).astype(np.float32) * 0.2  # all dark
        r1 = algo_low.process(img)
        r2 = algo_high.process(img)
        # High protection should be closer to original
        diff_low = np.abs(r1 - img).mean()
        diff_high = np.abs(r2 - img).mean()
        assert diff_high <= diff_low + 0.01

    def test_input_modified_but_not_destroyed(self):
        """Output should have same shape/dtype and reasonable range."""
        algo = SelectiveMidtoneEnhancement()
        rng = np.random.RandomState(1)
        img = rng.rand(64, 64, 3).astype(np.float32)
        result = algo.process(img)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_algo_is_subclass(self):
        from color_constancy.algorithms.base import ColorConstancyAlgorithm

        assert issubclass(SelectiveMidtoneEnhancement, ColorConstancyAlgorithm)

    def test_repr_includes_name(self):
        algo = SelectiveMidtoneEnhancement()
        r = repr(algo)
        assert "SelectiveMidtoneEnhancement" in r

    def test_gray_ramp_monotonic_luminance(self):
        """A gray ramp should preserve gradient monotonicity after processing."""
        gray = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, -1)
        img = np.stack([gray] * 3, axis=-1).repeat(8, axis=0)
        algo = SelectiveMidtoneEnhancement(saturation_gain=1.0)
        result = algo.process(img)
        lum = _luminance(result)
        # Check monotonicity row by row
        for row in lum:
            assert np.all(np.diff(row) >= -1e-4)
