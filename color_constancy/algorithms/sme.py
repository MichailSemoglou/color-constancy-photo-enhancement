"""Selective Midtone Enhancement (SME) — perceptually guided contrast & saturation.

Reference
---------
Semoglou, M. (2026). Selective Midtone Enhancement: Dynamic Contrast
Adjustment with Conditional Saturation Control.  Proposed algorithm.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .base import ColorConstancyAlgorithm


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luminance: L = 0.299·R + 0.587·G + 0.114·B."""
    return (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert float [0,1] RGB → float32 CIELAB via OpenCV.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)`` with channels L* ∈ [0, 100], a* ∈ [-128, 128],
        b* ∈ [-128, 128].
    """
    uint8 = (rgb * 255.0).astype(np.uint8)
    lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    return lab


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB → float [0,1] RGB."""
    uint8 = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(uint8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _chroma(lab: np.ndarray) -> np.ndarray:
    """Per-pixel chroma C* = sqrt(a*² + b*²) in CIELAB space.

    OpenCV encodes L*a*b* in uint8 [0, 255] with a* and b* centered at 128.
    We center them before computing chroma.
    """
    a = lab[..., 1].astype(np.float32) - 128.0
    b = lab[..., 2].astype(np.float32) - 128.0
    return np.sqrt(a * a + b * b)


def _sigmoid(x: np.ndarray | float, midpoint: float, steepness: float) -> np.ndarray:
    """Logistic sigmoid: 1 / (1 + exp(-steepness * (x - midpoint)))."""
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def _dynamic_s_curve(
    lum: np.ndarray,
    q1: float,
    q3: float,
    median: float,
    shadow_thresh: float = 0.05,
    highlight_thresh: float = 0.05,
    shadow_floor: float = 0.005,
) -> np.ndarray:
    """Build a dynamic S-curve that expands the interquartile range.

    Only the midtone band [Q1 ... Q3] receives expansion; shadows and
    highlights are blended back toward identity via soft transition zones.

    Parameters
    ----------
    lum:
        Luminance channel, float32, shape ``(H, W)``, values in [0, 1].
    q1, q3:
        Lower and upper quartiles of the luminance histogram.
    median:
        Median luminance.
    shadow_thresh:
        Fraction of L* range at the bottom blended to identity.
    highlight_thresh:
        Fraction of L* range at the top blended to identity.
    shadow_floor:
        Luminance value that defines the absolute shadow floor.

    Returns
    -------
    np.ndarray
        S-curve-mapped luminance, same shape and dtype.
    """
    iqr = max(q3 - q1, 0.01)
    k = 4.0 / iqr
    mid = median

    enhanced = _sigmoid(lum, mid, k)

    s_q1 = _sigmoid(q1, mid, k)
    s_q3 = _sigmoid(q3, mid, k)
    denom = s_q3 - s_q1
    if denom > 1e-6:
        enhanced = q1 + (q3 - q1) * (enhanced - s_q1) / denom
    else:
        return lum.astype(np.float32)

    enhanced = np.clip(enhanced, 0.0, 1.0)

    # ---- Shadow blend zone: below Q1, fade toward identity ----
    shadow_cutoff = np.percentile(lum, shadow_thresh * 100.0)
    # Soft transition: full identity at lum=0, no blend at lum >= q1
    shadow_weight = np.clip((q1 - lum) / max(q1 - shadow_cutoff, 0.001), 0.0, 1.0)
    floor_mask = (lum < shadow_floor).astype(np.float32)
    shadow_weight = np.maximum(shadow_weight, floor_mask)

    # ---- Highlight blend zone: above Q3, fade toward identity ----
    highlight_start = np.percentile(lum, (1.0 - highlight_thresh) * 100.0)
    # Soft transition: full identity at lum=1.0, no blend at lum <= q3
    highlight_weight = np.clip(
        (lum - q3) / max(highlight_start - q3, 0.001), 0.0, 1.0
    )

    total_weight = np.maximum(shadow_weight, highlight_weight)
    result = total_weight * lum + (1.0 - total_weight) * enhanced
    return result.astype(np.float32)


def _neutrality_mask(lab: np.ndarray, chroma_thresh: float = 8.0) -> np.ndarray:
    """Compute a soft mask that is 1.0 for neutral (low-chroma) pixels.

    Parameters
    ----------
    lab:
        CIELAB image, shape ``(H, W, 3)``.
    chroma_thresh:
        Chroma value below which a pixel is considered near-neutral.

    Returns
    -------
    np.ndarray
        Mask shape ``(H, W)``, values in [0, 1]. 1 = fully neutral.
    """
    c = _chroma(lab)
    # Smooth transition: fully neutral below thresh/2, fully saturated above thresh
    return (1.0 - _sigmoid(c, chroma_thresh / 2.0, 0.5)).astype(np.float32)


def _local_variance_ab(
    lab: np.ndarray, sigma: float = 3.0
) -> np.ndarray:
    """Per-pixel local variance of a* and b* (Gaussian-weighted neighborhood).

    Parameters
    ----------
    lab:
        CIELAB image.
    sigma:
        Gaussian sigma in pixels for the neighborhood window.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, mean local variance of a* and b*.
    """
    a = lab[..., 1].astype(np.float32)
    b = lab[..., 2].astype(np.float32)

    # Local mean
    a_mean = gaussian_filter(a, sigma, mode="reflect")
    b_mean = gaussian_filter(b, sigma, mode="reflect")

    # Local variance: E[X²] - E[X]²
    a_var = gaussian_filter(a * a, sigma, mode="reflect") - a_mean * a_mean
    b_var = gaussian_filter(b * b, sigma, mode="reflect") - b_mean * b_mean

    return np.maximum((a_var + b_var) / 2.0, 0.0)


def _color_definition_confidence(
    lab: np.ndarray,
    chroma_thresh: float = 12.0,
    chroma_steepness: float = 0.3,
    var_sigma: float = 3.0,
    var_scale: float = 100.0,
) -> np.ndarray:
    """Compute Color Definition Confidence (CDC) for each pixel.

    CDC ∈ [0, 1].  High values indicate vivid, locally-consistent colors.
    Low values indicate muted/ambiguous/near-neutral regions.

    Parameters
    ----------
    lab:
        CIELAB image.
    chroma_thresh:
        Chroma midpoint for the chroma-sigmoid component.
    chroma_steepness:
        Steepness of the chroma-sigmoid transition.
    var_sigma:
        Gaussian sigma for local-variance computation.
    var_scale:
        Normalization scale for the local-variance penalty term.

    Returns
    -------
    np.ndarray
        CDC map, shape ``(H, W)``, values in [0, 1].
    """
    c = _chroma(lab)
    local_var = _local_variance_ab(lab, sigma=var_sigma)

    # Chroma confidence: high when chroma is substantial
    chroma_conf = _sigmoid(c, chroma_thresh, chroma_steepness)

    # Color uniformity penalty: low variance → high confidence
    uniformity_penalty = 1.0 - np.clip(local_var / var_scale, 0.0, 1.0)

    return (chroma_conf * uniformity_penalty).astype(np.float32)


def _highlight_guard(
    rgb: np.ndarray, ceiling: float = 250.0 / 255.0, decay_rate: float = 60.0
) -> np.ndarray:
    """Soft-compress highlights so no channel reaches 255.

    For pixels where max(R,G,B) > ceiling, apply a soft compression that
    asymptotically approaches 1.0 (255/255) without ever reaching it.

    Parameters
    ----------
    rgb:
        Float32 RGB, shape ``(H, W, 3)``, values in [0, 1].
    ceiling:
        Normalized safe ceiling (default: 250/255 ≈ 0.9804).
    decay_rate:
        Controls how quickly compression kicks in above the ceiling.

    Returns
    -------
    np.ndarray
        Highlight-guarded RGB, same shape and dtype.
    """
    vmax = rgb.max(axis=-1).astype(np.float32)  # (H, W)
    over_mask = vmax > ceiling

    if not over_mask.any():
        return rgb

    excess = (vmax - ceiling) * 255.0  # back to [0, 255] scale
    # Soft compression: ceiling + (1-ceiling) * (1 - exp(-excess/decay_rate))
    # When excess=0, target=ceiling.  As excess→∞, target→1.0.
    target = ceiling + (1.0 - ceiling) * (1.0 - np.exp(-excess / decay_rate))
    scale = np.ones_like(vmax)
    safe = vmax > 1e-6
    scale[safe] = target[safe] / vmax[safe]

    result = rgb.copy()
    scale_3ch = np.stack([scale] * 3, axis=-1)
    # Only modify pixels above the ceiling
    result = np.where(scale_3ch < 1.0, result * scale_3ch, result)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _auto_derive_params(
    lab: np.ndarray,
    L: np.ndarray,
) -> tuple[float, float]:
    """Derive contrast_strength and saturation_gain from image statistics.

    Strategy
    -------
    *contrast_strength*: inversely proportional to luminance IQR spread.
    A flat image (narrow IQR) gets a stronger S-curve; an already
    high-contrast image gets a gentle touch.

    *saturation_gain*: inversely proportional to mean chroma.  A muted
    image gets more saturation boost; a vivid image gets less.

    Both parameters are clamped to safe perceptual ranges so the algorithm
    never over- or under-enhances, regardless of input.
    """
    # ---- Contrast strength from IQR spread ----
    iqr_spread = float(np.percentile(L, 75.0) - np.percentile(L, 25.0))
    q3_lum = float(np.percentile(L, 75.0))
    q1_lum = float(np.percentile(L, 25.0))

    # Detect near-black or near-white images where IQR expansion would be
    # pathological.  If the image is mostly in the bottom or top 10%, reduce
    # contrast strength to avoid over-stretching a tiny luminance band.
    if q3_lum < 0.10 or q1_lum > 0.90:
        contrast_strength = 0.3  # minimal touch for extreme key images
    else:
        # Map: narrow IQR (~0.05) → strength ≈ 1.8, wide IQR (~0.60) → strength ≈ 0.4
        contrast_strength = float(np.clip(1.8 - 2.4 * iqr_spread, 0.3, 1.5))

    # ---- Saturation gain from chroma distribution ----
    c = _chroma(lab)
    # Use the 75th percentile for a more robust "colorfulness" estimate
    chroma_p75 = float(np.percentile(c, 75.0))
    # Mapping (empirically tuned):
    #   chroma_p75 = 5   → gain ≈ 1.35 (very muted, strong boost)
    #   chroma_p75 = 20  → gain ≈ 1.18 (moderate)
    #   chroma_p75 = 50  → gain ≈ 1.05 (vivid, minimal boost)
    saturation_gain = float(np.clip(1.40 - 0.007 * chroma_p75, 1.03, 1.35))

    return contrast_strength, saturation_gain


class SelectiveMidtoneEnhancement(ColorConstancyAlgorithm):
    """Selective Midtone Enhancement (SME).

    A perceptually-guided enhancement algorithm that applies dynamic S-curve
    contrast adjustment to the luminance channel with shadow and neutral-tone
    protection, followed by conditional saturation enhancement in CIELAB space,
    and a final highlight-preservation guard.

    Parameters
    ----------
    auto:
        When ``True`` (default), ``contrast_strength`` and ``saturation_gain``
        are derived per-image from luminance spread and mean chroma.
        Explicit values passed alongside ``auto=True`` are ignored.
    contrast_strength:
        Overall strength of the contrast adjustment.  1.0 = full S-curve,
        0.0 = identity.  Values > 1.0 amplify the effect.
        Only used when ``auto=False``.
    saturation_gain:
        Maximum multiplicative gain applied to the chroma channels for
        high-CDC pixels.  Default 1.25 (25 % boost).
        Only used when ``auto=False``.
    shadow_protection:
        Fraction of the luminance range (from the bottom) that receives
        minimal-to-zero contrast boost.  Default 0.10 (bottom 10 %).
    chroma_threshold:
        Midpoint of the chroma sigmoid in CDC.  Lower values make the
        saturation boost engage at lower saturation levels.
    cdc_threshold:
        CDC value above which saturation gain transitions from near-zero
        to full.  Default 0.5.
    """

    def __init__(
        self,
        auto: bool = True,
        contrast_strength: float = 1.0,
        saturation_gain: float = 1.25,
        shadow_protection: float = 0.10,
        highlight_protection: float = 0.10,
        chroma_threshold: float = 12.0,
        cdc_threshold: float = 0.5,
    ):
        self.auto = bool(auto)
        self.contrast_strength = float(contrast_strength)
        self.saturation_gain = float(saturation_gain)
        self.shadow_protection = float(shadow_protection)
        self.highlight_protection = float(highlight_protection)
        self.chroma_threshold = float(chroma_threshold)
        self.cdc_threshold = float(cdc_threshold)

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply Selective Midtone Enhancement.

        Parameters
        ----------
        image:
            Float32 RGB, shape ``(H, W, 3)``, values in [0, 1].

        Returns
        -------
        np.ndarray
            Enhanced image, same shape and dtype.
        """
        rgb = image.astype(np.float32)

        # ---- Step 0: Convert to CIELAB once (used for neutrality + saturation) ----
        lab = _rgb_to_lab(rgb)

        # ---- Step 0b: Auto-derive parameters if enabled ----
        L = lab[..., 0].astype(np.float32) / 255.0  # OpenCV L* ∈ [0, 255]; normalize to [0, 1]
        if self.auto:
            contrast_strength, saturation_gain = _auto_derive_params(lab, L)
        else:
            contrast_strength = self.contrast_strength
            saturation_gain = self.saturation_gain

        # ---- Step 1: Dynamic S-curve contrast on L* channel ----
        q1 = np.percentile(L, 25.0)
        q3 = np.percentile(L, 75.0)
        median = np.median(L)

        enhanced_L = _dynamic_s_curve(
            L, q1, q3, median,
            shadow_thresh=self.shadow_protection,
            highlight_thresh=self.highlight_protection,
        )

        # Blend S-curve result with original based on neutrality mask
        neutral = _neutrality_mask(lab)
        enhanced_L = neutral * L + (1.0 - neutral) * enhanced_L

        # Blend with original based on contrast_strength
        if abs(contrast_strength - 1.0) > 1e-6:
            enhanced_L = (
                (1.0 - contrast_strength) * L
                + contrast_strength * enhanced_L
            )

        # Apply the modified L* and convert back
        lab_contrast = lab.copy()
        lab_contrast[..., 0] = np.clip(enhanced_L * 255.0, 0.0, 255.0)  # back to OpenCV [0, 255]
        cdc = _color_definition_confidence(
            lab_contrast,
            chroma_thresh=self.chroma_threshold,
        )
        # Gain map: sigmoid transition from near-zero to saturation_gain
        gain_map = 1.0 + (saturation_gain - 1.0) * _sigmoid(
            cdc, self.cdc_threshold, 15.0
        )

        # Center a*/b* around 128 (OpenCV's encoding), apply gain, then re-center
        a = lab_contrast[..., 1].astype(np.float32)
        b = lab_contrast[..., 2].astype(np.float32)
        a_centered = a - 128.0
        b_centered = b - 128.0
        a_boosted = np.clip(a_centered * gain_map + 128.0, 0.0, 255.0)
        b_boosted = np.clip(b_centered * gain_map + 128.0, 0.0, 255.0)
        lab_boosted = lab_contrast.copy()
        lab_boosted[..., 1] = a_boosted
        lab_boosted[..., 2] = b_boosted

        rgb_saturated = _lab_to_rgb(lab_boosted)

        # ---- Step 3: Highlight preservation guard ----
        result = _highlight_guard(rgb_saturated)

        return result.astype(np.float32)
