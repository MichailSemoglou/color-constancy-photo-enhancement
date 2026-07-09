# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] – 2026-07-09

### Added

- **Selective Midtone Enhancement (SME)**: novel auto-adaptive enhancement algorithm with three-stage pipeline:
  - IQR-driven adaptive S-curve contrast with shadow, highlight, and neutral-tone protection
  - Conditional saturation boost gated by a novel Color Definition Confidence (CDC) metric
  - Asymptotic highlight preservation guard (soft compression above 250/255)
- Auto-adaptive mode derives `contrast_strength` and `saturation_gain` per-image from luminance spread and chroma distribution
- 6 tunable parameters via `--param k=v` for manual control: `contrast_strength`, `saturation_gain`, `shadow_protection`, `highlight_protection`, `chroma_threshold`, `cdc_threshold`
- 37 unit tests; validated on 50-image corpus (0/50 clipped, max pixel = 250)
- Near-black image guard prevents pathological S-curve expansion on extreme low-key images
- New `sme` value for `--method`

## [1.2.0] – 2026-07-09

### Added

- **Multi-Scale Retinex (MSR)**: new `MultiScaleRetinex` class averaging SSR outputs at three scales (15, 80, 250) for balanced dynamic range and tonal rendition (Jobson et al., 1997).
- **MSRCR (Multi-Scale Retinex with Color Restoration)**: new `MSRCR` class with configurable gain/bias for vivid output without desaturation.
- **Benchmark harness** (`color_constancy.benchmark`): CLI (`color-constancy-benchmark`) + API for evaluating algorithms on standard CSV datasets with angular-error statistics (mean, median, trimean, best-25%, worst-5%).
- **Per-algorithm CLI parameters**: `--sigma`, `--sigmas`, `--blend-alpha`, `--adaptation-strength`, `--correction-strength`, `--cr-gain`, `--cr-bias`, `--param`, `--msrcr`.
- **Named presets** (`--preset night`, `indoor_tungsten`, `sunset`, `high_contrast`, `vivid`, `subtle`) for quick scenario-specific configuration.
- New `msr` and `msrcr` values for `--method`.

### Changed

- **Default combined pipeline now uses MSRCR** instead of SSR for higher-quality output.

## [1.1.2] – 2026-07-09

### Fixed

- Resolved PyPI publishing collision after history rewrite.

## [1.1.1] – 2026-07-08

### Fixed

- `visualize_illuminant()` now accepts a `show` parameter (matching `display_comparison`) so it can run safely in headless/server environments.
- Both `display_comparison()` and `visualize_illuminant()` now always close their matplotlib figures after display, preventing resource leaks on repeated calls.

## [1.1.0] – 2026-06-24

### Added

- Initial public release with Grey World, White Patch, Von Kries, Single-Scale Retinex, and Spatial Color Correction algorithms.
- Combined pipeline (Grey World → Von Kries → Retinex) for general-purpose enhancement.
- CLI entry point (`color-constancy-enhance`).
- Metrics: angular error, PSNR, SSIM, per-channel color statistics.
- Backward-compatible `ColorConstancyEnhancer` facade class.
