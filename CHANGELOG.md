# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.1] – 2026-09-04

### Added

- New `--cr-alpha` and `--cr-beta` CLI flags for MSRCR color restoration control.
- Tests for the backward-compatible facade, the visualization helpers, and the benchmark CLI; CI now enforces a minimum coverage of 85%.

### Changed

- **Combined pipeline** is now Gray World → MSRCR. The intermediate Von Kries stage was removed because its measured effect after Gray World was negligible (mean pixel change of about 0.003 in [0, 1]). Affects `build_combined_pipeline()` and `--method combined`.
- **Presets** `night`, `sunset`, and `vivid` now tune `cr_alpha` (150, 80, and 200 respectively) instead of `cr_gain`/`cr_bias`, which the output percentile normalization absorbs.

### Fixed

- **SME manual parameter control via the CLI** now works as documented: `--param` values for `auto` and `highlight_protection` are forwarded to `SelectiveMidtoneEnhancement`. Previously `auto` always stayed `True`, so manual `contrast_strength` and `saturation_gain` were silently ignored.
- **Repeated `--param` flags** now merge instead of keeping only the last one.
- **MSRCR color restoration** uses the canonical Jobson et al. (1997) parameterization: distinct inner gain `cr_alpha` (125.0) and outer gain `cr_beta` (46.0), plus the display gain/offset `cr_gain` (192.0) and `cr_bias` (-30.0). The previous implementation reused one constant in both gain roles and clipped the restoration factor, pinning nearly all of its entries at the clip value and reducing it to a near-constant rescaling.
- **Benchmark harness** no longer scores algorithms that do not expose `estimate_illuminant()` by angular error; the mean of an enhanced output is not an illuminant estimate. Such methods are excluded with a warning, and the default benchmark suite is now GrayWorld, WhitePatch, and VonKries.
- **SSIM** uses the canonical 11×11 Gaussian window (sigma = 1.5) specified by Wang et al. (2004) instead of a uniform window. SSIM values change slightly as a result.

### Documentation

- README feature list now distinguishes the eight algorithms from the pipeline API, metrics, CLI, presets, and benchmark harness.
- README examples updated for the simplified combined pipeline, MSRCR color restoration via `cr_alpha`, SME manual mode (manual gains require `auto=false`), and benchmarking with illuminant estimators.
- Facade docstring now lists all nine methods accepted by `ColorConstancyEnhancer.enhance_image()`.

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
