# Color Constancy Photo Enhancement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MichailSemoglou/color-constancy-photo-enhancement/actions/workflows/ci.yml/badge.svg)](https://github.com/MichailSemoglou/color-constancy-photo-enhancement/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/color-constancy-enhancement)](https://pypi.org/project/color-constancy-enhancement/)
[![PyPI downloads](https://img.shields.io/pypi/dm/color-constancy-enhancement)](https://pypi.org/project/color-constancy-enhancement/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/color-constancy-enhancement)](https://pypi.org/project/color-constancy-enhancement/)

A Python implementation of color constancy algorithms for photo enhancement, based on the comprehensive review by Foster (2011) in _Vision Research_.

## Features

Nine color constancy algorithms, a composable pipeline API, quantitative evaluation metrics, a full CLI, presets for common scenarios, and a benchmark harness for datasets:

- **Gray World Assumption**: Corrects color cast by assuming the spatial average of scene reflectances is neutral
- **White Patch / Max-RGB**: Normalizes colors based on the brightest surface in the image
- **Von Kries Adaptation**: Applies a diagonal cone-response transformation to simulate chromatic adaptation
- **Retinex Enhancement (SSR)**: Uses center-surround processing for local contrast enhancement (Single-Scale Retinex)
- **Multi-Scale Retinex (MSR)**: Averages SSR at three scales (15, 80, 250) for balanced dynamic range (Jobson et al., 1997)
- **MSRCR**: MSR with color restoration for vivid output without desaturation (Jobson et al., 1997)
- **Spatial Color Correction**: Estimates a per-pixel local illuminant using a vectorized Gaussian neighborhood mean
- **Combined Pipeline**: Sequential Grey World → Von Kries → MSRCR for comprehensive color improvement
- **Benchmark Harness**: Evaluate algorithms on standard CSV datasets with angular-error statistics
- **Named Presets**: `night`, `indoor_tungsten`, `sunset`, `high_contrast`, `vivid`, `subtle` for quick configuration

## Installation

```bash
git clone https://github.com/MichailSemoglou/color-constancy-photo-enhancement.git
cd color-constancy-photo-enhancement
```

Create an isolated environment (recommended):

```bash
# macOS / Linux
python -m venv .venv && source .venv/bin/activate

# Windows
python -m venv .venv && .venv\Scripts\activate
```

Then install:

```bash
pip install -e .
```

To install with development tools (pytest, ruff, mypy):

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Combined pipeline (default) — Grey World → Von Kries → MSRCR
color-constancy-enhance input.jpg --output enhanced.jpg

# Single-Scale Retinex
color-constancy-enhance input.jpg --method retinex --output retinex.jpg
# Tune SSR sigma
color-constancy-enhance input.jpg --method retinex --sigma 30.0 --output retinex.jpg

# Multi-Scale Retinex
color-constancy-enhance input.jpg --method msr --output msr.jpg
# Custom MSR scales
color-constancy-enhance input.jpg --method msr --sigmas 10,60,200 --output msr.jpg

# MSRCR (Multi-Scale Retinex with Color Restoration)
color-constancy-enhance input.jpg --method msrcr --output msrcr.jpg
# Vivid MSRCR
color-constancy-enhance input.jpg --method msrcr --cr-gain 200 --cr-bias -60 --output vivid.jpg

# Other single algorithms
color-constancy-enhance input.jpg --method gray_world --output gray_world.jpg
color-constancy-enhance input.jpg --method white_patch --output white_patch.jpg
color-constancy-enhance input.jpg --method von_kries --output von_kries.jpg
# Tune Von Kries
color-constancy-enhance input.jpg --method von_kries --adaptation-strength 0.8 --output vk.jpg
color-constancy-enhance input.jpg --method spatial --output spatial.jpg

# Named presets for common scenarios
color-constancy-enhance input.jpg --preset night --output night.jpg
color-constancy-enhance input.jpg --preset indoor_tungsten --output indoor.jpg
color-constancy-enhance input.jpg --preset sunset --output sunset.jpg
color-constancy-enhance input.jpg --preset high_contrast --output hc.jpg
color-constancy-enhance input.jpg --preset vivid --output vivid.jpg

# Side-by-side comparison
color-constancy-enhance input.jpg --show
color-constancy-enhance input.jpg --comparison before_after.jpg --show

# Channel statistics
color-constancy-enhance input.jpg --stats

# Illuminant diagnostic chart (for algorithms with estimate_illuminant())
color-constancy-enhance input.jpg --method gray_world --debug

# All options at once
color-constancy-enhance photo.jpg \
  --method combined \
  --output enhanced.jpg \
  --comparison before_after.jpg \
  --show \
  --stats
```

### Benchmarking

```bash
# Run all algorithms against a CSV dataset
color-constancy-benchmark dataset.csv --image-dir ./images

# Subset of algorithms, Markdown output
color-constancy-benchmark dataset.csv --image-dir ./images --method GrayWorld WhitePatch --format markdown

# Custom column names, save to file
color-constancy-benchmark dataset.csv \
  --image-dir ./images \
  --illuminant-cols r_gt,g_gt,b_gt \
  --image-col img_name \
  --format csv \
  --output results.csv
```

### Python API

#### Composable pipeline (recommended)

```python
from color_constancy import build_combined_pipeline, load_image, save_image
import numpy as np

image = load_image("photo.jpg") # uint8 RGB
pipeline = build_combined_pipeline()
enhanced = (pipeline.process(image.astype("float32") / 255.0) * 255).astype("uint8")
save_image(enhanced, "enhanced.jpg")
```

#### Individual algorithms

```python
from color_constancy import GrayWorldCorrection, VonKriesAdaptation
from color_constancy import load_image, save_image
import numpy as np

image = load_image("photo.jpg")
img_f = image.astype("float32") / 255.0

# Apply algorithms individually or in any order
gw = GrayWorldCorrection()
vk = VonKriesAdaptation()

corrected = vk.process(gw.process(img_f))
save_image((corrected * 255).astype("uint8"), "corrected.jpg")
```

#### Custom pipeline

```python
from color_constancy import AlgorithmPipeline, GrayWorldCorrection, MSRCR, MultiScaleRetinex, RetinexEnhancement

# SSR pipeline
pipeline_ssr = AlgorithmPipeline([
    GrayWorldCorrection(),
    RetinexEnhancement(surround_sigma=20.0, blend_alpha=0.7),
])

# MSR pipeline
pipeline_msr = AlgorithmPipeline([
    GrayWorldCorrection(),
    MultiScaleRetinex(sigmas=(15.0, 80.0, 250.0), blend_alpha=0.7),
])

# MSRCR pipeline (what build_combined_pipeline() returns)
pipeline_msrcr = AlgorithmPipeline([
    GrayWorldCorrection(),
    MSRCR(sigmas=(15.0, 80.0, 250.0), blend_alpha=0.7, cr_gain=125.0, cr_bias=-46.0),
])
```

#### Backward-compatible facade

The original single-class interface still works:

```python
from color_constancy_enhancer import ColorConstancyEnhancer

enhancer = ColorConstancyEnhancer()
enhanced = enhancer.enhance_image("input.jpg", method="combined", output_path="output.jpg")
stats = enhancer.analyze_color_statistics(enhanced)
enhancer.display_results()
```

## Evaluation Metrics

The `color_constancy.metrics` module provides standard quantitative metrics for academic evaluation:

```python
from color_constancy.metrics import angular_error, psnr, ssim, color_statistics
import numpy as np

# Angular error between illuminant estimates (degrees; lower is better)
estimated   = np.array([0.30, 0.33, 0.37])
ground_truth = np.array([0.33, 0.33, 0.34])
err = angular_error(estimated, ground_truth)

# PSNR and SSIM between float32 images in [0, 1]
score_psnr = psnr(reference_f, enhanced_f)
score_ssim = ssim(reference_f, enhanced_f)

# Per-channel statistics and color cast
stats = color_statistics(enhanced_f)
print(stats["red_cast"], stats["mean_r"])
```

## Benchmark API

The `color_constancy.benchmark` module provides a dataset evaluation harness:

```python
from color_constancy.benchmark import load_dataset, run_benchmark
from color_constancy.algorithms import GrayWorldCorrection, MSRCR

entries = load_dataset("dataset.csv", image_dir="./images")
results = run_benchmark(entries, algorithms={
    "GrayWorld": GrayWorldCorrection(),
    "MSRCR": MSRCR(),
})

print(results.summary_table())
#  Algorithm               Mean   Median  Trimean  Best25   Worst5    N
#  ---------------------------------------------------------------------
#  GrayWorld              4.32     3.87     3.91    1.23    12.45    568
#  MSRCR                  5.10     4.45     4.58    1.58    14.23    568

# Export as Markdown or CSV
print(results.to_markdown())
print(results.to_csv())
```

## Methods Explained

### Gray World Assumption

Assumes the spatial average of scene reflectances is achromatic. Estimates the illuminant as the per-channel mean and scales each channel toward a neutral mean brightness (Buchsbaum, 1980). Unreliable for images dominated by a single hue.

### White Patch / Max-RGB

Assumes the brightest surface in the image reflects maximally across all wavelengths. Normalizes each channel by its spatial maximum. Unreliable when specular highlights are chromatically colored.

### Von Kries Adaptation

Implements the von Kries coefficient rule via a diagonal per-channel scaling of cone responses (von Kries, 1902). The illuminant is estimated as a weighted blend of the Grey World and specular-highlight estimates. A configurable `adaptation_strength` parameter blends the correction toward the identity for natural-looking results.

### Retinex Enhancement (SSR)

Based on Land's Retinex theory (Land & McCann, 1971). Computes a log-domain image and subtracts a Gaussian-smoothed surround (slowly-varying illumination estimate) to enhance local contrast.

**Note:** this is Single-Scale Retinex (SSR) using a single surround `sigma`. For more advanced results, use the MSR or MSRCR methods.

### Multi-Scale Retinex (MSR)

Averages SSR outputs at three surround scales — typically 15, 80, and 250 — to balance dynamic range compression and tonal rendition (Jobson et al., 1997). By combining a small, medium, and large scale, MSR avoids the single-scale trade-off between local contrast and color fidelity.

```python
from color_constancy import MultiScaleRetinex

msr = MultiScaleRetinex(sigmas=(15.0, 80.0, 250.0), blend_alpha=0.7)
# Or via CLI: color-constancy-enhance input.jpg --method msr
```

### MSRCR (Multi-Scale Retinex with Color Restoration)

Extends MSR with a per-channel color restoration step that compensates for the desaturation MSR can introduce (Jobson et al., 1997). Configurable `cr_gain` and `cr_bias` control color vividness.

```python
from color_constancy import MSRCR

msrcr = MSRCR(sigmas=(15.0, 80.0, 250.0), blend_alpha=0.7, cr_gain=125.0, cr_bias=-46.0)
# Or via CLI: color-constancy-enhance input.jpg --method msrcr --cr-gain 200
```

The default **combined pipeline** (Grey World → Von Kries → MSRCR) uses MSRCR internally.

### Spatial Color Correction

Estimates a per-pixel local illuminant using `scipy.ndimage.gaussian_filter` — fully vectorised, no Python loops. Each pixel is corrected toward the global mean relative to its local neighborhood. Corrections are clipped to prevent over-saturation.

### Combined Pipeline

Sequentially applies Grey World correction, Von Kries adaptation (gentler parameters), and MSRCR for comprehensive color correction with vivid, well-balanced output.

## Running Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ --cov=color_constancy --cov-report=term-missing
```

## Requirements

- Python 3.9+
- OpenCV (`opencv-python >= 4.8`)
- NumPy (`>= 1.24`)
- SciPy (`>= 1.10`)
- Matplotlib (`>= 3.7`)

## Scientific Background

This implementation is based on the comprehensive review:

Foster, D. H. (2011). Color constancy. _Vision Research_, 51(7), 674–700. https://doi.org/10.1016/j.visres.2010.09.006

Additional references:

- Buchsbaum, G. (1980). A spatial processor model for object color perception. _Journal of the Franklin Institute_, 310(1), 1–26.
- von Kries, J. (1902). Theoretische Studien über die Umstimmung des Sehorgans. _Festschrift der Albrecht-Ludwigs-Universität_, 143–158.
- Land, E. H., & McCann, J. J. (1971). Lightness and retinex theory. _Journal of the Optical Society of America_, 61(1), 1–11.
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex for bridging the gap between color images and the human observation of scenes. _IEEE Transactions on Image Processing_, 6(7), 965–976.
- Hordley, S. D., & Finlayson, G. D. (2006). Reevaluation of color constancy algorithm performance. _Journal of the Optical Society of America A_, 23(5), 1008–1020.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- David H. Foster for the comprehensive review that inspired this implementation
- The computer vision community for advancing color constancy research
- Contributors to OpenCV, NumPy, and SciPy
