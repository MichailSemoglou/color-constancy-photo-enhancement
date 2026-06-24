# Color Constancy Photo Enhancement

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/MichailSemoglou/color-constancy-photo-enhancement/actions/workflows/ci.yml/badge.svg)](https://github.com/MichailSemoglou/color-constancy-photo-enhancement/actions/workflows/ci.yml)

A Python implementation of color constancy algorithms for photo enhancement, based on the comprehensive review by Foster (2011) in _Vision Research_.

## Features

Six color constancy algorithms, a composable pipeline API, quantitative evaluation metrics, and a full CLI:

- **Gray World Assumption**: Corrects color cast by assuming the spatial average of scene reflectances is neutral
- **White Patch / Max-RGB**: Normalises colors based on the brightest surface in the image
- **Von Kries Adaptation**: Applies a diagonal cone-response transformation to simulate chromatic adaptation
- **Retinex Enhancement**: Uses centre-surround processing for local contrast enhancement (Single-Scale Retinex)
- **Spatial Color Correction**: Estimates a per-pixel local illuminant using a Gaussian neighbourhood mean
- **Combined Pipeline**: Sequential Grey World → Von Kries → Retinex for comprehensive color improvement

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
# Combined pipeline (default)
color-constancy-enhance input.jpg --output enhanced.jpg

# Specific algorithm
color-constancy-enhance input.jpg --method gray_world --output gray_world.jpg
color-constancy-enhance input.jpg --method white_patch --output white_patch.jpg
color-constancy-enhance input.jpg --method von_kries --output von_kries.jpg
color-constancy-enhance input.jpg --method retinex --output retinex.jpg
color-constancy-enhance input.jpg --method spatial --output spatial.jpg

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
from color_constancy import AlgorithmPipeline, GrayWorldCorrection, RetinexEnhancement

pipeline = AlgorithmPipeline([
    GrayWorldCorrection(),
    RetinexEnhancement(surround_sigma=20.0, blend_alpha=0.7),
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

# Per-channel statistics and colour cast
stats = color_statistics(enhanced_f)
print(stats["red_cast"], stats["mean_r"])
```

## Methods Explained

### Gray World Assumption

Assumes the spatial average of scene reflectances is achromatic. Estimates the illuminant as the per-channel mean and scales each channel toward a neutral mean brightness (Buchsbaum, 1980). Unreliable for images dominated by a single hue.

### White Patch / Max-RGB

Assumes the brightest surface in the image reflects maximally across all wavelengths. Normalises each channel by its spatial maximum. Unreliable when specular highlights are chromatically coloured.

### Von Kries Adaptation

Implements the von Kries coefficient rule via a diagonal per-channel scaling of cone responses (von Kries, 1902). The illuminant is estimated as a weighted blend of the Grey World and specular-highlight estimates. A configurable `adaptation_strength` parameter blends the correction toward the identity for natural-looking results.

### Retinex Enhancement

Based on Land's Retinex theory (Land & McCann, 1971). Computes a log-domain image and subtracts a Gaussian-smoothed surround (slowly-varying illumination estimate) to enhance local contrast.

**Note:** this is Single-Scale Retinex (SSR) using a single surround `sigma`. Multi-Scale Retinex (MSR), which averages over several scales, is not implemented here (Jobson et al., 1997).

### Spatial Color Correction

Estimates a per-pixel local illuminant using `scipy.ndimage.gaussian_filter` — fully vectorised, no Python loops. Each pixel is corrected toward the global mean relative to its local neighbourhood. Corrections are clipped to prevent over-saturation.

### Combined Pipeline

Sequentially applies Grey World correction, Von Kries adaptation (gentler parameters), and Retinex enhancement.

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

- Buchsbaum, G. (1980). A spatial processor model for object colour perception. _Journal of the Franklin Institute_, 310(1), 1–26.
- von Kries, J. (1902). Theoretische Studien über die Umstimmung des Sehorgans. _Festschrift der Albrecht-Ludwigs-Universität_, 143–158.
- Land, E. H., & McCann, J. J. (1971). Lightness and retinex theory. _Journal of the Optical Society of America_, 61(1), 1–11.
- Jobson, D. J., Rahman, Z., & Woodell, G. A. (1997). A multiscale retinex for bridging the gap between colour images and the human observation of scenes. _IEEE Transactions on Image Processing_, 6(7), 965–976.
- Hordley, S. D., & Finlayson, G. D. (2006). Reevaluation of colour constancy algorithm performance. _Journal of the Optical Society of America A_, 23(5), 1008–1020.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- David H. Foster for the comprehensive review that inspired this implementation
- The computer vision community for advancing color constancy research
- Contributors to OpenCV, NumPy, and SciPy
