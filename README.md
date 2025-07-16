# Color Constancy Photo Enhancement

A Python implementation of color constancy algorithms for photo enhancement, based on the comprehensive review by Foster (2011) in *Vision Research*.

## Features

This tool implements several color constancy algorithms:

- **Gray World Assumption**: Corrects color cast by assuming spatial average of scene reflectances is neutral
- **White Patch/Max RGB**: Normalizes colors based on the brightest surface in the image
- **Von Kries Adaptation**: Applies diagonal transformation to simulate cone adaptation
- **Retinex Enhancement**: Uses center-surround processing for local contrast enhancement
- **Spatial Color Correction**: Implements local adaptation based on spatial cone-excitation ratios
- **Combined Method**: Sequential application of multiple algorithms for optimal results

## Installation

```bash
git clone https://github.com/MichailSemoglou/color-constancy-enhancement.git
cd color-constancy-enhancement
pip install -r requirements.txt
```

## Usage

### Basic Enhancement

```bash
python color_constancy_enhancer.py input.jpg --output enhanced.jpg
```

### Specific Methods

```bash
# Gray World correction
python color_constancy_enhancer.py input.jpg --method gray_world --output gray_world.jpg

# White Patch correction
python color_constancy_enhancer.py input.jpg --method white_patch --output white_patch.jpg

# Von Kries adaptation
python color_constancy_enhancer.py input.jpg --method von_kries --output von_kries.jpg

# Retinex enhancement
python color_constancy_enhancer.py input.jpg --method retinex --output retinex.jpg

# Spatial correction
python color_constancy_enhancer.py input.jpg --method spatial --output spatial.jpg

# Combined methods (default)
python color_constancy_enhancer.py input.jpg --method combined --output combined.jpg
```

### Display Results

```bash
# Show before/after comparison
python color_constancy_enhancer.py input.jpg --show

# Save comparison image
python color_constancy_enhancer.py input.jpg --comparison comparison.jpg --show

# Show color statistics
python color_constancy_enhancer.py input.jpg --stats
```

### Complete Example

```bash
python color_constancy_enhancer.py photo.jpg \
  --method combined \
  --output enhanced_photo.jpg \
  --comparison before_after.jpg \
  --show \
  --stats
```

## API Usage

```python
from color_constancy_enhancer import ColorConstancyEnhancer

# Create enhancer
enhancer = ColorConstancyEnhancer()

# Enhance image
enhanced = enhancer.enhance_image('input.jpg', method='combined', output_path='output.jpg')

# Analyze color statistics
stats = enhancer.analyze_color_statistics(enhanced)

# Display results
enhancer.display_results()
```

## Methods Explained

### Gray World Assumption
Assumes that the average reflectance of surfaces in the scene is achromatic. This method estimates the illuminant as the spatial average of the image and corrects the color cast accordingly.

### White Patch/Max RGB
Assumes that the brightest surface in the image reflects maximally across all wavelengths. This method normalizes each color channel by its maximum value.

### Von Kries Adaptation
Implements the von Kries coefficient rule through diagonal transformation of cone responses. This simulates how the human visual system adapts to different illuminants.

### Retinex Enhancement
Based on Land’s Retinex theory, this method uses center-surround processing to enhance local contrast while maintaining color relationships.

### Spatial Color Correction
Implements local adaptation based on spatial cone-excitation ratios, processing the image in overlapping windows for spatially-varying correction.

### Combined Method
Sequentially applies Gray World correction, Von Kries adaptation, and Retinex enhancement for comprehensive color improvement.

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- SciPy
- Matplotlib
- scikit-learn

## Scientific Background

This implementation is based on the comprehensive review:

Foster, D. H. (2011). Color constancy. *Vision Research*, 51(7), 674-700.

The algorithms implement key concepts from 25 years of color constancy research, including:
- Illuminant estimation techniques
- Cone adaptation mechanisms
- Spatial color processing
- Natural scene statistics

## License

MIT License – see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- David H. Foster for the comprehensive review that inspired this implementation
- The computer vision community for advancing color constancy research
- Contributors to OpenCV, NumPy, and SciPy libraries
