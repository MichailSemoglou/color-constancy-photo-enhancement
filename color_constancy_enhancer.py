#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class ColorConstancyEnhancer:
    
    def __init__(self) -> None:
        self.original_image: Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        self.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.original_image
    
    def gray_world_correction(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        channel_means = np.mean(img_float, axis=(0, 1))
        global_mean = np.mean(channel_means)
        correction_factors = global_mean / channel_means
        corrected = img_float * correction_factors
        return (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    
    def white_patch_correction(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        max_values = np.max(img_float, axis=(0, 1))
        corrected = img_float / max_values
        return (corrected * 255).astype(np.uint8)
    
    def von_kries_adaptation(self, image: np.ndarray, illuminant_estimate: Optional[np.ndarray] = None) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        
        if illuminant_estimate is None:
            gray_world_estimate = np.mean(img_float, axis=(0, 1))
            brightness = np.mean(img_float, axis=2)
            threshold = np.percentile(brightness, 95)
            bright_mask = brightness > threshold
            
            if np.sum(bright_mask) > 0:
                white_patch_estimate = np.mean(img_float[bright_mask], axis=0)
                illuminant_estimate = 0.7 * gray_world_estimate + 0.3 * white_patch_estimate
            else:
                illuminant_estimate = gray_world_estimate
        
        target_illuminant = np.array([0.31, 0.31, 0.31])
        illuminant_estimate = np.maximum(illuminant_estimate, 0.02)
        adaptation_coefficients = target_illuminant / illuminant_estimate
        adaptation_coefficients = np.clip(adaptation_coefficients, 0.6, 1.7)
        adaptation_coefficients = 1.0 + 0.6 * (adaptation_coefficients - 1.0)
        
        adapted = img_float * adaptation_coefficients
        return (np.clip(adapted, 0, 1) * 255).astype(np.uint8)
    
    def retinex_enhancement(self, image: np.ndarray, sigma: float = 15) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        enhanced = np.zeros_like(img_float)
        
        for channel in range(3):
            channel_data = img_float[:, :, channel]
            smoothed = ndimage.gaussian_filter(channel_data, sigma=0.5)
            log_img = np.log(smoothed + 0.04)
            surround = ndimage.gaussian_filter(log_img, sigma=sigma)
            enhanced[:, :, channel] = log_img - 0.3 * surround
        
        enhanced_min = np.percentile(enhanced, 5)
        enhanced_max = np.percentile(enhanced, 95)
        
        if enhanced_max > enhanced_min:
            enhanced = (enhanced - enhanced_min) / (enhanced_max - enhanced_min)
        
        enhanced = np.power(np.clip(enhanced, 0, 1), 0.9)
        original_normalized = img_float
        enhanced = 0.6 * enhanced + 0.4 * original_normalized
        
        return (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    
    def spatial_color_correction(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) / 255.0
        height, width = img_float.shape[:2]
        window_size = min(height, width) // 8
        sigma = window_size / 3.0
        adapted = img_float.copy()
        global_mean = np.mean(img_float, axis=(0, 1))
        
        step = window_size // 2
        for y in range(0, height - window_size, step):
            for x in range(0, width - window_size, step):
                patch = img_float[y:y+window_size, x:x+window_size]
                local_mean = np.mean(patch, axis=(0, 1))
                correction = global_mean / (local_mean + 1e-6)
                correction = np.clip(correction, 0.8, 1.2)
                
                center_y, center_x = window_size // 2, window_size // 2
                yy, xx = np.meshgrid(np.arange(window_size), np.arange(window_size), indexing='ij')
                weight = np.exp(-((yy - center_y)**2 + (xx - center_x)**2) / (2 * sigma**2))
                
                for c in range(3):
                    corrected_patch = patch[:, :, c] * correction[c]
                    weighted_correction = patch[:, :, c] + weight * (corrected_patch - patch[:, :, c])
                    y_end = min(y + window_size, height)
                    x_end = min(x + window_size, width)
                    adapted[y:y_end, x:x_end, c] = np.maximum(
                        adapted[y:y_end, x:x_end, c],
                        weighted_correction[:y_end-y, :x_end-x]
                    )
        
        return (np.clip(adapted, 0, 1) * 255).astype(np.uint8)
    
    def estimate_illuminant_from_specular_highlights(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        value_threshold = np.percentile(hsv[:, :, 2], 95)
        saturation_threshold = np.percentile(hsv[:, :, 1], 20)
        mask = (hsv[:, :, 2] > value_threshold) & (hsv[:, :, 1] < saturation_threshold)
        
        if np.sum(mask) > 0:
            highlights = image[mask]
            return np.mean(highlights, axis=0) / 255.0
        else:
            return np.mean(image, axis=(0, 1)) / 255.0
    
    def gentle_combined_enhancement(self, image: np.ndarray) -> np.ndarray:
        img = self.gray_world_correction(image)
        img_float = img.astype(np.float32) / 255.0
        illuminant_estimate = np.mean(img_float, axis=(0, 1))
        target_illuminant = np.array([0.33, 0.33, 0.33])
        adaptation_coefficients = target_illuminant / (illuminant_estimate + 1e-6)
        adaptation_coefficients = np.clip(adaptation_coefficients, 0.7, 1.4)
        adaptation_coefficients = 1.0 + 0.5 * (adaptation_coefficients - 1.0)
        img_float *= adaptation_coefficients
        img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        return img
    
    def comprehensive_enhancement(self, image: np.ndarray, method: str = 'combined') -> np.ndarray:
        enhancement_methods = {
            'gray_world': self.gray_world_correction,
            'white_patch': self.white_patch_correction,
            'von_kries': self.von_kries_adaptation,
            'retinex': self.retinex_enhancement,
            'spatial': self.spatial_color_correction,
        }
        
        if method in enhancement_methods:
            return enhancement_methods[method](image)
        elif method == 'combined':
            return self.gentle_combined_enhancement(image)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def enhance_image(self, image_path: str, method: str = 'combined', output_path: Optional[str] = None) -> np.ndarray:
        original = self.load_image(image_path)
        enhanced = self.comprehensive_enhancement(original, method)
        self.enhanced_image = enhanced
        
        if output_path:
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, enhanced_bgr)
        
        return enhanced
    
    def display_results(self, save_comparison: Optional[str] = None) -> None:
        if self.original_image is None or self.enhanced_image is None:
            raise ValueError("No images to display. Run enhance_image first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(self.original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(self.enhanced_image)
        ax2.set_title('Enhanced Image (Color Constancy)')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_comparison:
            plt.savefig(save_comparison, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_color_statistics(self, image: np.ndarray) -> Dict[str, float]:
        img_float = image.astype(np.float32) / 255.0
        means = np.mean(img_float, axis=(0, 1))
        stds = np.std(img_float, axis=(0, 1))
        global_mean = np.mean(means)
        
        return {
            'mean_r': means[0],
            'mean_g': means[1],
            'mean_b': means[2],
            'std_r': stds[0],
            'std_g': stds[1],
            'std_b': stds[2],
            'red_cast': means[0] - global_mean,
            'green_cast': means[1] - global_mean,
            'blue_cast': means[2] - global_mean,
        }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Enhance photo colors using color constancy principles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input_image', help='Path to input image')
    parser.add_argument(
        '--method', 
        choices=['gray_world', 'white_patch', 'von_kries', 'retinex', 'spatial', 'combined'],
        default='combined',
        help='Enhancement method to use'
    )
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--comparison', help='Save comparison image path')
    parser.add_argument('--show', action='store_true', help='Display results')
    parser.add_argument('--stats', action='store_true', help='Show color statistics')
    
    return parser


def print_color_statistics(stats: Dict[str, float], label: str) -> None:
    print(f"\n{label}:")
    print(f"  Mean RGB: ({stats['mean_r']:.3f}, {stats['mean_g']:.3f}, {stats['mean_b']:.3f})")
    print(f"  Color cast: R={stats['red_cast']:.3f}, G={stats['green_cast']:.3f}, B={stats['blue_cast']:.3f}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    
    if not Path(args.input_image).exists():
        print(f"Error: Input image '{args.input_image}' not found")
        return
    
    enhancer = ColorConstancyEnhancer()
    
    try:
        enhanced = enhancer.enhance_image(args.input_image, args.method, args.output)
        
        if args.stats:
            original_stats = enhancer.analyze_color_statistics(enhancer.original_image)
            enhanced_stats = enhancer.analyze_color_statistics(enhanced)
            
            print_color_statistics(original_stats, "Original Image Statistics")
            print_color_statistics(enhanced_stats, "Enhanced Image Statistics")
        
        if args.show:
            enhancer.display_results(args.comparison)
        
        print(f"\nEnhancement complete using '{args.method}' method.")
        if args.output:
            print(f"Enhanced image saved to: {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
