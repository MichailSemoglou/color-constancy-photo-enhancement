"""Command-line interface for color_constancy."""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .algorithms import (
    AlgorithmPipeline,
    ColorConstancyAlgorithm,
    GrayWorldCorrection,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
    build_combined_pipeline,
)
from .io import load_image, save_image
from .metrics import color_statistics
from .visualization import display_comparison, visualize_illuminant

# Registry of available single-algorithm methods.
# The "combined" entry is built via build_combined_pipeline() to keep it
# separate from the individually-parameterised algorithms.
_METHODS: Dict[str, ColorConstancyAlgorithm] = {
    "gray_world": GrayWorldCorrection(),
    "white_patch": WhitePatchCorrection(),
    "von_kries": VonKriesAdaptation(),
    "retinex": RetinexEnhancement(),
    "spatial": SpatialColorCorrection(),
    "combined": build_combined_pipeline(),
}


def create_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhance photo colours using colour constancy principles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument(
        "--method",
        choices=list(_METHODS),
        default="combined",
        help="Colour constancy algorithm to apply.",
    )
    parser.add_argument("--output", help="Save the enhanced image to this path.")
    parser.add_argument(
        "--comparison",
        help="Save a side-by-side before/after comparison to this path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the before/after comparison in a window.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-channel colour statistics to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show an illuminant-diagnostic histogram chart. "
            "Only applies to algorithms that expose estimate_illuminant()."
        ),
    )
    return parser


def _print_stats(stats: Dict[str, float], label: str) -> None:
    print(f"\n{label}:")
    print(
        f"  Mean RGB : ({stats['mean_r']:.3f}, {stats['mean_g']:.3f},"
        f" {stats['mean_b']:.3f})"
    )
    print(
        f"  Cast     : R={stats['red_cast']:+.3f}  G={stats['green_cast']:+.3f}"
        f"  B={stats['blue_cast']:+.3f}"
    )


def main() -> None:
    """Entry point for the ``color-constancy-enhance`` console script."""
    parser = create_parser()
    args = parser.parse_args()

    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: input image '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        original_uint8 = load_image(str(input_path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        original = original_uint8.astype(np.float32) / 255.0
        algorithm = _METHODS[args.method]
        enhanced = algorithm.process(original)
        enhanced_uint8 = (enhanced * 255.0).astype(np.uint8)

        if args.output:
            save_image(enhanced_uint8, args.output)

        if args.stats:
            _print_stats(color_statistics(original), "Original")
            _print_stats(color_statistics(enhanced), "Enhanced")

        if args.debug and hasattr(algorithm, "estimate_illuminant"):
            illuminant = algorithm.estimate_illuminant(original)  # type: ignore[attr-defined]
            print(
                f"\nEstimated illuminant: "
                f"R={illuminant[0]:.4f}  G={illuminant[1]:.4f}  B={illuminant[2]:.4f}"
            )
            visualize_illuminant(original, illuminant)

        if args.show or args.comparison:
            display_comparison(
                original_uint8,
                enhanced_uint8,
                save_path=args.comparison,
                show=args.show,
            )

        print(f"\nEnhancement complete ({args.method}).")
        if args.output:
            print(f"Saved: {args.output}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error during processing: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
