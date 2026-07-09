"""Command-line interface for color_constancy."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import (
    MSRCR,
    AlgorithmPipeline,
    ColorConstancyAlgorithm,
    GrayWorldCorrection,
    MultiScaleRetinex,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
)
from .io import load_image, save_image
from .metrics import color_statistics
from .visualization import display_comparison, visualize_illuminant


def _build_algorithm(method: str, params: dict[str, Any]) -> ColorConstancyAlgorithm:
    """Construct the requested algorithm with the given parameters.

    ``method`` can be any of the single-algorithm names, ``'combined'``, or the
    new MSR/MSRCR variants ``'msr'`` and ``'msrcr'``.
    """
    if method == "combined":
        # If user suppressed color restoration, fall back to MSR.
        if not params.get("msrcr", True):
            return AlgorithmPipeline(
                [
                    GrayWorldCorrection(),
                    VonKriesAdaptation(adaptation_strength=0.5, clip_range=(0.7, 1.4)),
                    MultiScaleRetinex(
                        sigmas=params.get("sigmas", (15.0, 80.0, 250.0)),
                        blend_alpha=params.get("blend_alpha", 0.7),
                    ),
                ],
                _repr_name="Combined (MSR)",
            )
        return AlgorithmPipeline(
            [
                GrayWorldCorrection(),
                VonKriesAdaptation(adaptation_strength=0.5, clip_range=(0.7, 1.4)),
                MSRCR(
                    sigmas=params.get("sigmas", (15.0, 80.0, 250.0)),
                    blend_alpha=params.get("blend_alpha", 0.7),
                    cr_gain=params.get("cr_gain", 125.0),
                    cr_bias=params.get("cr_bias", -46.0),
                ),
            ],
            _repr_name="Combined (MSRCR)",
        )

    if method == "gray_world":
        return GrayWorldCorrection()

    if method == "white_patch":
        return WhitePatchCorrection()

    if method == "von_kries":
        return VonKriesAdaptation(
            adaptation_strength=params.get("adaptation_strength", 0.6),
            clip_range=params.get("clip_range", (0.6, 1.7)),
            gray_world_weight=params.get("gray_world_weight", 0.7),
        )

    if method == "retinex":
        return RetinexEnhancement(
            surround_sigma=params.get("sigma", 15.0),
            blend_alpha=params.get("blend_alpha", 0.6),
        )

    if method == "msr":
        return MultiScaleRetinex(
            sigmas=params.get("sigmas", (15.0, 80.0, 250.0)),
            blend_alpha=params.get("blend_alpha", 0.7),
        )

    if method == "msrcr":
        return MSRCR(
            sigmas=params.get("sigmas", (15.0, 80.0, 250.0)),
            blend_alpha=params.get("blend_alpha", 0.7),
            cr_gain=params.get("cr_gain", 125.0),
            cr_bias=params.get("cr_bias", -46.0),
        )

    if method == "spatial":
        return SpatialColorCorrection(
            correction_strength=params.get("correction_strength", 0.2),
        )

    raise ValueError(f"Unknown method: {method!r}")


# Preset configurations for common photographic scenarios.
_PRESETS: dict[str, dict[str, Any]] = {
    "default": {"method": "combined"},
    "night": {
        "method": "combined",
        "adaptation_strength": 0.4,
        "clip_range": (0.5, 2.0),
        "blend_alpha": 0.8,
        "cr_gain": 150.0,
        "cr_bias": -50.0,
    },
    "indoor_tungsten": {
        "method": "von_kries",
        "adaptation_strength": 0.8,
        "clip_range": (0.5, 1.8),
        "gray_world_weight": 0.5,
    },
    "sunset": {
        "method": "combined",
        "adaptation_strength": 0.3,
        "blend_alpha": 0.5,
        "cr_gain": 80.0,
        "cr_bias": -30.0,
    },
    "high_contrast": {
        "method": "msr",
        "sigmas": (10.0, 60.0, 200.0),
        "blend_alpha": 0.85,
    },
    "vivid": {
        "method": "msrcr",
        "sigmas": (15.0, 80.0, 250.0),
        "blend_alpha": 0.6,
        "cr_gain": 200.0,
        "cr_bias": -60.0,
    },
    "subtle": {
        "method": "spatial",
        "correction_strength": 0.1,
    },
}


def _parse_key_value(raw: str) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dictionary with typed values."""
    result: dict[str, Any] = {}
    for pair in raw.split(","):
        key, _, val = pair.partition("=")
        if not key or not val:
            continue
        # Try to coerce to number / boolean.
        if val.lower() == "true":
            result[key.strip()] = True
        elif val.lower() == "false":
            result[key.strip()] = False
        else:
            try:
                result[key.strip()] = int(val)
            except ValueError:
                try:
                    result[key.strip()] = float(val)
                except ValueError:
                    result[key.strip()] = val
    return result


def _load_preset(name_or_path: str) -> dict[str, Any]:
    """Load preset parameters by name or from a JSON/YAML-adjacent file."""
    if name_or_path in _PRESETS:
        return dict(_PRESETS[name_or_path])
    path = Path(name_or_path)
    if path.suffix in (".json",):
        with open(path) as f:
            return json.load(f)
    raise ValueError(
        f"Unknown preset: {name_or_path!r}. "
        f"Available presets: {', '.join(sorted(_PRESETS))}"
    )


def create_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhance photo colors using color constancy principles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument(
        "--method",
        choices=["gray_world", "white_patch", "von_kries", "retinex",
                 "msr", "msrcr", "spatial", "combined"],
        default="combined",
        help="Color constancy algorithm to apply.",
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
        help="Print per-channel color statistics to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show an illuminant-diagnostic histogram chart. "
            "Only applies to algorithms that expose estimate_illuminant()."
        ),
    )

    # --- Algorithm parameters ---
    param_group = parser.add_argument_group("Algorithm parameters")
    param_group.add_argument(
        "--sigma", type=float, metavar="FLOAT",
        help="Surround sigma for SSR Retinex (default: 15.0).",
    )
    param_group.add_argument(
        "--sigmas", type=str, metavar="A,B,C",
        help="Comma-separated sigma triplet for MSR/MSRCR (default: 15,80,250).",
    )
    param_group.add_argument(
        "--blend-alpha", type=float, metavar="FLOAT",
        help="Blend weight for Retinex/MSR/MSRCR output vs original.",
    )
    param_group.add_argument(
        "--adaptation-strength", type=float, metavar="FLOAT",
        help="Von Kries adaptation strength in [0, 1] (default: 0.6).",
    )
    param_group.add_argument(
        "--correction-strength", type=float, metavar="FLOAT",
        help="Spatial correction clipping half-width (default: 0.2).",
    )
    param_group.add_argument(
        "--cr-gain", type=float, metavar="FLOAT",
        help="MSRCR color restoration gain (default: 125.0).",
    )
    param_group.add_argument(
        "--cr-bias", type=float, metavar="FLOAT",
        help="MSRCR color restoration bias (default: -46.0).",
    )
    param_group.add_argument(
        "--msrcr", type=bool, default=True, metavar="BOOL",
        help="Enable MSRCR color restoration in combined pipeline (default: true).",
    )

    # --- Bridging old-style param key=value ---
    param_group.add_argument(
        "--param", type=str, metavar="k=v,...",
        help="Additional algorithm parameters as comma-separated key=value pairs.",
    )

    # --- Presets ---
    preset_group = parser.add_argument_group("Presets")
    preset_group.add_argument(
        "--preset",
        choices=list(_PRESETS),
        default="default",
        help="Load a named preset for quick configuration.",
    )
    preset_group.add_argument(
        "--preset-file", type=str, metavar="PATH",
        help="Load parameters from a JSON preset file.",
    )
    return parser


def _collect_params(args: argparse.Namespace) -> dict[str, Any]:
    """Merge argparse flags into a params dictionary with correct typing."""
    params: dict[str, Any] = {}

    if args.sigma is not None:
        params["sigma"] = args.sigma
    if args.sigmas is not None:
        params["sigmas"] = tuple(float(x.strip()) for x in args.sigmas.split(","))
    if args.blend_alpha is not None:
        params["blend_alpha"] = args.blend_alpha
    if args.adaptation_strength is not None:
        params["adaptation_strength"] = args.adaptation_strength
    if args.correction_strength is not None:
        params["correction_strength"] = args.correction_strength
    if args.cr_gain is not None:
        params["cr_gain"] = args.cr_gain
    if args.cr_bias is not None:
        params["cr_bias"] = args.cr_bias
    if args.param is not None:
        params.update(_parse_key_value(args.param))

    return params


def _print_stats(stats: dict[str, float], label: str) -> None:
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

    # Load preset, then override with CLI params.
    preset_params: dict[str, Any] = {}
    if args.preset != "default":
        preset_params = _load_preset(args.preset)
    if args.preset_file:
        preset_params.update(_load_preset(args.preset_file))

    cli_params = _collect_params(args)
    merged = {**preset_params, **cli_params}

    # Determine method: presets may override.
    method = cli_params.pop("method", None) or preset_params.pop("method", None) or args.method

    try:
        original_uint8 = load_image(str(input_path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        original = original_uint8.astype(np.float32) / 255.0
        algorithm = _build_algorithm(method, merged)
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

        print(f"\nEnhancement complete ({method}).")
        if args.output:
            print(f"Saved: {args.output}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error during processing: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
