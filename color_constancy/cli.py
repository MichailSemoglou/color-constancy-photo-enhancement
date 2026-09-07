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
    SelectiveMidtoneEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
)
from .io import load_image, save_image
from .metrics import color_statistics
from .visualization import display_comparison, visualize_illuminant


def _tuple_param(
    params: dict[str, Any],
    name: str,
    default: tuple[float, ...],
    flag: str | None = None,
) -> tuple[float, ...]:
    """Return a tuple-typed parameter, rejecting scalars with a useful error.

    A scalar almost always means a comma-separated value was truncated when
    ``--param`` split on commas (``sigmas=15,80,250`` becomes ``15``).
    """
    value = params.get(name, default)
    if isinstance(value, (int, float)):
        hint = f"use --{flag}" if flag else f'use --param "{name}=[A,B,C]" (brackets required)'
        raise ValueError(
            f"Parameter {name!r} expects a sequence of numbers, got the scalar "
            f"{value!r}. To pass several values, {hint}."
        )
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Parameter {name!r} expects a sequence of numbers, got {value!r}."
        ) from None
    if not result:
        raise ValueError(
            f"Parameter {name!r} expects a non-empty sequence of numbers, got []."
        )
    return result


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
                    MultiScaleRetinex(
                        sigmas=_tuple_param(params, "sigmas", (15.0, 80.0, 250.0), flag="sigmas A,B,C"),
                        blend_alpha=params.get("blend_alpha", 0.7),
                    ),
                ],
                _repr_name="Combined (MSR)",
            )
        return AlgorithmPipeline(
            [
                GrayWorldCorrection(),
                MSRCR(
                    sigmas=_tuple_param(params, "sigmas", (15.0, 80.0, 250.0), flag="sigmas A,B,C"),
                    blend_alpha=params.get("blend_alpha", 0.7),
                    cr_alpha=params.get("cr_alpha", 125.0),
                    cr_beta=params.get("cr_beta", 46.0),
                    cr_gain=params.get("cr_gain", 192.0),
                    cr_bias=params.get("cr_bias", -30.0),
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
            clip_range=_tuple_param(params, "clip_range", (0.6, 1.7)),
            gray_world_weight=params.get("gray_world_weight", 0.7),
        )

    if method == "retinex":
        return RetinexEnhancement(
            surround_sigma=params.get("sigma", 15.0),
            blend_alpha=params.get("blend_alpha", 0.6),
        )

    if method == "msr":
        return MultiScaleRetinex(
            sigmas=_tuple_param(params, "sigmas", (15.0, 80.0, 250.0), flag="sigmas A,B,C"),
            blend_alpha=params.get("blend_alpha", 0.7),
        )

    if method == "msrcr":
        return MSRCR(
            sigmas=_tuple_param(params, "sigmas", (15.0, 80.0, 250.0), flag="sigmas A,B,C"),
            blend_alpha=params.get("blend_alpha", 0.7),
            cr_alpha=params.get("cr_alpha", 125.0),
            cr_beta=params.get("cr_beta", 46.0),
            cr_gain=params.get("cr_gain", 192.0),
            cr_bias=params.get("cr_bias", -30.0),
        )

    if method == "spatial":
        return SpatialColorCorrection(
            correction_strength=params.get("correction_strength", 0.2),
        )

    if method == "sme":
        return SelectiveMidtoneEnhancement(
            auto=params.get("auto", True),
            contrast_strength=params.get("contrast_strength", 1.0),
            saturation_gain=params.get("saturation_gain", 1.25),
            shadow_protection=params.get("shadow_protection", 0.10),
            highlight_protection=params.get("highlight_protection", 0.10),
            chroma_threshold=params.get("chroma_threshold", 12.0),
            cdc_threshold=params.get("cdc_threshold", 0.5),
        )

    raise ValueError(f"Unknown method: {method!r}")


# Preset configurations for common photographic scenarios.
_PRESETS: dict[str, dict[str, Any]] = {
    "default": {"method": "combined"},
    "night": {
        "method": "combined",
        "blend_alpha": 0.8,
        "cr_alpha": 150.0,
    },
    "indoor_tungsten": {
        "method": "von_kries",
        "adaptation_strength": 0.8,
        "clip_range": (0.5, 1.8),
        "gray_world_weight": 0.5,
    },
    "sunset": {
        "method": "combined",
        "blend_alpha": 0.5,
        "cr_alpha": 80.0,
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
        "cr_alpha": 200.0,
    },
    "subtle": {
        "method": "spatial",
        "correction_strength": 0.1,
    },
}


def _split_pairs(raw: str) -> list[str]:
    """Split on commas, ignoring commas inside square brackets."""
    pairs: list[str] = []
    depth = 0
    current = ""
    for ch in raw:
        if ch == "[":
            depth += 1
        elif ch == "]" and depth:
            depth -= 1
        if ch == "," and depth == 0:
            pairs.append(current)
            current = ""
        else:
            current += ch
    if current:
        pairs.append(current)
    return pairs


def _parse_key_value(raw: str) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dictionary with typed values.

    Sequence values need square brackets so their commas survive pair
    splitting: ``--param "sigmas=[15,80,250]"``.
    """
    result: dict[str, Any] = {}
    for pair in _split_pairs(raw):
        key, _, val = pair.partition("=")
        if not key or not val:
            continue
        key = key.strip()
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                result[key] = tuple(float(x) for x in val[1:-1].split(",") if x.strip())
                continue
            except ValueError:
                pass  # not a numeric sequence; fall through to scalar parsing
        # Try to coerce to number / boolean.
        if val.lower() == "true":
            result[key] = True
        elif val.lower() == "false":
            result[key] = False
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def _load_preset(name_or_path: str) -> dict[str, Any]:
    """Load preset parameters by name or from a JSON/YAML-adjacent file."""
    if name_or_path in _PRESETS:
        return dict(_PRESETS[name_or_path])
    path = Path(name_or_path)
    if path.suffix in (".json",):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Preset file {path} must contain a JSON object, "
                f"got {type(data).__name__}."
            )
        return data
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
                 "msr", "msrcr", "spatial", "sme", "combined"],
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
        help="MSRCR display gain after color restoration (default: 192.0).",
    )
    param_group.add_argument(
        "--cr-bias", type=float, metavar="FLOAT",
        help="MSRCR display offset after color restoration (default: -30.0).",
    )
    param_group.add_argument(
        "--cr-alpha", type=float, metavar="FLOAT",
        help="MSRCR color restoration inner gain alpha; shapes the restored "
             "colors (default: 125.0).",
    )
    param_group.add_argument(
        "--cr-beta", type=float, metavar="FLOAT",
        help="MSRCR color restoration outer gain beta (default: 46.0); "
             "absorbed by output normalization.",
    )
    param_group.add_argument(
        "--msrcr", action=argparse.BooleanOptionalAction, default=None,
        help="Enable MSRCR color restoration in combined pipeline. "
             "Use --no-msrcr to disable. Enabled unless a preset or this flag says otherwise.",
    )

    # --- Bridging old-style param key=value ---
    param_group.add_argument(
        "--param", type=str, action="append", metavar="k=v,...",
        help="Additional algorithm parameters as comma-separated key=value "
             "pairs. May be repeated. Use brackets for sequence values, "
             "e.g. \"sigmas=[15,80,250]\".",
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
    if args.cr_alpha is not None:
        params["cr_alpha"] = args.cr_alpha
    if args.cr_beta is not None:
        params["cr_beta"] = args.cr_beta
    if args.msrcr is not None:
        params["msrcr"] = args.msrcr
    if args.param:
        for raw in args.param:
            params.update(_parse_key_value(raw))

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
    try:
        preset_params: dict[str, Any] = {}
        if args.preset != "default":
            preset_params = _load_preset(args.preset)
        if args.preset_file:
            preset_params.update(_load_preset(args.preset_file))
    except (OSError, ValueError) as exc:
        # JSONDecodeError subclasses ValueError, so malformed JSON lands here.
        print(f"Error loading preset: {exc}", file=sys.stderr)
        sys.exit(1)

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
        elif args.debug:
            print(
                f"Note: --debug is not available for '{method}' "
                "(no illuminant estimate to display).",
                file=sys.stderr,
            )

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
