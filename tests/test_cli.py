"""Tests for the color_constancy CLI (create_parser, main)."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from color_constancy.cli import (
    _build_algorithm,
    _collect_params,
    _parse_key_value,
    create_parser,
    main,
)


def _make_png(path: Path, value: int = 128, size: int = 64) -> None:
    img = np.full((size, size, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_default_method():
    args = create_parser().parse_args(["img.jpg"])
    assert args.method == "combined"


def test_parser_default_flags_are_false():
    args = create_parser().parse_args(["img.jpg"])
    assert args.output is None
    assert args.comparison is None
    assert not args.show
    assert not args.stats
    assert not args.debug


@pytest.mark.parametrize(
    "method",
    ["gray_world", "white_patch", "von_kries", "retinex", "msr", "msrcr", "spatial", "combined"],
)
def test_parser_accepts_all_methods(method):
    args = create_parser().parse_args(["img.jpg", "--method", method])
    assert args.method == method


def test_parser_rejects_invalid_method():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["img.jpg", "--method", "unknown"])


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_exits_1_for_missing_input(tmp_path):
    with patch("sys.argv", ["prog", str(tmp_path / "no_such.png")]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_main_saves_output(tmp_path):
    src = tmp_path / "src.png"
    out = tmp_path / "out.png"
    _make_png(src)

    with patch("sys.argv", ["prog", str(src), "--output", str(out)]):
        main()

    assert out.exists()


@pytest.mark.parametrize(
    "method",
    ["gray_world", "white_patch", "von_kries", "retinex", "msr", "msrcr", "spatial", "combined"],
)
def test_main_all_methods_produce_output(tmp_path, method):
    src = tmp_path / "src.png"
    out = tmp_path / f"{method}.png"
    _make_png(src)

    with patch("sys.argv", ["prog", str(src), "--method", method, "--output", str(out)]):
        main()

    assert out.exists()


def test_main_stats_flag_prints_output(tmp_path, capsys):
    src = tmp_path / "src.png"
    _make_png(src)

    with patch("sys.argv", ["prog", str(src), "--stats"]):
        main()

    captured = capsys.readouterr()
    assert "Mean RGB" in captured.out
    assert "Cast" in captured.out


def test_main_no_show_flag_skips_display(tmp_path):
    """Ensure main() does not call plt.show() when --show is absent."""
    src = tmp_path / "src.png"
    _make_png(src)

    with patch("sys.argv", ["prog", str(src)]):
        with patch("color_constancy.visualization.plt.show") as mock_show:
            main()
    mock_show.assert_not_called()


# ---------------------------------------------------------------------------
# SME parameter forwarding (regression tests)
# ---------------------------------------------------------------------------


def test_build_algorithm_sme_forwards_all_params():
    """Every SME constructor parameter must reach the instance."""
    algo = _build_algorithm(
        "sme",
        {
            "auto": False,
            "contrast_strength": 1.2,
            "saturation_gain": 1.4,
            "shadow_protection": 0.05,
            "highlight_protection": 0.2,
            "chroma_threshold": 10.0,
            "cdc_threshold": 0.6,
        },
    )
    assert algo.auto is False
    assert algo.contrast_strength == 1.2
    assert algo.saturation_gain == 1.4
    assert algo.shadow_protection == 0.05
    assert algo.highlight_protection == 0.2
    assert algo.chroma_threshold == 10.0
    assert algo.cdc_threshold == 0.6


def test_main_sme_manual_params_not_noop(tmp_path):
    """Manual SME parameters via --param must change the output."""
    src = tmp_path / "src.png"
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    img = np.tile(gradient, (64, 1))[:, :, None].repeat(3, axis=2)
    cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out_weak = tmp_path / "weak.png"
    out_strong = tmp_path / "strong.png"

    with patch("sys.argv", ["prog", str(src), "--method", "sme",
                            "--param", "auto=false,contrast_strength=0.0",
                            "--output", str(out_weak)]):
        main()
    with patch("sys.argv", ["prog", str(src), "--method", "sme",
                            "--param", "auto=false,contrast_strength=2.0",
                            "--output", str(out_strong)]):
        main()

    weak = cv2.imread(str(out_weak))
    strong = cv2.imread(str(out_strong))
    assert not np.array_equal(weak, strong)


def test_repeated_param_flags_merge():
    """Repeated --param flags must accumulate, not overwrite."""
    args = create_parser().parse_args(
        ["img.jpg", "--param", "contrast_strength=1.2", "--param", "saturation_gain=1.4"]
    )
    params = _collect_params(args)
    assert params == {"contrast_strength": 1.2, "saturation_gain": 1.4, "msrcr": True}


# ---------------------------------------------------------------------------
# CLI trust repairs (regression tests)
# ---------------------------------------------------------------------------


def test_no_msrcr_flag_disables_color_restoration():
    """--no-msrcr must select the MSR fallback pipeline."""
    args = create_parser().parse_args(["img.jpg", "--no-msrcr"])
    params = _collect_params(args)
    assert params["msrcr"] is False
    algo = _build_algorithm("combined", params)
    step_names = [type(s).__name__ for s in algo.steps]
    assert "MultiScaleRetinex" in step_names
    assert "MSRCR" not in step_names


def test_msrcr_default_stays_enabled():
    args = create_parser().parse_args(["img.jpg"])
    algo = _build_algorithm("combined", _collect_params(args))
    assert any(type(s).__name__ == "MSRCR" for s in algo.steps)


def test_scalar_sigmas_param_raises_named_error():
    """A scalar sigmas (truncated by --param splitting) must fail helpfully."""
    with pytest.raises(ValueError, match="--sigmas"):
        _build_algorithm("msr", {"sigmas": 15})


def test_bracketed_sequence_param_parses():
    params = _parse_key_value("sigmas=[15,80,250]")
    assert params == {"sigmas": (15.0, 80.0, 250.0)}
    algo = _build_algorithm("msr", params)
    assert algo.sigmas == (15.0, 80.0, 250.0)


def test_main_scalar_sigmas_exits_with_message(tmp_path, capsys):
    """The truncated --param sigmas case must exit 1 with guidance, not a traceback."""
    src = tmp_path / "src.png"
    _make_png(src)
    with patch("sys.argv", ["prog", str(src), "--method", "msr",
                            "--param", "sigmas=15,80,250"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "--sigmas" in capsys.readouterr().err


def test_main_debug_warns_for_non_estimator(tmp_path, capsys):
    """--debug on a method without an illuminant estimate must say so."""
    src = tmp_path / "src.png"
    _make_png(src)
    with patch("sys.argv", ["prog", str(src), "--method", "msrcr", "--debug"]):
        main()
    assert "not available" in capsys.readouterr().err


def test_main_missing_preset_file_exits_1(tmp_path, capsys):
    """A nonexistent --preset-file must exit 1 with a message, not a traceback."""
    src = tmp_path / "src.png"
    _make_png(src)
    with patch("sys.argv", ["prog", str(src),
                            "--preset-file", str(tmp_path / "missing.json")]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "Error loading preset" in capsys.readouterr().err


def test_main_malformed_preset_file_exits_1(tmp_path, capsys):
    """Malformed JSON in --preset-file must exit 1 with a message."""
    src = tmp_path / "src.png"
    _make_png(src)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with patch("sys.argv", ["prog", str(src), "--preset-file", str(bad)]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "Error loading preset" in capsys.readouterr().err
