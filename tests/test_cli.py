"""Tests for the color_constancy CLI (create_parser, main)."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from color_constancy.cli import create_parser, main


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
    ["gray_world", "white_patch", "von_kries", "retinex", "spatial", "combined"],
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
    ["gray_world", "white_patch", "von_kries", "retinex", "spatial", "combined"],
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
