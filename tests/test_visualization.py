"""Tests for the visualization helpers, run headless."""

import matplotlib

matplotlib.use("Agg")  # no display server in CI

import matplotlib.pyplot as plt
import numpy as np

from color_constancy.visualization import display_comparison, visualize_illuminant


def test_display_comparison_saves_and_closes_figure(tmp_path):
    original = np.full((16, 16, 3), 128, dtype=np.uint8)
    enhanced = np.full((16, 16, 3), 160, dtype=np.uint8)
    out = tmp_path / "comparison.png"

    display_comparison(original, enhanced, save_path=str(out), show=False)

    assert out.exists()
    assert plt.get_fignums() == []  # figure closed, no resource leak


def test_display_comparison_show_true_invokes_show():
    calls = []
    original = plt.show
    plt.show = lambda: calls.append(1)  # type: ignore[assignment]
    try:
        display_comparison(
            np.zeros((8, 8, 3), np.uint8),
            np.zeros((8, 8, 3), np.uint8),
            show=True,
        )
    finally:
        plt.show = original  # type: ignore[assignment]
    assert len(calls) == 1
    assert plt.get_fignums() == []


def test_visualize_illuminant_saves_and_closes_figure(tmp_path):
    image = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    illuminant = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    out = tmp_path / "illuminant.png"

    visualize_illuminant(image, illuminant, save_path=str(out), show=False)

    assert out.exists()
    assert plt.get_fignums() == []


def test_visualize_illuminant_show_false_does_not_call_show():
    calls = []
    original = plt.show
    plt.show = lambda: calls.append(1)  # type: ignore[assignment]
    try:
        image = np.full((8, 8, 3), 0.5, dtype=np.float32)
        visualize_illuminant(image, np.full(3, 0.5, dtype=np.float32), show=False)
    finally:
        plt.show = original  # type: ignore[assignment]
    assert calls == []
    assert plt.get_fignums() == []
