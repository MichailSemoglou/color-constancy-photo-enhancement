"""Visualisation helpers for results and diagnostic inspection."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def display_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Display a side-by-side comparison of the original and enhanced images.

    Parameters
    ----------
    original:
        Original RGB image, dtype ``uint8``.
    enhanced:
        Enhanced RGB image, dtype ``uint8``, same spatial dimensions.
    save_path:
        If provided, save the figure to this path at 300 DPI before
        displaying.
    show:
        If ``True``, call ``plt.show()`` to open an interactive window.
        Set to ``False`` for save-only or headless runs; the figure is
        closed automatically so no resources are leaked.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.imshow(original)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(enhanced)
    ax2.set_title("Enhanced Image (Colour Constancy)")
    ax2.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def visualize_illuminant(
    image: np.ndarray,
    illuminant: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Show per-channel histograms with the estimated illuminant marked.

    Displays the pixel-value histogram for each of the three RGB channels
    and draws a vertical dashed line at the estimated illuminant value.
    Useful for diagnosing illuminant estimation quality without access to
    ground-truth data.

    Parameters
    ----------
    image:
        Float32 RGB image, values in ``[0, 1]``, shape ``(H, W, 3)``.
    illuminant:
        Estimated illuminant, shape ``(3,)``.
    save_path:
        If provided, save the figure before displaying.
    show:
        If ``True``, call ``plt.show()`` to open an interactive window.
        Set to ``False`` for save-only or headless runs; the figure is
        closed automatically so no resources are leaked.
    """
    channel_names = ["Red", "Green", "Blue"]
    colors = ["red", "green", "blue"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Channel Histograms with Estimated Illuminant", fontsize=13)

    for c, (ax, name, color) in enumerate(zip(axes, channel_names, colors)):
        ax.hist(
            image[:, :, c].ravel(),
            bins=256,
            range=(0.0, 1.0),
            color=color,
            alpha=0.7,
        )
        ax.axvline(
            float(illuminant[c]),
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Illuminant ({illuminant[c]:.3f})",
        )
        ax.set_title(f"{name} Channel")
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
