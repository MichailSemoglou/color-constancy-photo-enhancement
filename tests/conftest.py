"""Shared pytest fixtures for the color_constancy test suite."""

import numpy as np
import pytest


@pytest.fixture()
def neutral_image() -> np.ndarray:
    """Perfectly neutral mid-grey image — colour casts should be preserved at zero."""
    return np.full((64, 64, 3), 0.5, dtype=np.float32)


@pytest.fixture()
def red_cast_image() -> np.ndarray:
    """Image with a noticeable red cast."""
    img = np.full((64, 64, 3), 0.4, dtype=np.float32)
    img[:, :, 0] = 0.75  # boosted red channel
    return img


@pytest.fixture()
def random_image() -> np.ndarray:
    """Reproducible random image with seed 42."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 3)).astype(np.float32)


@pytest.fixture()
def near_black_image() -> np.ndarray:
    """Near-black image for testing degenerate-input robustness."""
    return np.full((64, 64, 3), 0.001, dtype=np.float32)
