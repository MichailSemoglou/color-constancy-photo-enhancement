"""Shared constants for color constancy algorithms."""

import numpy as np

# Neutral (equal-energy) illuminant — used as the adaptation target in Von Kries.
# Expressed as equal R, G, B fractions summing to 1.
NEUTRAL_ILLUMINANT: np.ndarray = np.array(
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32
)
