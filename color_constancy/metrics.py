"""Quantitative evaluation metrics for color constancy research.

All functions operate on **float32** arrays with pixel values in ``[0, 1]``
unless otherwise stated.
"""


import numpy as np
from scipy.ndimage import uniform_filter


def angular_error(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute the angular error between two illuminant estimates.

    The angular error is the angle (in degrees) between two chromaticity
    vectors in RGB space.  It is the standard evaluation metric for illuminant
    estimation algorithms (Hordley & Finlayson, 2006).  A value of ``0°``
    is a perfect estimate; larger values indicate greater divergence.

    Parameters
    ----------
    estimated:
        Estimated illuminant, shape ``(3,)``.
    ground_truth:
        Ground-truth illuminant, shape ``(3,)``.

    Returns
    -------
    float
        Angular error in degrees, in ``[0°, 180°]``.

    References
    ----------
    Hordley, S. D., & Finlayson, G. D. (2006). Reevaluation of color
    constancy algorithm performance. *Journal of the Optical Society of
    America A*, 23(5), 1008–1020.
    """

    def _normalise(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v

    cos_theta = np.clip(
        np.dot(_normalise(estimated.astype(np.float64)),
               _normalise(ground_truth.astype(np.float64))),
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cos_theta)))


def psnr(reference: np.ndarray, distorted: np.ndarray, max_value: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio between two float32 images.

    Parameters
    ----------
    reference:
        Reference (ground-truth) image, dtype float32, values in ``[0, 1]``.
    distorted:
        Distorted or reconstructed image, same shape and range.
    max_value:
        Maximum possible pixel value.  Default ``1.0`` for normalised images.

    Returns
    -------
    float
        PSNR in dB.  Returns ``inf`` when the images are identical.

    References
    ----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    *IEEE Transactions on Image Processing*, 13(4), 600–612.
    """
    mse = float(
        np.mean(
            (reference.astype(np.float64) - distorted.astype(np.float64)) ** 2
        )
    )
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(max_value**2 / mse))


def ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Structural Similarity Index Measure (SSIM).

    Implements the luminance + contrast + structure formulation from Wang et
    al. (2004).  Computed per channel with an 11-pixel uniform filter and
    averaged across the three channels.

    Parameters
    ----------
    reference:
        Reference image, dtype float32, values in ``[0, 1]``, shape
        ``(H, W, 3)``.
    distorted:
        Distorted image, same shape and range.

    Returns
    -------
    float
        Mean SSIM in ``[-1, 1]``.  ``1.0`` indicates identical images.

    References
    ----------
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    *IEEE Transactions on Image Processing*, 13(4), 600–612.
    """
    C1 = 0.01**2
    C2 = 0.03**2

    channel_ssims = []
    for c in range(reference.shape[2]):
        r = reference[:, :, c].astype(np.float64)
        d = distorted[:, :, c].astype(np.float64)

        mu_r = uniform_filter(r, size=11)
        mu_d = uniform_filter(d, size=11)
        mu_r2 = mu_r**2
        mu_d2 = mu_d**2
        mu_rd = mu_r * mu_d

        sigma_r2 = uniform_filter(r * r, size=11) - mu_r2
        sigma_d2 = uniform_filter(d * d, size=11) - mu_d2
        sigma_rd = uniform_filter(r * d, size=11) - mu_rd

        numerator = (2.0 * mu_rd + C1) * (2.0 * sigma_rd + C2)
        denominator = (mu_r2 + mu_d2 + C1) * (sigma_r2 + sigma_d2 + C2)
        channel_ssims.append(float(np.mean(numerator / denominator)))

    return float(np.mean(channel_ssims))


def color_statistics(image: np.ndarray) -> dict[str, float]:
    """Compute per-channel summary statistics for a float32 image.

    Parameters
    ----------
    image:
        RGB image, dtype float32, values in ``[0, 1]``, shape ``(H, W, 3)``.

    Returns
    -------
    dict
        Keys: ``mean_r``, ``mean_g``, ``mean_b``, ``std_r``, ``std_g``,
        ``std_b``, ``red_cast``, ``green_cast``, ``blue_cast``.
        The ``*_cast`` values are the per-channel deviation from the global
        mean brightness; a value near zero indicates a neutral channel.
    """
    img = image.astype(np.float64)
    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    global_mean = means.mean()

    return {
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "std_r": float(stds[0]),
        "std_g": float(stds[1]),
        "std_b": float(stds[2]),
        "red_cast": float(means[0] - global_mean),
        "green_cast": float(means[1] - global_mean),
        "blue_cast": float(means[2] - global_mean),
    }
