"""Image loading and saving utilities."""

from pathlib import Path

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Raise ``ValueError`` for images that are too small or wrong shape.

    Parameters
    ----------
    image:
        Candidate RGB array to validate.

    Raises
    ------
    ValueError
        If the image does not have shape ``(H, W, 3)`` or if either spatial
        dimension is smaller than 8 pixels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB image with shape (H, W, 3), got shape {image.shape}."
        )
    if min(image.shape[:2]) < 8:
        raise ValueError(
            f"Image dimensions {image.shape[:2]} are too small to process; "
            "the minimum supported spatial dimension is 8 pixels."
        )


def load_image(image_path: str) -> np.ndarray:
    """Load an image file and return it as an RGB ``uint8`` array.

    Grayscale images are promoted to three-channel RGB automatically.

    Parameters
    ----------
    image_path:
        Path to the source image.  Supports all formats recognised by OpenCV
        (JPEG, PNG, TIFF, BMP, …).

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``, channel order RGB.

    Raises
    ------
    FileNotFoundError
        If *image_path* does not exist on disk.
    ValueError
        If OpenCV cannot decode the file, or if the image is too small.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(
            f"Could not decode image: {path}. "
            "Ensure the file is a valid, non-corrupted image."
        )

    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    _validate_image(rgb)
    return rgb


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save an RGB ``uint8`` image to disk.

    Parent directories are created automatically if they do not already exist.

    Parameters
    ----------
    image:
        RGB image, dtype ``uint8``, shape ``(H, W, 3)``.
    output_path:
        Destination file path.  The format is inferred from the file extension.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"cv2.imwrite failed to write image to '{path}'. "
                      "Check that the file extension is supported and the path is writable.")
