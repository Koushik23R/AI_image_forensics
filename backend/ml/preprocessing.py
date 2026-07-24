from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def apply_fft_to_image(image_path: str, img_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Applies 2D Fast Fourier Transform (FFT) to an image.
    Returns the log magnitude spectrum as a grayscale image (normalized).
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    img_gray = cv2.resize(img_gray, img_size)
    f = np.fft.fft2(img_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        cv2.CV_8U,
    )
    return np.expand_dims(magnitude_spectrum, axis=-1)


def preprocess_image_for_spatial_branch(
    image_path: str,
    img_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Loads and preprocesses an image for the spatial CNN branch."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img
