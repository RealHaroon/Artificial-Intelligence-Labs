"""Document cleanup filters for straightened page images."""

from __future__ import annotations

from typing import Literal, get_args

import cv2
import numpy as np
from numpy.typing import NDArray


EnhanceMode = Literal["original", "grayscale", "bw", "magic"]
ENHANCE_MODES: tuple[EnhanceMode, ...] = get_args(EnhanceMode)
ImageArray = NDArray[np.uint8]


def _validate_image(image: ImageArray) -> None:
    """Validate that an image can be processed by the enhancement filters."""

    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("A non-empty grayscale or color image is required.")
    if image.dtype != np.uint8:
        raise ValueError("Enhancement expects an 8-bit image.")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Color images must have three or four channels.")


def to_grayscale(image: ImageArray) -> ImageArray:
    """Convert an image to 8-bit grayscale."""

    _validate_image(image)
    if image.ndim == 2:
        return image.copy()
    conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, conversion)


def enhance_bw(image: ImageArray) -> ImageArray:
    """Create a high-contrast black-and-white scan using adaptive thresholding."""

    gray = to_grayscale(image)
    # A light blur suppresses camera noise without noticeably softening text.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def enhance_magic(image: ImageArray) -> ImageArray:
    """Remove broad shadows with background division and boost local contrast."""

    _validate_image(image)
    if image.ndim == 2:
        background = cv2.GaussianBlur(image, (0, 0), sigmaX=21, sigmaY=21)
        normalized = cv2.divide(image, background, scale=255)
        return cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(normalized)

    bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image
    background = cv2.GaussianBlur(bgr, (0, 0), sigmaX=21, sigmaY=21)
    normalized = cv2.divide(bgr, background, scale=255)
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, channel_a, channel_b)), cv2.COLOR_LAB2BGR)


def enhance_document(image: ImageArray, mode: EnhanceMode = "original") -> ImageArray:
    """Apply one of the supported document enhancement modes."""

    _validate_image(image)
    if mode == "original":
        return image.copy()
    if mode == "grayscale":
        return to_grayscale(image)
    if mode == "bw":
        return enhance_bw(image)
    if mode == "magic":
        return enhance_magic(image)
    raise ValueError(f"Unknown enhancement mode '{mode}'. Choose from: {', '.join(ENHANCE_MODES)}.")
