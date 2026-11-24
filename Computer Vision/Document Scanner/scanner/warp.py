"""Perspective correction for detected document pages."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .detect import order_corners


ImageArray = NDArray[np.uint8]


def calculate_output_size(corners: NDArray[np.number]) -> tuple[int, int]:
    """Calculate the output width and height from four page corners."""

    top_left, top_right, bottom_right, bottom_left = order_corners(corners)
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    output_width, output_height = int(round(width)), int(round(height))
    if output_width < 2 or output_height < 2:
        raise ValueError("The selected page area is too small to straighten.")
    return output_width, output_height


def warp_document(image: ImageArray, corners: NDArray[np.number]) -> ImageArray:
    """Straighten a document from four corners using a perspective transform."""

    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("A non-empty grayscale or color image is required.")

    source = order_corners(corners)
    width, height = calculate_output_size(source)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
