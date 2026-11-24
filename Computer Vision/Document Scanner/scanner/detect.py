"""Detect page boundaries in photographs using OpenCV contours."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class DetectionResult:
    """Result of automatic page-corner detection."""

    corners: FloatArray | None
    method: str
    message: str

    @property
    def found(self) -> bool:
        """Return whether usable page corners were found."""

        return self.corners is not None


def order_corners(points: NDArray[np.number]) -> FloatArray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""

    corners = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if corners.shape != (4, 2):
        raise ValueError("Exactly four 2D corner points are required.")
    if not np.isfinite(corners).all():
        raise ValueError("Corner coordinates must be finite numbers.")

    # Walk around the centroid, then begin at the point nearest the image's
    # top-left. This remains reliable when perspective makes a bottom corner
    # appear higher than the opposite top corner.
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0).astype(np.float32)

    if cv2.contourArea(ordered, oriented=True) < 0:
        ordered = ordered[[0, 3, 2, 1]]

    if abs(cv2.contourArea(ordered)) < 1.0:
        raise ValueError("Corner points do not form a valid quadrilateral.")
    return ordered


def _resize_for_detection(image: ImageArray, target_height: int) -> tuple[ImageArray, float, float]:
    """Return a detection-sized image and original-to-copy scale factors."""

    height, width = image.shape[:2]
    if height <= target_height:
        return image.copy(), 1.0, 1.0

    new_width = max(1, round(width * target_height / height))
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized, width / new_width, height / target_height


def _edge_map(image: ImageArray) -> ImageArray:
    """Create a connected edge map suited to page-boundary detection."""

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(edges, kernel, iterations=1)


def find_page_corners(
    image: ImageArray,
    target_height: int = 800,
    minimum_area_ratio: float = 0.05,
) -> DetectionResult:
    """Find page corners, falling back to the largest contour's rotated box."""

    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim not in (2, 3):
        return DetectionResult(None, "none", "No valid image was provided; using the original image.")
    if target_height <= 0:
        return DetectionResult(None, "none", "Detection height must be positive; using the original image.")

    try:
        resized, scale_x, scale_y = _resize_for_detection(image, target_height)
        edges = _edge_map(resized)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        minimum_area = resized.shape[0] * resized.shape[1] * minimum_area_ratio

        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                break
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                corners = approximation.reshape(4, 2).astype(np.float32)
                corners *= np.array([scale_x, scale_y], dtype=np.float32)
                return DetectionResult(
                    order_corners(corners),
                    "contour",
                    "Page detected automatically.",
                )

        if contours and cv2.contourArea(contours[0]) >= minimum_area:
            box = cv2.boxPoints(cv2.minAreaRect(contours[0])).astype(np.float32)
            box *= np.array([scale_x, scale_y], dtype=np.float32)
            return DetectionResult(
                order_corners(box),
                "min_area_rect",
                "A clear four-corner page was not found; using the largest detected region.",
            )
    except (cv2.error, ValueError, TypeError):
        pass

    return DetectionResult(
        None,
        "none",
        "No page boundary was found; using the original image.",
    )
