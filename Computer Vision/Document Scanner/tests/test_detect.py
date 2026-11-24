"""Tests for page-boundary detection."""

import cv2
import numpy as np
import pytest

from scanner.detect import find_page_corners, order_corners


def test_order_corners_from_shuffled_points() -> None:
    points = np.array([[500, 390], [110, 400], [100, 80], [510, 70]])

    ordered = order_corners(points)

    expected = np.array([[100, 80], [510, 70], [500, 390], [110, 400]], dtype=np.float32)
    np.testing.assert_array_equal(ordered, expected)


def test_order_corners_rejects_wrong_number_of_points() -> None:
    with pytest.raises(ValueError, match="Exactly four"):
        order_corners(np.array([[0, 0], [1, 1], [2, 2]]))


def test_order_corners_handles_steep_perspective() -> None:
    points = np.array([[50, 220], [450, 600], [500, 260], [100, 100]])

    ordered = order_corners(points)

    expected = np.array([[100, 100], [500, 260], [450, 600], [50, 220]], dtype=np.float32)
    np.testing.assert_array_equal(ordered, expected)


def test_find_page_corners_on_synthetic_photo() -> None:
    image = np.full((1200, 1600, 3), 35, dtype=np.uint8)
    expected = np.array([[310, 170], [1320, 260], [1240, 1010], [220, 900]], dtype=np.int32)
    cv2.fillConvexPoly(image, expected, (245, 245, 245))
    cv2.polylines(image, [expected], True, (5, 5, 5), 10)

    result = find_page_corners(image)

    assert result.found
    assert result.method == "contour"
    np.testing.assert_allclose(result.corners, expected, atol=15)


def test_find_page_corners_returns_clear_failure() -> None:
    result = find_page_corners(np.zeros((500, 500, 3), dtype=np.uint8))

    assert not result.found
    assert result.method == "none"
    assert "original image" in result.message
