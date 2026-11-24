"""Tests for Streamlit-independent application helpers."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from app import apply_corner_click, decode_image, default_corners, process_page
from scanner.detect import DetectionResult


def test_streamlit_app_starts_without_errors() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Document Scanner"


def test_decode_image_applies_exif_rotation() -> None:
    image = Image.new("RGB", (60, 30), "white")
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    decoded = decode_image(buffer.getvalue())

    assert decoded.shape == (60, 30, 3)


def test_default_corners_use_image_boundary_after_failure() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = DetectionResult(None, "none", "Not found")

    corners = default_corners(image, detection)

    expected = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32)
    np.testing.assert_array_equal(corners, expected)


def test_corner_click_scales_display_coordinates_and_moves_nearest() -> None:
    corners = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], dtype=np.float32)
    click = {"x": 190, "y": 140, "width": 200, "height": 150}

    updated = apply_corner_click(corners, click, (300, 400, 3))

    np.testing.assert_allclose(updated[2], [380, 280])
    np.testing.assert_array_equal(updated[:2], corners[:2])


def test_corner_click_builds_selection_from_scratch() -> None:
    click = {"x": 25, "y": 20, "width": 100, "height": 50}

    updated = apply_corner_click(np.empty((0, 2)), click, (200, 400, 3))

    np.testing.assert_allclose(updated, [[100, 80]])


def test_process_page_uses_manual_corners() -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    corners = np.array([[50, 40], [350, 40], [350, 260], [50, 260]], dtype=np.float32)
    cv2.rectangle(image, (50, 40), (350, 260), (255, 255, 255), -1)
    detection = DetectionResult(None, "none", "Not found")

    result = process_page(image, detection, "grayscale", corners)

    assert result.shape == (220, 300)
    assert result.mean() > 245
