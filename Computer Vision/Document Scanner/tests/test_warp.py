"""Tests for perspective correction and the command-line workflow."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cli import load_image, main
from scanner.warp import calculate_output_size, warp_document


def test_calculate_output_size() -> None:
    corners = np.array([[100, 100], [500, 100], [500, 400], [100, 400]])

    assert calculate_output_size(corners) == (400, 300)


def test_warp_document_returns_expected_shape() -> None:
    image = np.zeros((600, 700, 3), dtype=np.uint8)
    corners = np.array([[120, 100], [570, 150], [520, 500], [80, 440]], dtype=np.int32)
    cv2.fillConvexPoly(image, corners, (255, 255, 255))

    warped = warp_document(image, corners)

    expected_width, expected_height = calculate_output_size(corners)
    assert warped.shape == (expected_height, expected_width, 3)
    assert warped[expected_height // 2, expected_width // 2].mean() > 245


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    input_path = tmp_path / "rotated.jpg"
    photo = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise when displayed.
    photo.save(input_path, exif=exif)

    loaded = load_image(input_path)

    assert loaded.shape == (40, 20, 3)


def test_cli_scans_an_image(tmp_path: Path) -> None:
    image = np.full((900, 1200, 3), 30, dtype=np.uint8)
    page = np.array([[180, 100], [1030, 160], [970, 800], [130, 720]], dtype=np.int32)
    cv2.fillConvexPoly(image, page, (245, 245, 245))
    input_path = tmp_path / "photo.jpg"
    output_path = tmp_path / "scan.png"
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(input_path)

    exit_code = main([str(input_path), "--output", str(output_path), "--mode", "bw"])

    assert exit_code == 0
    assert output_path.is_file()
    with Image.open(output_path) as result:
        assert result.width < image.shape[1]
        assert result.height < image.shape[0]
        assert result.mode == "L"


def test_cli_can_export_pdf(tmp_path: Path) -> None:
    image = np.full((500, 400, 3), 220, dtype=np.uint8)
    input_path = tmp_path / "page.png"
    output_path = tmp_path / "page.pdf"
    Image.fromarray(image).save(input_path)

    exit_code = main([str(input_path), "--output", str(output_path), "--mode", "grayscale"])

    assert exit_code == 0
    assert output_path.read_bytes().startswith(b"%PDF")
