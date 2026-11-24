"""Tests for PNG and PDF export."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pikepdf import Pdf

from scanner.export import pdf_bytes, png_bytes, save_pdf, save_png


def test_png_bytes_preserve_bgr_colors() -> None:
    bgr_red = np.zeros((20, 30, 3), dtype=np.uint8)
    bgr_red[:, :] = (0, 0, 255)

    encoded = png_bytes(bgr_red)

    with Image.open(BytesIO(encoded)) as result:
        assert result.format == "PNG"
        assert result.size == (30, 20)
        assert result.getpixel((0, 0)) == (255, 0, 0)


def test_pdf_bytes_create_multiple_pages() -> None:
    first = np.full((80, 60), 255, dtype=np.uint8)
    second = np.full((100, 70, 3), 100, dtype=np.uint8)

    encoded = pdf_bytes([first, second])

    assert encoded.startswith(b"%PDF")
    with Pdf.open(BytesIO(encoded)) as document:
        assert len(document.pages) == 2


def test_pdf_rejects_empty_page_list() -> None:
    with pytest.raises(ValueError, match="At least one image"):
        pdf_bytes([])


def test_save_helpers_create_files(tmp_path: Path) -> None:
    image = np.full((50, 40, 3), 200, dtype=np.uint8)
    png_path = tmp_path / "nested" / "scan.png"
    pdf_path = tmp_path / "document.pdf"

    assert save_png(png_path, image) == png_path
    assert save_pdf(pdf_path, [image, image]) == pdf_path
    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert pdf_path.read_bytes().startswith(b"%PDF")
