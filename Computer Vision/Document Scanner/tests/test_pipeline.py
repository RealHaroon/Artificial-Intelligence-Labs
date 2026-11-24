"""End-to-end tests using the real photos in the samples directory."""

from pathlib import Path

import pytest

from cli import load_image
from scanner.detect import find_page_corners
from scanner.enhance import enhance_document
from scanner.export import pdf_bytes, png_bytes
from scanner.warp import warp_document


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLE_IMAGES = sorted(
    path for path in SAMPLES_DIR.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
)


def test_sample_images_are_available() -> None:
    assert SAMPLE_IMAGES, "Add at least one PNG or JPEG photo to samples/."


@pytest.mark.parametrize("sample_path", SAMPLE_IMAGES, ids=lambda path: path.name)
def test_full_pipeline_on_sample_image(sample_path: Path) -> None:
    image = load_image(sample_path)
    detection = find_page_corners(image)
    straightened = warp_document(image, detection.corners) if detection.found else image.copy()
    result = enhance_document(straightened, "magic")

    assert result.size > 0
    assert result.shape[0] >= 2
    assert result.shape[1] >= 2
    assert png_bytes(result).startswith(b"\x89PNG")
    assert pdf_bytes([result]).startswith(b"%PDF")
