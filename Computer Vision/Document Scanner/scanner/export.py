"""Export scanned document images as PNG files or multi-page PDFs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


ImageArray = NDArray[np.uint8]


def _validate_image(image: ImageArray) -> None:
    """Validate an image supplied for export."""

    if not isinstance(image, np.ndarray) or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Export requires a non-empty 8-bit image.")
    if image.ndim == 2:
        return
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Export supports grayscale, BGR, or BGRA images.")


def _to_pillow(image: ImageArray, preserve_alpha: bool = True) -> Image.Image:
    """Convert an OpenCV image to a Pillow image."""

    _validate_image(image)
    if image.ndim == 2:
        return Image.fromarray(image, mode="L")
    if image.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mode="RGB")

    rgba = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), mode="RGBA")
    if preserve_alpha:
        return rgba
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def png_bytes(image: ImageArray, compress_level: int = 6) -> bytes:
    """Encode one OpenCV image as PNG bytes."""

    if not 0 <= compress_level <= 9:
        raise ValueError("PNG compression level must be between 0 and 9.")
    buffer = BytesIO()
    _to_pillow(image).save(buffer, format="PNG", compress_level=compress_level)
    return buffer.getvalue()


def pdf_bytes(images: Iterable[ImageArray], resolution: float = 200.0) -> bytes:
    """Encode one or more OpenCV images as a multi-page PDF."""

    if resolution <= 0:
        raise ValueError("PDF resolution must be positive.")
    pages = [_to_pillow(image, preserve_alpha=False) for image in images]
    if not pages:
        raise ValueError("At least one image is required to create a PDF.")

    # PDF output is most predictable with RGB pages; this also handles scans
    # whose modes differ within the same document.
    rgb_pages = [page.convert("RGB") for page in pages]
    buffer = BytesIO()
    rgb_pages[0].save(
        buffer,
        format="PDF",
        save_all=True,
        append_images=rgb_pages[1:],
        resolution=resolution,
    )
    return buffer.getvalue()


def _write_bytes(path: Path, content: bytes) -> None:
    """Write exported content, creating its parent directory when needed."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except (PermissionError, OSError) as error:
        raise ValueError(f"Could not save '{path}': {error}") from error


def save_png(path: str | Path, image: ImageArray, compress_level: int = 6) -> Path:
    """Save one OpenCV image as a PNG file."""

    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("PNG output must use a .png extension.")
    _write_bytes(output, png_bytes(image, compress_level))
    return output


def save_pdf(
    path: str | Path,
    images: Iterable[ImageArray],
    resolution: float = 200.0,
) -> Path:
    """Save one or more OpenCV images as a multi-page PDF."""

    output = Path(path)
    if output.suffix.lower() != ".pdf":
        raise ValueError("PDF output must use a .pdf extension.")
    _write_bytes(output, pdf_bytes(images, resolution))
    return output
