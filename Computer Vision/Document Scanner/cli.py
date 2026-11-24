"""Command-line interface for straightening document photos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from scanner.detect import find_page_corners
from scanner.enhance import ENHANCE_MODES, enhance_document
from scanner.export import save_pdf, save_png
from scanner.warp import warp_document


def load_image(path: Path) -> np.ndarray:
    """Load an image as BGR while applying its EXIF orientation."""

    try:
        with Image.open(path) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except (FileNotFoundError, PermissionError, UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image '{path}': {error}") from error


def save_image(path: Path, image: np.ndarray) -> None:
    """Save a BGR image as PNG or JPEG."""

    suffix = path.suffix.lower()
    if suffix == ".png":
        save_png(path, image)
        return
    if suffix not in {".jpg", ".jpeg"}:
        raise ValueError("Output must use a .png, .jpg, or .jpeg extension.")

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
    try:
        Image.fromarray(rgb).save(path)
    except (PermissionError, OSError) as error:
        raise ValueError(f"Could not save image '{path}': {error}") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Detect and straighten a photographed document.")
    parser.add_argument("input", type=Path, help="Input PNG or JPEG photo")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .png, .jpg, or .pdf path (default: <input>_scanned.png)",
    )
    parser.add_argument(
        "--mode",
        choices=ENHANCE_MODES,
        default="original",
        help="Cleanup filter to apply (default: original)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scanner CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    output = args.output or args.input.with_name(f"{args.input.stem}_scanned.png")

    try:
        image = load_image(args.input)
        detection = find_page_corners(image)
        if detection.found:
            straightened = warp_document(image, detection.corners)
        else:
            straightened = image.copy()
        result = enhance_document(straightened, args.mode)
        if output.suffix.lower() == ".pdf":
            save_pdf(output, [result])
        else:
            save_image(output, result)
    except (ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(detection.message)
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
