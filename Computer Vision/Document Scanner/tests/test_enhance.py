"""Tests for document enhancement modes."""

import cv2
import numpy as np
import pytest

from scanner.enhance import ENHANCE_MODES, enhance_document


@pytest.fixture
def document_image() -> np.ndarray:
    """Return a small document-like color image with text strokes."""

    image = np.full((240, 320, 3), 220, dtype=np.uint8)
    cv2.putText(image, "Document", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (25, 25, 25), 3)
    cv2.line(image, (35, 155), (285, 155), (50, 50, 50), 3)
    return image


def test_all_modes_return_uint8_images(document_image: np.ndarray) -> None:
    for mode in ENHANCE_MODES:
        result = enhance_document(document_image, mode)

        assert result.dtype == np.uint8
        assert result.shape[:2] == document_image.shape[:2]


def test_grayscale_mode_has_one_channel(document_image: np.ndarray) -> None:
    assert enhance_document(document_image, "grayscale").ndim == 2


def test_bw_mode_contains_only_black_and_white(document_image: np.ndarray) -> None:
    result = enhance_document(document_image, "bw")

    assert result.ndim == 2
    assert set(np.unique(result)).issubset({0, 255})


def test_magic_mode_reduces_smooth_shadow_variation() -> None:
    gradient = np.linspace(90, 220, 400, dtype=np.uint8)
    gray = np.tile(gradient, (250, 1))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    result = enhance_document(image, "magic")
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    assert result.shape == image.shape
    assert result_gray.std() < gray.std() * 0.5


def test_unknown_mode_is_rejected(document_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown enhancement mode"):
        enhance_document(document_image, "sepia")  # type: ignore[arg-type]
