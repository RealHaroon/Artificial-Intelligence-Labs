"""Streamlit interface for the local document scanner."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates

from scanner.detect import DetectionResult, find_page_corners, order_corners
from scanner.enhance import ENHANCE_MODES, EnhanceMode, enhance_document
from scanner.export import pdf_bytes, png_bytes
from scanner.warp import warp_document


CORNER_NAMES = ("Top left", "Top right", "Bottom right", "Bottom left")


def decode_image(content: bytes) -> np.ndarray:
    """Decode uploaded image bytes as BGR and apply EXIF orientation."""

    try:
        with Image.open(BytesIO(content)) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Could not decode the uploaded image: {error}") from error


def default_corners(image: np.ndarray, detection: DetectionResult) -> np.ndarray:
    """Return detected corners, or the full image boundary when detection fails."""

    if detection.found:
        return detection.corners.copy()
    height, width = image.shape[:2]
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def process_page(
    image: np.ndarray,
    detection: DetectionResult,
    mode: EnhanceMode,
    manual_corners: np.ndarray | None = None,
) -> np.ndarray:
    """Warp and enhance one page, safely retaining it when detection fails."""

    corners = manual_corners if manual_corners is not None else detection.corners
    straightened = warp_document(image, corners) if corners is not None else image.copy()
    return enhance_document(straightened, mode)


def draw_corner_overlay(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Draw the selected page boundary and numbered corners for preview."""

    preview = image.copy()
    points = np.rint(order_corners(corners)).astype(np.int32)
    cv2.polylines(preview, [points], True, (40, 210, 40), max(2, image.shape[0] // 400))
    radius = max(5, image.shape[0] // 120)
    for number, point in enumerate(points, start=1):
        location = tuple(point)
        cv2.circle(preview, location, radius, (30, 80, 255), -1)
        cv2.putText(
            preview,
            str(number),
            (location[0] + radius, location[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, image.shape[0] / 1400),
            (30, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def apply_corner_click(
    points: np.ndarray,
    click: dict[str, int | float],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Map a display click to the image and append or move a page corner."""

    display_width = float(click.get("width", 0))
    display_height = float(click.get("height", 0))
    if display_width <= 0 or display_height <= 0:
        raise ValueError("The corner picker returned an invalid display size.")

    image_height, image_width = image_shape[:2]
    x = np.clip(float(click["x"]) * image_width / display_width, 0, image_width - 1)
    y = np.clip(float(click["y"]) * image_height / display_height, 0, image_height - 1)
    clicked = np.array([x, y], dtype=np.float32)
    current = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    if len(current) < 4:
        return np.vstack((current, clicked))

    # Once all corners exist, clicking near a corner moves the nearest handle.
    updated = current.copy()
    nearest = int(np.argmin(np.linalg.norm(updated - clicked, axis=1)))
    updated[nearest] = clicked
    return updated


def draw_editor_overlay(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Draw complete or in-progress corner selection on an image."""

    current = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(current) == 4:
        return draw_corner_overlay(image, current)

    preview = image.copy()
    integer_points = np.rint(current).astype(np.int32)
    if len(integer_points) > 1:
        cv2.polylines(preview, [integer_points], False, (40, 210, 40), max(2, image.shape[0] // 400))
    radius = max(5, image.shape[0] // 120)
    for number, point in enumerate(integer_points, start=1):
        location = tuple(point)
        cv2.circle(preview, location, radius, (30, 80, 255), -1)
        cv2.putText(
            preview,
            str(number),
            (location[0] + radius, location[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, image.shape[0] / 1400),
            (30, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def _visual_corner_editor(image: np.ndarray, automatic: np.ndarray, key_prefix: str) -> np.ndarray:
    """Render a cursor-based perspective-corner editor."""

    points_key = f"{key_prefix}_points"
    event_key = f"{key_prefix}_last_event"
    if points_key not in st.session_state:
        st.session_state[points_key] = automatic.tolist()

    reset_column, fresh_column = st.columns(2)
    if reset_column.button("Use automatic corners", key=f"{key_prefix}_auto", use_container_width=True):
        st.session_state[points_key] = automatic.tolist()
        st.session_state.pop(event_key, None)
        st.rerun()
    if fresh_column.button("Pick 4 from scratch", key=f"{key_prefix}_fresh", use_container_width=True):
        st.session_state[points_key] = []
        st.session_state.pop(event_key, None)
        st.rerun()

    points = np.asarray(st.session_state[points_key], dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        next_name = CORNER_NAMES[len(points)]
        st.info(f"Click {next_name.lower()} ({len(points) + 1} of 4).")
    else:
        st.caption("Click near any corner to move its nearest numbered handle.")

    overlay = draw_editor_overlay(image, points)
    click = streamlit_image_coordinates(
        Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
        key=f"{key_prefix}_picker",
        use_column_width="always",
        cursor="crosshair",
        png_compression_level=3,
    )
    if click is not None and click.get("unix_time") != st.session_state.get(event_key):
        st.session_state[event_key] = click.get("unix_time")
        st.session_state[points_key] = apply_corner_click(points, click, image.shape).tolist()
        st.rerun()

    return points


def _drag_crop_editor(image: np.ndarray, automatic: np.ndarray, key_prefix: str) -> np.ndarray:
    """Render a smooth browser-side crop box with draggable handles."""

    generation_key = f"{key_prefix}_crop_generation"
    if generation_key not in st.session_state:
        st.session_state[generation_key] = 0
    if st.button("Reset to detected area", key=f"{key_prefix}_crop_reset", use_container_width=True):
        st.session_state[generation_key] += 1
        st.rerun()

    left, top = automatic.min(axis=0)
    right, bottom = automatic.max(axis=0)
    rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    st.caption("Drag inside the box to move it, or drag its handles to resize it.")
    box = st_cropper(
        rgb_image,
        realtime_update=True,
        default_coords=(int(left), int(right), int(top), int(bottom)),
        box_color="#27c93f",
        aspect_ratio=None,
        return_type="box",
        key=f"{key_prefix}_smooth_crop_{st.session_state[generation_key]}",
        should_resize_image=True,
        stroke_width=3,
    )
    crop_left = float(box["left"])
    crop_top = float(box["top"])
    crop_right = crop_left + float(box["width"])
    crop_bottom = crop_top + float(box["height"])
    return np.array(
        [
            [crop_left, crop_top],
            [crop_right, crop_top],
            [crop_right, crop_bottom],
            [crop_left, crop_bottom],
        ],
        dtype=np.float32,
    )


@st.cache_data(show_spinner=False)
def _decode_and_detect(content: bytes) -> tuple[np.ndarray, DetectionResult]:
    """Cache upload decoding and automatic page detection."""

    image = decode_image(content)
    return image, find_page_corners(image)


def _render_page(content: bytes, filename: str, index: int) -> np.ndarray | None:
    """Render controls and downloads for one uploaded page."""

    digest = hashlib.sha256(content).hexdigest()[:12]
    key_prefix = f"page_{index}_{digest}"
    st.subheader(f"Page {index + 1}: {filename}")

    try:
        image, detection = _decode_and_detect(content)
    except ValueError as error:
        st.error(str(error))
        return None

    mode = st.selectbox(
        "Enhancement mode",
        ENHANCE_MODES,
        index=ENHANCE_MODES.index("magic"),
        key=f"{key_prefix}_mode",
    )
    detected_corners = default_corners(image, detection)
    use_manual = st.checkbox(
        "Adjust corners on the image",
        key=f"{key_prefix}_manual",
        help="Click directly on the photo to move the perspective-crop corners.",
    )

    selected_corners = detected_corners
    if use_manual:
        with st.expander("Visual corner editor", expanded=True):
            adjustment_style = st.radio(
                "Selection method",
                ("Drag a crop box", "Pick 4 perspective corners"),
                horizontal=True,
                key=f"{key_prefix}_adjustment_style",
            )
            if adjustment_style == "Drag a crop box":
                selected_corners = _drag_crop_editor(image, detected_corners, key_prefix)
            else:
                selected_corners = _visual_corner_editor(image, detected_corners, key_prefix)

    if detection.method == "contour":
        st.success(detection.message)
    elif detection.method == "min_area_rect":
        st.warning(detection.message)
    else:
        st.warning(detection.message + " Enable manual adjustment to select the page.")

    try:
        manual = selected_corners if use_manual and len(selected_corners) == 4 else None
        result = process_page(image, detection, mode, manual)
        preview_corners = selected_corners if use_manual and len(selected_corners) == 4 else detected_corners
        preview = draw_corner_overlay(image, preview_corners)
    except (ValueError, cv2.error) as error:
        st.error(f"Could not process this page: {error}")
        return None

    original_column, result_column = st.columns(2)
    with original_column:
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Original and selected corners")
    with result_column:
        display = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if result.ndim == 3 else result
        st.image(display, caption=f"Scanned result · {mode}")

    safe_stem = Path(filename).stem or f"page_{index + 1}"
    png_column, pdf_column = st.columns(2)
    with png_column:
        st.download_button(
            "Download PNG",
            data=png_bytes(result),
            file_name=f"{safe_stem}_scanned.png",
            mime="image/png",
            key=f"{key_prefix}_png",
            use_container_width=True,
        )
    with pdf_column:
        st.download_button(
            "Download PDF",
            data=pdf_bytes([result]),
            file_name=f"{safe_stem}_scanned.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_pdf",
            use_container_width=True,
        )
    st.divider()
    return result


def main() -> None:
    """Render the document scanner application."""

    st.set_page_config(page_title="Document Scanner", page_icon="📄", layout="wide")
    st.title("Document Scanner")
    st.write("Upload photographed pages to straighten, clean, and export them locally.")
    uploads = st.file_uploader(
        "Upload one or more page photos",
        type=("png", "jpg", "jpeg"),
        accept_multiple_files=True,
    )
    if not uploads:
        st.info("Choose one or more PNG or JPEG photos to begin.")
        return

    results: list[np.ndarray] = []
    for index, upload in enumerate(uploads):
        result = _render_page(upload.getvalue(), upload.name, index)
        if result is not None:
            results.append(result)

    if len(results) > 1:
        st.subheader("Combined document")
        combine = st.checkbox("Combine all successfully scanned pages into one PDF", value=True)
        if combine:
            st.download_button(
                "Download combined PDF",
                data=pdf_bytes(results),
                file_name="scanned_document.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
