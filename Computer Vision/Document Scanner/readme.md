# Document Scanner

A lightweight, local document scanner built with Python, OpenCV, and Streamlit. It detects a photographed page, corrects its perspective, improves readability, and exports the result as PNG or PDF. Everything runs locally without a GPU, paid API, or internet connection after installation.

Created by **M Haroon Abbas**.

## Features

- Automatic page-corner detection with a safe rotated-rectangle fallback
- Full-resolution perspective correction
- EXIF orientation support for phone photos
- Four enhancement modes: original, grayscale, black and white, and magic
- Smooth draggable crop box for quick manual correction
- Four-corner perspective picker for heavily angled pages
- Multiple image uploads and side-by-side previews
- Individual PNG/PDF downloads and combined multi-page PDF export
- Command-line interface for scripts and batch workflows
- CPU-only processing designed for laptops with 8 GB RAM

## Requirements

- Python 3.10 or newer
- The existing `pyglobal` Python environment
- No GPU is required

## Installation

Open a terminal in the project directory and activate `pyglobal`:

```bash
pyglobal
python -m pip install -r requirements.txt
```

This project intentionally uses the existing `pyglobal` environment; it does not require a new virtual environment.

## Streamlit web app

Start the application:

```bash
streamlit run app.py
```

Then:

1. Upload one or more PNG or JPEG photos.
2. Select an enhancement mode for each page.
3. Check the detected green page boundary and scanned preview.
4. If necessary, enable **Adjust corners on the image**.
5. Use **Drag a crop box** for smooth rectangular cropping, or **Pick 4 perspective corners** for an angled page.
6. Download each page as PNG/PDF or combine all pages into one PDF.

The perspective picker uses this order: top-left, top-right, bottom-right, bottom-left.

## Command-line interface

Straighten a photo and save a PNG:

```bash
python cli.py samples/page.jpg --mode original --output scanned.png
```

Create a clean black-and-white scan:

```bash
python cli.py samples/page.jpg --mode bw --output scanned.png
```

Create a shadow-corrected PDF:

```bash
python cli.py samples/page.jpg --mode magic --output scanned.pdf
```

If `--output` is omitted, the CLI creates `<input-name>_scanned.png` beside the input image.

Available modes:

| Mode | Result |
| --- | --- |
| `original` | Perspective-corrected color image without cleanup |
| `grayscale` | Single-channel grayscale scan |
| `bw` | Adaptive black-and-white threshold for text documents |
| `magic` | Shadow removal through background division plus local contrast enhancement |

## Processing pipeline

1. A detection copy is resized to approximately 800 pixels high. The original-to-copy scale is retained.
2. The copy is converted to grayscale, blurred, processed with Canny edge detection, and dilated to connect small gaps.
3. Contours are sorted by area. The largest credible four-point polygon becomes the page boundary.
4. If no four-point page is found, the detector tries the largest credible contour's minimum-area rectangle.
5. If detection remains uncertain, the original image is returned with a clear warning instead of failing or applying a bad crop.
6. Corners are ordered as top-left, top-right, bottom-right, and bottom-left.
7. OpenCV applies the perspective transform to the original full-resolution image.
8. The selected cleanup filter is applied and the result is encoded as PNG or PDF.

## Project structure

```text
scanner/
  __init__.py
  detect.py       Page-corner detection
  warp.py         Perspective transformation
  enhance.py      Cleanup filters
  export.py       PNG and multi-page PDF export
app.py            Streamlit interface
cli.py            Command-line interface
tests/            Unit and end-to-end tests
samples/          Example phone photos used by pipeline tests
requirements.txt  Runtime and test dependencies
```

Each module under `scanner/` can be imported and used without Streamlit.

## Tests

Run all tests in `pyglobal`:

```bash
python -m pytest -q
```

The suite covers corner ordering, perspective correction, all enhancement modes, EXIF orientation, PNG/PDF export, the CLI, Streamlit startup, manual selection calculations, and the complete pipeline on photos in `samples/`.

## Performance

Detection works on an 800-pixel-high copy while warping remains full resolution. During development, a synthetic 12 MP photo completed detection, perspective correction, and the heaviest `magic` enhancement in about 0.6 seconds on CPU. Actual speed depends on the processor, image content, and export format.

## Troubleshooting

- **The detected boundary is wrong:** enable manual adjustment. Use the draggable crop box for a nearly straight page or the four-corner picker for perspective correction.
- **No page was detected:** the app deliberately keeps the original image. Select the page manually instead.
- **The page is rotated:** phone EXIF orientation is applied automatically. Re-saving the source photo can repair invalid EXIF metadata.
- **Text looks washed out:** try `bw`; for uneven lighting or shadows, try `magic`.
- **Streamlit does not show a newly installed component:** stop the server completely and run `streamlit run app.py` again.
