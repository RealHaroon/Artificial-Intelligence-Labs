from pathlib import Path
import base64
import io

import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import HTTPException 
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mnist.keras"

app = FastAPI()

# Static + templates (recommended by FastAPI docs)
# app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# main.py
from PIL import Image, ImageOps
import numpy as np

def preprocess_pil(img: Image.Image) -> np.ndarray:
    # 1) grayscale
    img = img.convert("L")

    # 2) to float in [0,1]
    arr = np.array(img).astype("float32") / 255.0

    # 3) auto-invert if background looks white
    # (canvas usually black bg, so mean should be low; this keeps it robust)
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    # 4) threshold -> find bounding box of digit
    mask = arr > 0.08
    if not mask.any():
        return np.zeros((1, 28, 28), dtype="float32")

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    digit = arr[y0:y1, x0:x1]

    # 5) resize digit to fit in 20x20 (keep aspect)
    h, w = digit.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    digit_img = Image.fromarray((digit * 255).astype("uint8"), mode="L")
    digit_img = digit_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 6) paste into a 28x28 canvas (with 4px margin like MNIST convention)
    canvas = Image.new("L", (28, 28), 0)
    left = 4 + (20 - new_w) // 2
    top  = 4 + (20 - new_h) // 2
    canvas.paste(digit_img, (left, top))

    # 7) center-of-mass shift to center
    canv = np.array(canvas).astype("float32") / 255.0
    total = canv.sum()
    if total > 0:
        yy, xx = np.indices((28, 28))
        cx = (xx * canv).sum() / total
        cy = (yy * canv).sum() / total
        dx = int(round(13.5 - cx))
        dy = int(round(13.5 - cy))

        # PIL affine translation: x' = x - dx, y' = y - dy  => data shifts by (dx,dy)
        canvas = canvas.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -dx, 0, 1, -dy),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    out = (np.array(canvas).astype("float32") / 255.0).reshape(1, 28, 28)
    return out


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(payload: dict):
    """
    Expects JSON: { "image": "data:image/png;base64,...." }  (from canvas)
    """
    data_url = payload.get("image", "")
    if "," in data_url:
        _, b64 = data_url.split(",", 1)
    else:
        b64 = data_url

    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes))

    x = preprocess_pil(img)

    if float(x.sum()) < 1.0:   # you can tune 1.0 -> 0.2 / 2.0 depending on your strokes
        raise HTTPException(status_code=400, detail="No digit drawn.")

    from pathlib import Path
    Path("debug").mkdir(exist_ok=True)
    Image.fromarray((x[0] * 255).astype("uint8"), mode="L").save("debug/last_28x28.png")

    logits = model(x, training=False).numpy()[0]
    probs = tf.nn.softmax(logits).numpy()

    pred = int(np.argmax(probs))
    conf = float(np.max(probs))

    return JSONResponse({
        "prediction": pred,
        "confidence": conf,
        "probs": probs.tolist()
    })
