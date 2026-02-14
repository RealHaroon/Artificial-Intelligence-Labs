import io
import json
import numpy as np
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

IMG_SIZE = (64, 64)
THRESHOLD = 0.5

app = FastAPI(title="Malaria Detection (Educational)")
templates = Jinja2Templates(directory="templates")

# Load artifacts once at startup
model = tf.keras.models.load_model("D:\\Github\\Artificial-Intelligence-Labs\\Malaria Cell Classification System\\model\\malaria_cnn.keras")
with open("D:\\Github\\Artificial-Intelligence-Labs\\Malaria Cell Classification System\\model\\class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)  # ["Uninfected", "Parasitized"]


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Match training preprocessing: RGB, resize 64x64, normalize /255."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,64,64,3)
    return arr


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Render UI page
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic content-type check (optional but helpful)
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG/JPG image")

    image_bytes = await file.read()
    x = preprocess_image_bytes(image_bytes)

    # Model outputs P(Parasitized) because your final layer is sigmoid with 1 unit
    prob_parasitized = float(model.predict(x, verbose=0)[0][0])
    pred = 1 if prob_parasitized >= THRESHOLD else 0

    return {
        "label": CLASS_NAMES[pred],
        "probability_parasitized": prob_parasitized,
        "threshold": THRESHOLD,
        "disclaimer": "Educational only — not for medical diagnosis."
    }
