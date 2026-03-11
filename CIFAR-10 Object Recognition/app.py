import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os
import base64

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CIFAR-10 Object Recognition",
    description="CNN-based image classifier for 10 object categories.",
    version="1.0.0"
)

# ── Templates Folder ───────────────────────────────────────────────────────────
templates = Jinja2Templates(directory="templates")

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

CLASS_EMOJIS = {
    "Airplane": "✈️", "Automobile": "🚗", "Bird": "🐦", "Cat": "🐱",
    "Deer": "🦌", "Dog": "🐶", "Frog": "🐸", "Horse": "🐴",
    "Ship": "🚢", "Truck": "🚛"
}

IMG_SIZE = (32, 32)

# ── Load Model Once at Startup ─────────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "cifar10_cnn_best.keras")

# Fallback to alternate filename if needed
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("model", "cifar10_cnn.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No model file found in /model. "
        f"Expected 'cifar10_cnn_best.keras' or 'cifar10_cnn.keras'."
    )

model = keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from: {MODEL_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Home Page ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "class_names": CLASS_NAMES}
    )


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "✅ running",
        "model" : MODEL_PATH,
        "classes": CLASS_NAMES
    }


# ── Predict Endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validate file type
    allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Please upload a JPEG or PNG image."
        )

    # 2. Read and decode image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file.")

    # 3. Convert image to base64 for preview in UI response
    preview_b64 = base64.b64encode(contents).decode("utf-8")
    mime_type   = file.content_type

    # 4. Resize to CIFAR-10 input size
    image_resized = image.resize(IMG_SIZE, Image.LANCZOS)

    # 5. Normalize + batch dimension
    img_array = np.array(image_resized, dtype="float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 32, 32, 3)

    # 6. Inference
    preds = model.predict(img_array, verbose=0)[0]  # (10,)

    predicted_idx   = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence      = float(preds[predicted_idx])

    # 7. Build sorted probability list
    all_probs = sorted(
        [
            {
                "class"      : CLASS_NAMES[i],
                "emoji"      : CLASS_EMOJIS[CLASS_NAMES[i]],
                "probability": round(float(preds[i]) * 100, 2)
            }
            for i in range(len(CLASS_NAMES))
        ],
        key=lambda x: x["probability"],
        reverse=True
    )

    return JSONResponse(content={
        "predicted_class": predicted_class,
        "emoji"          : CLASS_EMOJIS[predicted_class],
        "confidence"     : round(confidence * 100, 2),
        "all_probs"      : all_probs,
        "preview_b64"    : preview_b64,
        "mime_type"      : mime_type
    })
