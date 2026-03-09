import io
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
MODEL_PATH  = os.path.join("model", "chest_xray_densenet121.keras")
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA"]
DISCLAIMER  = (
    "⚠️ This model is for educational purposes only and must NOT "
    "be used for clinical diagnosis. Always consult a qualified medical professional."
)

CLASS_META = {
    "COVID19": {
        "emoji": "🦠",
        "color": "#E74C3C",
        "description": "COVID-19 viral infection pattern detected. "
                       "Typically presents as bilateral ground-glass opacities."
    },
    "NORMAL": {
        "emoji": "✅",
        "color": "#2ECC71",
        "description": "No significant abnormality detected. "
                       "Lung fields appear clear."
    },
    "PNEUMONIA": {
        "emoji": "🫁",
        "color": "#E67E22",
        "description": "Bacterial/viral pneumonia pattern detected. "
                       "Typically presents as lobar consolidation."
    },
}

# ══════════════════════════════════════════════════════════════════
#  APP INIT
# ══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Chest X-Ray Classifier API",
    description="COVID-19 vs Pneumonia vs Normal — DenseNet121 Transfer Learning",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL ON STARTUP
# ══════════════════════════════════════════════════════════════════
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Ensure chest_xray_densenet121.keras is inside the /model folder."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded → {MODEL_PATH}")
    print(f"   Input shape  : {model.input_shape}")
    print(f"   Output shape : {model.output_shape}")


# ══════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Load image bytes → PIL RGB → resize 224x224 →
    float32 array → DenseNet121 preprocess → add batch dim.
    Returns shape (1, 224, 224, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = img_to_array(img).astype(np.float32)   # (224, 224, 3)
    arr = preprocess_input(arr)                  # DenseNet121 normalisation
    return np.expand_dims(arr, axis=0)           # (1, 224, 224, 3)


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════

# ── Serve UI ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main HTML UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ── Predict ───────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a chest X-ray image upload (JPEG / PNG).

    Returns:
        predicted_class     : COVID19 | NORMAL | PNEUMONIA
        confidence          : float 0–1
        all_probabilities   : scores for all 3 classes
        description         : brief clinical pattern note
        color               : hex color for UI badge
        disclaimer          : mandatory educational warning
    """
    # ── Validate ──────────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload JPEG or PNG only."
        )

    # ── Read & Preprocess ─────────────────────────────────────────
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        input_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

    # ── Inference ─────────────────────────────────────────────────
    predictions   = model.predict(input_array, verbose=0)[0]   # shape (3,)
    pred_idx      = int(np.argmax(predictions))
    predicted_cls = CLASS_NAMES[pred_idx]
    confidence    = float(predictions[pred_idx])

    all_probs = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(CLASS_NAMES, predictions)
    }

    return JSONResponse(content={
        "predicted_class"   : predicted_cls,
        "confidence"        : round(confidence * 100, 2),      # as percentage
        "all_probabilities" : all_probs,                       # as percentages
        "emoji"             : CLASS_META[predicted_cls]["emoji"],
        "color"             : CLASS_META[predicted_cls]["color"],
        "description"       : CLASS_META[predicted_cls]["description"],
        "disclaimer"        : DISCLAIMER
    })


# ── Health Check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick liveness check for the API."""
    return {
        "status"      : "ok",
        "model_loaded": model is not None,
        "model_path"  : MODEL_PATH,
        "classes"     : CLASS_NAMES
    }
