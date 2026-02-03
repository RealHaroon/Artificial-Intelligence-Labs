from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import joblib

MODEL_PATH = "D:\\Github\\Artificial-Intelligence-Labs\\Iris Species Classifier\\model\\iris_knn_pipeline.joblib"
HTML_PATH = "D:\\Github\Artificial-Intelligence-Labs\\Iris Species Classifier\\template\index.html"

app = FastAPI(title="Iris KNN Classifier API")

# Load model once at startup
model = joblib.load(MODEL_PATH)
# target names order is the same as model.classes_ indices for predict_proba output
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

class IrisRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def home():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict(req: IrisRequest):
    X = np.array([[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]], dtype=float)

    pred_idx = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]  # shape (3,)

    probs = {TARGET_NAMES[i]: float(proba[i]) for i in range(len(TARGET_NAMES))}
    return {
        "predicted_class_name": TARGET_NAMES[pred_idx],
        "class_probabilities": probs
    }
