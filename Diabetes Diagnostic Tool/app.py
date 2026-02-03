from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

artifact = joblib.load("D:\\Github\Artificial-Intelligence-Labs\\Diabetes Diagnostic Tool\\model\diabetes_classification_model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

app = FastAPI(title="Diabetes Prediction API")



templates = Jinja2Templates(directory="templates") 

class Patient(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})  

@app.post("/predict-diabetes")
def predict_diabetes(p: Patient):
    row = pd.DataFrame([p.model_dump()])[feature_names]

    # Optional but recommended if you treated zeros as missing during training:
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        row[c] = row[c].replace(0, np.nan)

    proba = float(model.predict_proba(row)[0, 1])
    pred_class = int(proba >= threshold)
    return {
        "diabetes_probability": proba,
        "threshold_used": threshold,
        "predicted_class": pred_class
    }
