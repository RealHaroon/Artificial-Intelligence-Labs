import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# 1. Load the model from your specific path
MODEL_PATH = r"D:\\Talha\\Real state Price Estimator\\model\\house_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI()

# 2. Setup templates folder for UI
templates = Jinja2Templates(directory="templates")

# Schema for validation
class HouseData(BaseModel):
    median_income: float
    house_age: float
    avg_rooms: float
    avg_bedrooms: float
    population: float
    avg_occupancy: float
    latitude: float
    longitude: float

# Route to serve the HTML UI
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for prediction
@app.post("/predict")
async def predict(data: HouseData):
    features = np.array([[
        data.median_income, data.house_age, data.avg_rooms,
        data.avg_bedrooms, data.population, data.avg_occupancy,
        data.latitude, data.longitude
    ]])
    
    prediction = model.predict(features)
    # Convert from $100k units to actual USD
    price = float(prediction[0]) * 100000
    return {"price": f"${price:,.2f}"}