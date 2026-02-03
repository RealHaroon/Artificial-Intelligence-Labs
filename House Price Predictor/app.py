from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI(title="House Price Predictor")

# Load templates
templates = Jinja2Templates(directory="templates")

# Load model & scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()

    # Extract values from form
    overall_qual = int(form["OverallQual"])
    gr_liv_area = float(form["GrLivArea"])
    garage_cars = int(form["GarageCars"])
    total_bsmt_sf = float(form["TotalBsmtSF"])
    year_built = int(form["YearBuilt"])

    # Prepare data
    data = np.array([[ 
        overall_qual,
        gr_liv_area,
        garage_cars,
        total_bsmt_sf,
        year_built
    ]])

    # Scale + Predict
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": round(float(prediction), 2)
        }
    )
