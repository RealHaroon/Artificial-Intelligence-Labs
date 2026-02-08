from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Salary Prediction API")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

model = joblib.load('D:\\Github\\Artificial-Intelligence-Labs\\Salary Predictor\\model\\salary_model.pkl')

class ExperienceInput(BaseModel):
    years_experience: float

# 1. GET route to render the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

# 2. POST route that the HTML form will submit to
@app.post("/", response_class=HTMLResponse)
async def predict_from_ui(request: Request, years_experience: float = Form(...)):
    input_data = np.array([[years_experience]])
    prediction = model.predict(input_data)
    result = round(float(prediction.item()), 2)
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "prediction": result, 
        "years": years_experience
    })

# Keeping your original JSON endpoint for API testing
@app.post("/predict")
def predict_salary(data: ExperienceInput):
    input_data = np.array([[data.years_experience]])
    prediction = model.predict(input_data)
    return {
        "years_experience": data.years_experience,
        "predicted_salary": round(float(prediction.item()), 2)
    }