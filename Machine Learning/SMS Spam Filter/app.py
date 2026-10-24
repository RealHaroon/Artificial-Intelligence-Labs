from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from pathlib import Path

# 1. Load the Model and Vectorizer (paths relative to this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.joblib"
VECTORIZER_PATH = BASE_DIR / "model" / "vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(title="SMS Spam Detector API")

# 2. Setup templates folder for UI
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 3. Define Input Schema
class SpamRequest(BaseModel):
    message: str

# 4. Route to serve the HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 5. API endpoint for prediction
@app.post("/predict")
async def predict_spam(data: SpamRequest):
    # Transform the raw text into numerical features using the saved vectorizer
    text_vectorized = vectorizer.transform([data.message])
    
    # Make the prediction
    prediction = model.predict(text_vectorized)
    
    # Get the probability score to show confidence
    proba = model.predict_proba(text_vectorized)[0][prediction[0]]
    
    # Format the response
    label = "SPAM" if prediction[0] == 1 else "HAM (Clean)"
    
    return {
        "label": label, 
        "confidence": f"{proba * 100:.2f}%"
    }