from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

# 1. Load the Model and Vectorizer using your exact file paths
# (Using the 'r' prefix for raw strings to handle Windows backslashes safely)
MODEL_PATH = r"D:\Talha\SMS Spam Filter\model\spam_model.joblib"
VECTORIZER_PATH = r"D:\Talha\SMS Spam Filter\model\vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = FastAPI(title="SMS Spam Detector API")

# 2. Setup templates folder for UI
templates = Jinja2Templates(directory="templates")

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