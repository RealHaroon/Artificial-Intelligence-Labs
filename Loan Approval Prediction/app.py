from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Loan Approval Predictor")

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "template"
MODEL_PATH = BASE_DIR / "model" / "loan_tree.pkl"

# Serve your CSS from the same folder (template/html.css)
app.mount("/static", StaticFiles(directory=str(TEMPLATE_DIR)), name="static")  # mounts /static [web:122]

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))  # Jinja2 templates [web:85]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(str(MODEL_PATH))


# Must match training columns (note: Self_Employed, not Self_Employment)
class LoanApplicant(BaseModel):
    Gender: str | None = None
    Married: str | None = None
    Dependents: str | None = None
    Education: str | None = None
    Self_Employed: str | None = None
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float | None = None
    Loan_Amount_Term: float | None = None
    Credit_History: float | None = None
    Property_Area: str | None = None


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None, "form": {}},
    )


@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    Gender: str | None = Form(None),
    Married: str | None = Form(None),
    Dependents: str | None = Form(None),
    Education: str | None = Form(None),
    Self_Employed: str | None = Form(None),
    ApplicantIncome: float = Form(...),
    CoapplicantIncome: float = Form(...),
    LoanAmount: float | None = Form(None),
    Loan_Amount_Term: float | None = Form(None),
    Credit_History: float | None = Form(None),
    Property_Area: str | None = Form(None),
):
    row = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }

    try:
        X = pd.DataFrame([row])
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0, 1])
        label = "Approved" if pred == 1 else "Not Approved"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": {"prediction": label, "approval_probability": round(proba, 4)},
                "error": None,
                "form": row,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": str(e), "form": row},
            status_code=400,
        )


@app.post("/predict-loan")
def predict_loan(applicant: LoanApplicant):
    X = pd.DataFrame([applicant.model_dump()])
    pred = int(model.predict(X)[0])
    label = "Approved" if pred == 1 else "Not Approved"
    proba = float(model.predict_proba(X)[0, 1])
    return {"prediction": label, "approval_probability": proba}
