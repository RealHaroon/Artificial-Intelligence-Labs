from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
STATIC_DIR = BASE_DIR / "static"


class CarDetails(BaseModel):
    make: Annotated[str, Field(min_length=1, max_length=80)]
    model: Annotated[str, Field(min_length=1, max_length=100)]
    variant: Annotated[str, Field(min_length=1, max_length=150)] = "Unknown"
    year: Annotated[int, Field(ge=1980, le=2027)]
    mileage_km: Annotated[int, Field(ge=0, le=1_500_000)]
    engine_cc: Annotated[float, Field(ge=100, le=10_000)]
    fuel: Annotated[str, Field(min_length=1, max_length=40)]
    transmission: Annotated[str, Field(min_length=1, max_length=40)]
    city: Annotated[str, Field(min_length=1, max_length=100)]

    @field_validator("make", "model", "variant", "fuel", "transmission", "city")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Prediction(BaseModel):
    predicted_price_pkr: int
    predicted_price_lakh: float
    formatted_price: str


def format_pkr(price: int) -> str:
    """Format a price using the Pakistani lakh/crore number system."""
    if price >= 10_000_000:
        return f"PKR {price / 10_000_000:,.2f} crore"
    if price >= 100_000:
        return f"PKR {price / 100_000:,.2f} lakh"
    return f"PKR {price:,}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        app.state.model = joblib.load(MODEL_DIR / "car_price_model.pkl")
    except Exception as exc:
        raise RuntimeError(f"Could not load model files from {MODEL_DIR}: {exc}") from exc
    yield
    app.state.preprocessor = None
    app.state.model = None


app = FastAPI(
    title="Used Car Price Predictor",
    description="Predict used-car prices in PKR with the trained regression model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.get("/api/options")
async def options() -> dict[str, list[str]]:
    """Return categories learned by the fitted one-hot encoder."""
    try:
        categorical_pipeline = app.state.preprocessor.named_transformers_["cat"]
        encoder = categorical_pipeline.named_steps["encoder"]
        columns = ["make", "model", "variant", "fuel", "transmission", "city"]
        return {
            column: sorted(str(value) for value in values)
            for column, values in zip(columns, encoder.categories_)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model options are unavailable.") from exc


@app.post("/api/predict", response_model=Prediction)
async def predict(details: CarDetails) -> Prediction:
    row = pd.DataFrame(
        [details.model_dump()],
        columns=[
            "make",
            "model",
            "variant",
            "year",
            "mileage_km",
            "engine_cc",
            "fuel",
            "transmission",
            "city",
        ],
    )

    try:
        processed = app.state.preprocessor.transform(row)
        raw_prediction = float(app.state.model.predict(processed)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The prediction could not be completed.") from exc

    price = max(0, round(raw_prediction))
    return Prediction(
        predicted_price_pkr=price,
        predicted_price_lakh=round(price / 100_000, 2),
        formatted_price=format_pkr(price),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
