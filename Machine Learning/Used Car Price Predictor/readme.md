# Used Car Price Predictor

A machine-learning web application that estimates the market price of a used car in Pakistan. The application uses a trained scikit-learn regression model, a FastAPI backend, and a responsive HTML frontend.

## Features

- Predicts used-car prices in Pakistani rupees (PKR)
- Displays estimates in lakh or crore format
- Uses the saved preprocessing pipeline from model training
- Provides category suggestions learned by the preprocessor
- Validates prediction inputs with Pydantic
- Includes interactive API documentation through FastAPI
- Offers a responsive browser interface

## Project Structure

```text
Used Car Price Predictor/
├── main.py                     # FastAPI application
├── readme.md                   # Project documentation
├── data/
│   ├── pakwheels_raw.csv
│   └── pakwheels_featured.csv
├── model/
│   ├── car_price_model.pkl     # Trained regression model
│   └── preprocessor.pkl        # Fitted preprocessing pipeline
├── notebook/
│   └── UsedCarPricePredictor.ipynb
├── plots/
└── static/
    └── index.html              # Frontend interface
```

## Requirements

The application requires Python and the following packages:

```text
fastapi
uvicorn
pandas
scikit-learn
joblib
pydantic
```

Install them with:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic
```

The saved model and preprocessor should be loaded with a scikit-learn version compatible with the version used during training.

## Running the Application

From the project directory, run:

```bash
uvicorn main:app --reload
```

Alternatively, run the application directly:

```bash
python main.py
```

Open the frontend in a browser:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Prediction Inputs

The model expects the following vehicle details:

| Field | Description | Example |
|---|---|---|
| `make` | Vehicle manufacturer | `Toyota` |
| `model` | Vehicle model | `Corolla` |
| `variant` | Model trim or variant | `GLi 1.3` |
| `year` | Manufacturing year | `2020` |
| `mileage_km` | Distance driven in kilometres | `50000` |
| `engine_cc` | Engine displacement in cc | `1300` |
| `fuel` | Fuel type | `Petrol` |
| `transmission` | Transmission type | `Automatic` |
| `city` | Listing city | `Lahore` |

Use `Unknown` when the exact variant is unavailable.

## API Endpoints

### Web Interface

```http
GET /
```

Returns the frontend application.

### Health Check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model": "loaded"
}
```

### Model Options

```http
GET /api/options
```

Returns the makes, models, variants, fuel types, transmissions, and cities learned by the fitted preprocessor.

### Predict a Price

```http
POST /api/predict
Content-Type: application/json
```

Example request:

```json
{
  "make": "Toyota",
  "model": "Corolla",
  "variant": "GLi 1.3",
  "year": 2020,
  "mileage_km": 50000,
  "engine_cc": 1300,
  "fuel": "Petrol",
  "transmission": "Automatic",
  "city": "Lahore"
}
```

Example response:

```json
{
  "predicted_price_pkr": 4500000,
  "predicted_price_lakh": 45.0,
  "formatted_price": "PKR 45.00 lakh"
}
```

You can also test the endpoint with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Corolla",
    "variant": "GLi 1.3",
    "year": 2020,
    "mileage_km": 50000,
    "engine_cc": 1300,
    "fuel": "Petrol",
    "transmission": "Automatic",
    "city": "Lahore"
  }'
```

## Model Information

The prediction workflow uses two saved artifacts:

- `preprocessor.pkl` imputes missing values, scales numeric features, and one-hot encodes categorical features.
- `car_price_model.pkl` predicts the price from the transformed features.

The current trained model is a tuned random forest regressor. Its notebook evaluation produced approximately:

- MAE: PKR 509,442
- RMSE: PKR 1,991,780
- R² score: 0.875

Model estimates are informational and may differ from an actual negotiated sale price because of vehicle condition, documentation, location, market movement, and other factors not represented in the training data.
