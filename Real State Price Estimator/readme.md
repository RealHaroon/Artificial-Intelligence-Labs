
# 🏠 Real Estate Price Estimator - ML Lab


An end-to-end Machine Learning project that predicts California house prices based on 8 geographic and demographic features. The project serves an **XGBoost Regressor** model through a **FastAPI** backend and includes a clean, responsive frontend UI.

## 🚀 Tech Stack
* **Machine Learning:** `scikit-learn`, `xgboost`, `pandas`, `numpy`
* **Backend:** `FastAPI`, `uvicorn`, `pydantic`
* **Frontend:** `HTML5`, `Bootstrap 5`, `Vanilla JavaScript` (Fetch API)
* **Serialization:** `joblib`

## ✨ Features
* **High Accuracy:** Utilizes an optimized XGBoost model ($R^2$ Score > 0.81).
* **Real-time Inference:** Fast API endpoints capable of serving predictions in milliseconds.
* **Data Validation:** Strict input validation using Pydantic schemas to ensure model stability.
* **Interactive UI:** A single-page web interface to test predictions without touching code.
* **Swagger Documentation:** Auto-generated API docs available at `/docs`.

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RealHaroon/real-estate-price-estimator.git](https://github.com/RealHaroon/real-estate-price-estimator.git)
   cd real-estate-price-estimator

```

2. **Install the required dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the FastAPI server:**
```bash
uvicorn app:app --reload

```


4. **Access the application:**
* **Web UI:** Open `http://127.0.0.1:8000` in your browser.
* **API Docs:** Open `http://127.0.0.1:8000/docs` to test the endpoints directly.



## 📡 API Endpoint Details

### `POST /predict`

Accepts a JSON payload with house features and returns the estimated price in USD.

**Request Body:**

```json
{
  "median_income": 8.32,
  "house_age": 41.0,
  "avg_rooms": 6.98,
  "avg_bedrooms": 1.02,
  "population": 322.0,
  "avg_occupancy": 2.55,
  "latitude": 37.88,
  "longitude": -122.23
}

```

**Response:**

```json
{
  "price": "$452,600.00"
}

```

## 👨‍💻 Author

**M Haroon Abbas** BSCS | OCI AI & Generative AI Certified Professional

GitHub: [@RealHaroon](https://github.com/RealHaroon)


---

