# 🏘️ Real Estate Price Estimator — XGBoost

An end-to-end Machine Learning lab that predicts California house prices from 8 geographic and demographic features. An **XGBoost Regressor** is served through a **FastAPI** backend with a responsive Bootstrap UI.

---

## 🚀 Tech Stack

- **Machine Learning:** `scikit-learn`, `xgboost`, `numpy`
- **Backend:** `FastAPI`, `uvicorn`, `pydantic`
- **Frontend:** HTML5, Bootstrap 5, Vanilla JavaScript (Fetch API)
- **Serialization:** `joblib`

---

## 📁 Project Structure

```text
Real State Price Estimator/
├── app.py                      # FastAPI backend (UI + /predict)
├── model/
│   └── house_model.joblib      # Trained XGBRegressor
├── notebook/
│   └── notebook.ipynb          # Training notebook
├── templates/
│   └── index.html              # Bootstrap UI
├── ss.png                      # UI screenshot
└── readme.md
```

---

## 📊 Dataset

**California Housing** (`sklearn.datasets.fetch_california_housing`) — 20,640 block groups from the 1990 U.S. census.

| Feature | Description |
|---------|-------------|
| `median_income` | Median income in the block (tens of thousands USD) |
| `house_age` | Median house age |
| `avg_rooms` | Average rooms per household |
| `avg_bedrooms` | Average bedrooms per household |
| `population` | Block population |
| `avg_occupancy` | Average household members |
| `latitude` | Block latitude |
| `longitude` | Block longitude |

**Target:** median house value, in units of **$100,000**.

---

## 🔬 ML Pipeline

1. Load the dataset with `fetch_california_housing()`.
2. Split 80% train / 20% test (`random_state=42`).
3. Train `XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)`.
4. Evaluate with R².
5. Save with `joblib`.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test R² | **0.8141** |

Gradient-boosted trees capture the non-linear effect of location (latitude/longitude) and income, which a linear model would miss.

---

## ✨ Features

- **Real-time inference** with millisecond responses.
- **Input validation** through a Pydantic schema.
- **Interactive UI** to test predictions without writing code.
- **Swagger docs** generated at `/docs`.

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn jinja2 joblib numpy scikit-learn xgboost
```

### 2. Start the server

From inside this folder:

```bash
uvicorn app:app --reload
```

- **Web UI:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs

---

## 📡 API Reference

### `POST /predict`

Accepts the 8 features as JSON and returns the estimated price in USD (the model output × 100,000).

**Request:**

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

**Response** (illustrative):

```json
{
  "price": "$452,600.00"
}
```

---

## 👨‍💻 Author

**M Haroon Abbas** — BSCS | OCI AI & Generative AI Certified Professional
[GitHub @RealHaroon](https://github.com/RealHaroon)
