# 🏠 House Price Predictor — Linear Regression

An end-to-end regression lab that predicts house sale prices from five key features of the **Ames Housing** dataset using **Linear Regression**. It covers the full pipeline — EDA, preprocessing, training, evaluation and model saving — and serves predictions through **FastAPI** with an HTML form.

---

## 📁 Project Structure

```text
House Price Predictor/
├── app.py                              # FastAPI backend (form UI + /predict)
├── data/
│   └── raw/
│       └── AmesHousing.csv             # 2,930 houses × 82 columns
├── models/
│   ├── model.pkl                       # Trained LinearRegression
│   └── scaler.pkl                      # Fitted StandardScaler
├── notebook/
│   └── House Price Predictor.ipynb     # Training notebook
├── templates/
│   └── index.html                      # Input form + result
├── slides/
│   └── House Price Predictor.pptx      # Presentation
├── ss.png                              # UI screenshot
├── testCases.txt                       # Sample inputs with expected price ranges
└── readme.md
```

---

## 📊 Dataset

- **Ames Housing Dataset** — [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- 2,930 residential sales in Ames, Iowa, with 82 columns.
- Target: `SalePrice` (USD).

### Features Used

| Feature | Description |
|---------|-------------|
| `OverallQual` | Overall material and finish quality (1–10) |
| `GrLivArea` | Above-ground living area (sq ft) |
| `GarageCars` | Garage capacity (cars) |
| `TotalBsmtSF` | Total basement area (sq ft) |
| `YearBuilt` | Original construction year |

These were chosen for their strong correlation with `SalePrice` during EDA.

---

## 🔬 ML Pipeline

1. **Load** the dataset and inspect shape, types and summary statistics.
2. **EDA:** missing values, target distribution, correlation heatmap.
3. **Preprocessing:** select the 5 features, handle missing values, scale with `StandardScaler`.
4. **Split:** 70% train / 15% validation / 15% test.
5. **Train** `LinearRegression`.
6. **Evaluate** with R² and RMSE.
7. **Save** the model and scaler with `joblib` to `models/`.

---

## 📈 Results

| Split | R² | RMSE |
|-------|----|------|
| Validation | **0.830** | $33,049 |
| Test | **0.766** | $43,965 |

The model explains roughly 77–83% of the price variance using only five features. The higher test RMSE points to expensive outliers that a linear model struggles with.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn jinja2 python-multipart joblib numpy scikit-learn
```

### 2. Start the server

From inside this folder (model paths are relative):

```bash
uvicorn app:app --reload
```

### 3. Use the app

Open http://127.0.0.1:8000, fill in the five features, and click **Predict Price**.

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Renders the input form |
| `POST` | `/predict` | Accepts **form data** (`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`) and re-renders the page with the predicted price |

### Sample Inputs (from `testCases.txt`)

| OverallQual | GrLivArea | GarageCars | TotalBsmtSF | YearBuilt | Expected range |
|-------------|-----------|------------|-------------|-----------|----------------|
| 6 | 1600 | 2 | 900 | 2003 | $160k – $220k |
| 8 | 2400 | 3 | 1200 | 2015 | $280k – $380k |
| 4 | 1100 | 1 | 600 | 1975 | $90k – $120k |
| 9 | 3200 | 3 | 1600 | 2018 | $350k – $550k |

---

## 📝 Notes

- Linear Regression fits because the target is continuous; classification metrics (F1, confusion matrix) don't apply.
- To improve accuracy, try log-transforming `SalePrice`, adding more features, or using Random Forest / XGBoost (see the [Real Estate Price Estimator](../Real%20State%20Price%20Estimator/) lab).

---

## 👨‍💻 Author

**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
