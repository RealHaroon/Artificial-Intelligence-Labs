
# House Price Predictor – Lab Project

## Project Overview

This project demonstrates building a **House Price Prediction system** using **Linear Regression**.
The system predicts house prices based on features like **Overall Quality, Ground Living Area, Garage Capacity, Basement Area, and Year Built**.
It includes the **full ML pipeline**: data preprocessing, model training, evaluation, saving the model, and deploying it via **FastAPI** with a simple HTML interface.

---

## Dataset

* **Ames Housing Dataset (Kaggle)**
* Contains tabular housing data including features and sale prices
* Download link: [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

---

## Features Used

* `OverallQual` – Overall material and finish quality
* `GrLivArea` – Ground living area (sq ft)
* `GarageCars` – Garage capacity (number of cars)
* `TotalBsmtSF` – Total basement area (sq ft)
* `YearBuilt` – Year house was built

---

## ML Pipeline

1. **Load dataset** and explore basic statistics
2. **EDA:** check distributions, correlations, missing values
3. **Preprocessing:** handle missing data, select features, scale numeric values
4. **Train/Test Split:** 70% train, 15% validation, 15% test
5. **Train Linear Regression** model
6. **Evaluate:** R² and RMSE metrics
7. **Save model** and scaler using `joblib`
8. **Deploy with FastAPI:** `/predict` endpoint + HTML frontend

---

## How to Run

1. Install dependencies:

```bash
pip install fastapi uvicorn jinja2 joblib numpy scikit-learn
```

2. Start FastAPI app:

```bash
uvicorn app:app --reload
```

3. Open browser at: `http://127.0.0.1:8000`

4. Fill the form with house features and click **Predict Price**

---

## Results

* **Validation R²:** ~0.83
* **Test R²:** ~0.77
* RMSE indicates reasonable prediction errors in dollars
* Model generalizes well to unseen data

---

## Notes

* Linear Regression is used because the target variable is **continuous**.
* For classification metrics like F1 or confusion matrix, this model is **not applicable**.
* To improve predictions further, advanced models like **Random Forest or XGBoost** can be explored.

---

