# 💼 Salary Predictor — Simple Linear Regression

A beginner-level machine learning lab that predicts a person's **salary from their years of experience** using **Simple Linear Regression**. The trained model is served through **FastAPI** with both an HTML form and a JSON endpoint.

---

## 🎯 Objectives

- Load and explore a small tabular dataset.
- Train and evaluate a single-feature Linear Regression model.
- Save the model with `joblib`.
- Serve predictions through a web form and a REST API.

---

## 📁 Project Structure

```text
Salary Predictor/
├── app.py                          # FastAPI backend (form + JSON endpoints)
├── data/
│   └── Salary_dataset.csv          # 30 rows: YearsExperience, Salary
├── data.zip                        # Zipped copy of the dataset
├── model/
│   └── salary_model.pkl            # Trained LinearRegression model
├── notebook/
│   └── Salary Predictor.ipynb      # EDA → training → evaluation → saving
├── templates/
│   └── index.html                  # Web UI
└── readme.md
```

---

## 📊 Dataset

| Column | Type | Description |
|--------|------|-------------|
| `YearsExperience` | float | Years of professional experience |
| `Salary` | float | Annual salary (USD) |

- **Rows:** 30, no missing values
- An extra `Unnamed: 0` index column is ignored.

---

## 🔬 ML Pipeline

1. **Load** `data/Salary_dataset.csv` with pandas.
2. **EDA:** `info()`, `describe()`, missing-value check, and histograms of salary and experience.
3. **Features / target:** `X = YearsExperience`, `y = Salary`.
4. **Split:** 80% train / 20% test (`random_state=42`).
5. **Train:** `LinearRegression()`.
6. **Evaluate:** R² on the test set.
7. **Save:** `joblib.dump(model, "model/salary_model.pkl")`.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test R² | **0.9024** |
| Example | 5.5 years → **$76,211.19** |

Salary grows almost linearly with experience, so a single-feature linear model explains about 90% of the variance.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn jinja2 python-multipart scikit-learn joblib numpy pandas
```

### 2. Start the server

From inside this folder:

```bash
uvicorn app:app --reload
```

### 3. Open the app

- **Web UI:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs

---

## 📡 API Reference

### `GET /`

Renders the HTML form.

### `POST /`

Form submission from the UI (`years_experience` form field). Re-renders the page with the prediction.

### `POST /predict`

**Request:**

```json
{
  "years_experience": 5.5
}
```

**Response:**

```json
{
  "years_experience": 5.5,
  "predicted_salary": 76211.19
}
```

---

## 🔮 Possible Improvements

- Report MAE / RMSE alongside R².
- Use cross-validation, since 30 rows gives a very small test set (6 samples).
- Add input validation (e.g., `years_experience >= 0`).

---

## 👨‍💻 Author

**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
