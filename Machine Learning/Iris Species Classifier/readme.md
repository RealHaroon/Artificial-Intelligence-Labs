# 🌸 Iris Species Classifier (KNN)

A beginner-friendly end-to-end ML lab that trains a **K-Nearest Neighbors** classifier on scikit-learn's built-in Iris dataset (4 numeric features, 3 classes), tunes `k` on a validation set, evaluates it on a held-out test set, saves the trained pipeline, and serves predictions through **FastAPI** with a simple HTML/CSS/JS UI.

---

## 📁 Project Structure

```text
Iris Species Classifier/
├── app.py                              # FastAPI backend (/, /health, /predict)
├── model/
│   └── iris_knn_pipeline.joblib        # Saved pipeline: StandardScaler + KNN
├── notebook/
│   └── Iris Species Classifier.ipynb   # Training notebook
├── template/
│   └── index.html                      # Single-file HTML + CSS + JS UI
├── slides/
│   └── Iris  Classifier.pptx           # Presentation
├── ss.png                              # UI screenshot
├── testCase.txt                        # Sample inputs per species
└── readme.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | `sklearn.datasets.load_iris()` |
| Samples | 150 (50 per class) |
| Features | sepal length, sepal width, petal length, petal width (cm) |
| Classes | `setosa`, `versicolor`, `virginica` |

---

## 🔬 ML Pipeline

1. **Load** the Iris dataset.
2. **Split 70 / 15 / 15** with stratification → Train 104, Validation 23, Test 23.
3. **Tune `k`** over `[3, 5, 7, 9, 11]` using a `Pipeline(StandardScaler → KNeighborsClassifier)`.
   - Every `k` scored 100% on validation, so the smallest (`k = 3`) was chosen.
4. **Retrain** the final pipeline on Train + Validation combined.
5. **Evaluate** on the untouched test set.
6. **Save** the pipeline with `joblib`.

Scaling matters for KNN because it is distance-based: without it, larger-range features dominate the distance.

---

## 📈 Results (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | **95.65%** (22 / 23) |
| Macro F1 | 0.958 |

Confusion matrix (rows = true, cols = predicted):

```text
             setosa  versicolor  virginica
setosa          7         0          0
versicolor      0         8          0
virginica       0         1          7
```

Setosa is perfectly separable; the single error is a virginica predicted as versicolor, the two classes that overlap.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install numpy scikit-learn joblib fastapi uvicorn
```

### 2. (Optional) Retrain

Run all cells in `notebook/Iris Species Classifier.ipynb`. The notebook saves to an absolute Windows path in `MODEL_PATH`; point it at `model/iris_knn_pipeline.joblib` if you're on another machine.

### 3. Start the server

From inside this folder:

```bash
uvicorn app:app --reload
```

- **Web UI:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs

---

## 📡 API Reference

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /predict`

**Request:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "predicted_class_name": "setosa",
  "class_probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

Probabilities come from `predict_proba()`, which for KNN is the share of the `k = 3` neighbors in each class (so values are 0, 0.33, 0.67 or 1.0). Request bodies are validated by a Pydantic model. More sample inputs are in `testCase.txt`.

---

## 👨‍💻 Author

**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
