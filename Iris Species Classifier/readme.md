# Iris Flower Classifier (KNN) — README

A beginner-friendly end-to-end ML lab that trains a KNN model on scikit-learn’s built-in Iris dataset (4 numeric features, 3 classes), evaluates it, saves the trained pipeline, and serves predictions via FastAPI (with a simple HTML+CSS UI). 

## 1) Project files
- `Iris Spcies Classifier.ipynb` (your notebook): loads data, splits (train/val/test), tunes `n_neighbors`, evaluates, and saves the trained pipeline with Joblib. [scikit-learn](https://scikit-learn.org/stable/model_persistence.html)
- `iris_knn_pipeline.joblib`: saved pipeline (StandardScaler + KNN). [scikit-learn](https://scikit-learn.org/stable/model_persistence.html)
- `app.py`: FastAPI app that loads the saved pipeline and exposes `/health` and `/predict`.
- `index.html`: single-file HTML + CSS + JS UI that calls `/predict`.

## 2) Setup (local)
Create/activate an environment, then install dependencies:
```bash
pip install numpy scikit-learn joblib fastapi uvicorn
```

## 3) Train, evaluate, and save
1. Open `iris_knn_train.ipynb` in Jupyter Notebook.
2. Run all cells to:
   - Load Iris via `load_iris()`. 
   - Train a pipeline and select the best `n_neighbors` using validation accuracy.
   - Evaluate on the test set (accuracy, confusion matrix, classification report).
   - Save the final pipeline to `iris_knn_pipeline.joblib` using Joblib. [scikit-learn](https://scikit-learn.org/stable/model_persistence.html)

## 4) Run the API + HTML UI
Make sure `app.py`, `index.html`, and `iris_knn_pipeline.joblib` are in the same folder, then start the server:
```bash
uvicorn app:app --reload
```

Open in browser:
- UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 5) API endpoints
- `GET /health` → `{"status":"ok"}`
- `POST /predict` → returns:
  - `predicted_class_name`
  - `class_probabilities` (all 3 classes) produced using `predict_proba()` from KNN. [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

Example request JSON (same as the UI sends):
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

FastAPI uses a Pydantic model to validate the JSON request body for `/predict`.