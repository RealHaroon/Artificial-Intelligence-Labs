## Loan Approval Prediction (Decision Tree) + FastAPI UI

A university-level end-to-end ML project that predicts whether a loan application will be **Approved** or **Not Approved** using a Decision Tree Classifier, then deploys the trained pipeline as a FastAPI service with a simple web UI.

***

## Objective
- Build a complete machine learning pipeline on a Kaggle-style loan approval dataset (tabular).
- Perform EDA, preprocessing, train/validate/test split, minimal hyperparameter tuning.
- Evaluate with confusion matrix, accuracy, precision/recall/F1, and ROC-AUC.
- Save the trained model pipeline as a `.pkl`.
- Deploy using FastAPI with:
  - `POST /predict-loan` (JSON)
  - Optional UI page to test inputs.

***

## Tech Stack
- Python, pandas, numpy
- scikit-learn (DecisionTreeClassifier, preprocessing, metrics)
- joblib (model saving/loading)
- FastAPI + Uvicorn (deployment)
- HTML/CSS (optional UI)

***

## Project Structure
```text
Loan Approval Prediction/
├── app.py                              # FastAPI backend (UI + JSON API)
├── model/
│   └── loan_tree.pkl                   # Full pipeline: preprocessing + Decision Tree
├── template/
│   └── index.html                      # Web form UI
├── notebook/
│   └── Loan Approval Prediction.ipynb  # Training notebook
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── slides/
│   └── Loan Approval Prediction.pptx   # Presentation
├── ss.png                              # UI screenshot
├── testCase.txt                        # Sample applicant inputs
└── requirements.txt                    # (currently empty — see below)
```

***

## Dataset
Use a Kaggle Loan Prediction dataset (tabular). Common columns used:
- Categorical: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`
- Numerical: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- Target: `Loan_Status` (in your case it’s numeric `0.0/1.0`)

> Note: Your model expects the exact column names used during training. In your pipeline, it uses `Self_Employed` (not `Self_Employment`).

***

## Workflow (What we built)
### 1) EDA
You performed:
- Dataset shape inspection
- Missing values check
- Class distribution check
- Correlation analysis (numeric features)
- Feature relevance review (and later Decision Tree feature effects)

### 2) Preprocessing
Implemented using `ColumnTransformer` inside a single sklearn `Pipeline`:
- Numerical: median imputation
- Categorical: most-frequent imputation + OneHotEncoding

This ensures the same preprocessing runs during:
- training
- evaluation
- API inference (deployment)

### 3) Splitting
Used a train/validation/test split with stratification (to maintain class ratio).

### 4) Model Training + Minimal Tuning
DecisionTreeClassifier tuned on validation using:
- `criterion`
- `max_depth`
- `min_samples_split`

Best validation run (your output):
- Best params: `criterion='gini', max_depth=7, min_samples_split=20`
- Validation AUC: `0.7679`

### 5) Evaluation (Test Set)
Your test results:
- Accuracy: **0.8293**
- Confusion matrix:
  ```
  [[22 16]
   [ 5 80]]
  ```
- Approved class: precision **0.83**, recall **0.94**, F1 **0.88**
- Not Approved class: precision **0.81**, recall **0.58**, F1 **0.68**

### Why these scores are considered good
- **Accuracy ~83%** is strong for a baseline Decision Tree on a real-world noisy tabular dataset.
- **Approved recall 0.94** means the model rarely misses approvals (low false negatives for Approved).
- The confusion matrix shows the model is better at catching approvals than rejections, which is a realistic behavior and a good point to discuss in your report (business trade-off).

### 6) Saving Model
Saved the full pipeline:
- preprocessing + decision tree
as:
- `model/loan_tree.pkl`

This is important because deployment must reproduce training preprocessing exactly.

***

## Running the API
### 1) Install dependencies
The bundled `requirements.txt` is empty, so install directly:
```bash
pip install pandas numpy scikit-learn joblib fastapi uvicorn jinja2 python-multipart
```
Python **3.10+** is required (the app uses `str | None` type hints). Model and template paths are resolved relative to `app.py`, so no path edits are needed.

### 2) Start server
From the project folder:
```bash
uvicorn app:app --reload
```

Server runs at:
- http://127.0.0.1:8000
Docs:
- http://127.0.0.1:8000/docs

***

## API Endpoints
### `GET /`
Renders the web form.

### `POST /predict-ui`
Form submission from the UI; re-renders the page with the result (or an error message).

### `POST /predict-loan`
**Request:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 128,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```
Only `ApplicantIncome` and `CoapplicantIncome` are required; missing fields are imputed by the pipeline.

**Response** (illustrative):
```json
{
  "prediction": "Approved",
  "approval_probability": 0.84
}
```

***

## Common Issues
- **Column name mismatch**: If training used `Self_Employed` but API/UI sends `Self_Employment`, prediction will fail or behave incorrectly.
- **Different dataset schema**: If you trained on a different Kaggle loan dataset, update your API fields accordingly.

***

## Future Improvements (optional enhancements)
- Try `class_weight="balanced"` in DecisionTreeClassifier if “Not Approved” recall is too low.
- Add threshold tuning (don’t always use 0.5).
- Compare with Random Forest / XGBoost for better generalization.
- Add input validation rules (ranges for income/loan amount).

***

## Author
**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
