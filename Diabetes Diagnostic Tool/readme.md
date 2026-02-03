# Diabetes Prediction System

A complete machine learning lab demonstrating end-to-end diabetes prediction using Logistic Regression on the Pima Indians Diabetes dataset. Includes data preprocessing, threshold tuning, model evaluation, and FastAPI deployment with a minimal web UI.

## 📁 Project Structure

```
Diabetes Diagnostic Tool/
├── data/
│   └── raw/
│       └── diabetes.csv          # Pima Indians Diabetes dataset
├── notebook/
│   └── Diabetes Classification.ipynb  # Complete ML pipeline notebook
├── model/
│   └── diabetes_classification_model.pkl  # Trained model artifact
├── templates/
│   └── index.html                # Web UI for predictions
├── app.py                        # FastAPI deployment
├── testCase.txt                  # Test cases for API validation
└── requirements.txt              # Python dependencies
```

## 🎯 Lab Objectives

- Build end-to-end ML pipeline from raw data to deployment
- Master preprocessing techniques for medical datasets (handling zeros as missing values)
- Understand threshold tuning for imbalanced classification
- Evaluate models using comprehensive metrics (Confusion Matrix, Precision, Recall, F1, ROC-AUC)
- Deploy model as production-ready FastAPI service

## 🔬 Dataset

**Pima Indians Diabetes Dataset** (768 patients, 8 features)
- **Features**: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Target**: Outcome (0 = No Diabetes, 1 = Diabetes)
- **Class Distribution**: 65% non-diabetic, 35% diabetic
- **Known Issue**: Zero values represent missing data in clinical features

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open and execute `notebook/Diabetes Classification.ipynb` to:
- Perform EDA and visualize feature distributions
- Handle missing values (zeros → NaN → median imputation)
- Split data (70% train, 15% validation, 15% test)
- Train Logistic Regression with hyperparameter tuning (C parameter)
- Tune classification threshold on validation set
- Evaluate on test set using multiple metrics
- Save model artifact as `.pkl`

### 3. Deploy the API
```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000` for the web UI or `http://127.0.0.1:8000/docs` for Swagger documentation.

### 4. Test the API
Use the web interface at `http://127.0.0.1:8000` or test with example inputs from `testCase.txt`:

**Example Request** (paste into UI or send via API):
```json
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 28.4,
  "DiabetesPedigreeFunction": 0.35,
  "Age": 29
}
```

**Response**:
```json
{
  "diabetes_probability": 0.63,
  "threshold_used": 0.42,
  "predicted_class": 1
}
```

## 📊 Key Concepts

### Why Logistic Regression?
- Fast training and inference
- Probabilistic outputs enable threshold tuning
- Interpretable coefficients for medical stakeholders
- Strong baseline before complex models

### Why Threshold Tuning?
In medical diagnosis, **missing a diabetic patient (false negative) is more costly** than a false alarm (false positive). Lowering the threshold below 0.5 increases **Recall** (fewer missed cases) at the cost of **Precision** (more false alarms).

### Evaluation Metrics
- **Confusion Matrix**: TP, TN, FP, FN breakdown
- **Accuracy**: Overall correctness (misleading for imbalanced data)
- **Precision**: Of predicted positives, how many were correct?
- **Recall**: Of actual positives, how many did we catch? (Critical for medical screening)
- **F1-Score**: Harmonic mean of Precision and Recall
- **ROC-AUC**: Threshold-independent discrimination ability

## 🛠️ Technical Stack

- **Data Processing**: pandas, NumPy
- **ML Framework**: scikit-learn (Pipeline, LogisticRegression, GridSearchCV)
- **Visualization**: matplotlib, seaborn
- **Deployment**: FastAPI, Pydantic, Uvicorn
- **Serialization**: joblib

## 📝 Lab Deliverables

1. **Notebook** with:
   - EDA insights (zero-as-missing evidence, correlations, class balance)
   - Leakage-safe preprocessing pipeline
   - Hyperparameter tuning results (best C value)
   - Threshold selection rationale
   - Full metric suite on test set

2. **Model Artifact** (`diabetes_classification_model.pkl`) containing:
   - Trained pipeline (imputer + scaler + LogisticRegression)
   - Tuned threshold value
   - Feature names for prediction ordering

3. **FastAPI Service** with:
   - `/predict-diabetes` endpoint
   - Request validation (Pydantic schema)
   - Probability and class predictions
   - Interactive web UI

## 🧪 Test Cases

See `testCase.txt` for complete test scenarios including:
- Low-risk profile (expected class 0)
- High-risk profile (expected class 1)
- Edge cases (zeros, negative values, missing fields)
- Validation error cases (422 responses)

## 📚 Learning Outcomes

Students will understand:
- Medical dataset preprocessing (handling implausible values)
- Train/validation/test splitting for unbiased evaluation
- Regularization and hyperparameter tuning
- Precision-Recall tradeoff and threshold optimization
- Deploying ML models as REST APIs
- Production-ready model serialization

## 🔗 Resources

- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

***

**Author**: ML Lab -  University Level  
**License**: Educational Use