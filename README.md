# 🧠 Artificial Intelligence Labs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-blue?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-green)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688?logo=fastapi)

A hands-on portfolio of **end-to-end Machine Learning and Deep Learning labs**, organized into two folders:

- 🧠 [**Deep Learning**](./Deep%20Learning/) — neural networks for computer vision (4 labs)
- 📊 [**Machine Learning**](./Machine%20Learning/) — classical ML for classification, regression and NLP (8 labs)

Every deployed lab goes beyond a standalone notebook: data is explored and preprocessed, a model is trained and evaluated, the trained artifact is saved, and the model is served through a **FastAPI** app with a browser UI.

---

## 📑 Table of Contents

1. [Projects at a Glance](#-projects-at-a-glance)
2. [Core Methodology](#️-core-methodology)
3. [Project Breakdown](#-project-breakdown)
4. [Tech Stack](#️-tech-stack)
5. [Repository Structure](#-repository-structure)
6. [Getting Started](#-getting-started)
7. [Running a Lab](#-running-a-lab)
8. [Known Caveats](#️-known-caveats)
9. [Author](#-author)

---

## 📊 Projects at a Glance

### 🧠 Deep Learning

| # | Lab | Task | Model | Dataset | Key Result |
|---|-----|------|-------|---------|------------|
| 1 | [Chest X-Ray Classifier](./Deep%20Learning/Chest%20X-Ray%20Classifier%20COVID-19%20Pneumonia%20&%20Normal%20Detection/) | 3-class image classification | DenseNet121 (transfer learning) | Kaggle Chest X-ray (COVID-19 & Pneumonia) | Acc **92.7%**, Macro-F1 **0.934**, AUC **0.994** |
| 2 | [Malaria Cell Classification](./Deep%20Learning/Malaria%20Cell%20Classification%20System/) | Binary image classification | Custom CNN | Kaggle Cell Images for Malaria | Acc **95.3%**, ROC-AUC **0.990** |
| 3 | [CIFAR-10 Object Recognition](./Deep%20Learning/CIFAR-10%20Object%20Recognition/) | 10-class image classification | Custom CNN + augmentation | CIFAR-10 | Test Acc **82.3%** |
| 4 | [MNIST Digit Classifier](./Deep%20Learning/MNIST%20Digit%20Classifier/) | 10-class digit recognition | Dense neural network | MNIST | Test Acc **97.8%** |

### 📊 Machine Learning

| # | Lab | Task | Model | Dataset | Key Result |
|---|-----|------|-------|---------|------------|
| 1 | [Diabetes Diagnostic Tool](./Machine%20Learning/Diabetes%20Diagnostic%20Tool/) | Binary classification | Logistic Regression + threshold tuning | Pima Indians Diabetes | Recall **0.85** @ tuned threshold |
| 2 | [Loan Approval Prediction](./Machine%20Learning/Loan%20Approval%20Prediction/) | Binary classification | Decision Tree | Kaggle Loan Prediction | Acc **82.9%** |
| 3 | [Iris Species Classifier](./Machine%20Learning/Iris%20Species%20Classifier/) | 3-class classification | KNN (k=3) | Iris (scikit-learn) | Test Acc **95.7%** |
| 4 | [SMS Spam Filter](./Machine%20Learning/SMS%20Spam%20Filter/) | Text classification (NLP) | TF-IDF + Multinomial Naive Bayes | SMS Spam Collection | Acc **96.7%** |
| 5 | [House Price Predictor](./Machine%20Learning/House%20Price%20Predictor/) | Regression | Linear Regression | Ames Housing | Test R² **0.77** |
| 6 | [Real Estate Price Estimator](./Machine%20Learning/Real%20State%20Price%20Estimator/) | Regression | XGBoost Regressor | California Housing | R² **0.81** |
| 7 | [Salary Predictor](./Machine%20Learning/Salary%20Predictor/) | Simple regression | Linear Regression | Salary vs. Experience | R² **0.90** |
| 8 | [DNA Sequence Classifier](./Machine%20Learning/DNA%20Sequence%20Classifier/) | Multi-class classification | k-mer features + Logistic Regression / XGBoost | Synthetic DNA sequences | 🚧 In progress |

Each lab folder has its own README with the full pipeline, results and API reference.

---

## 🏗️ Core Methodology

Each lab follows the same lifecycle:

1. **Data Exploration (EDA):** distributions, class balance, correlations, missing values.
2. **Preprocessing:** imputation, scaling/normalization, encoding, image resizing and augmentation, text vectorization.
3. **Model Development:** a baseline model, then tuning (grid search, validation-based selection, fine-tuning for transfer learning).
4. **Evaluation:** metrics chosen for the task — accuracy, precision, recall, F1, ROC-AUC and confusion matrices for classification; R² and RMSE for regression.
5. **Serialization:** models saved as `.pkl` / `.joblib` (scikit-learn, XGBoost) or `.keras` / `.h5` (TensorFlow).
6. **Deployment:** a FastAPI backend that loads the saved model and serves predictions through a JSON endpoint and an HTML UI.

---

## 🔬 Project Breakdown

### 🧠 Deep Learning & Computer Vision

- 🫁 **Chest X-Ray Classifier** — Classifies chest X-rays as COVID19, PNEUMONIA or NORMAL using DenseNet121 with two-phase fine-tuning, class weighting for imbalance, and Grad-CAM explainability.
- 🦠 **Malaria Cell Classification System** — A lightweight 3-block CNN that flags parasitized vs. uninfected cells in blood smear images (64×64 input).
- 🖼️ **CIFAR-10 Object Recognition** — A CNN built from scratch with BatchNorm, Dropout and on-the-fly augmentation, served through a drag-and-drop UI.
- 🔢 **MNIST Digit Classifier** — Draw a digit on a canvas in the browser; the backend crops, centers and resizes the drawing MNIST-style before predicting.

### 📊 Machine Learning (Classification & Regression)

- 🏥 **Diabetes Diagnostic Tool** — Logistic Regression pipeline that treats zeros as missing values and tunes the decision threshold to favor recall.
- 💰 **Loan Approval Prediction** — A Decision Tree inside a full scikit-learn `Pipeline` (imputation + one-hot encoding) so training and serving use identical preprocessing.
- 🌸 **Iris Species Classifier** — StandardScaler + KNN pipeline with validation-based selection of `k`.
- 📱 **SMS Spam Filter** — TF-IDF features with Multinomial Naive Bayes, returning a label and confidence score.
- 🏠 **House Price Predictor** — Linear Regression on five key Ames Housing features.
- 🏘️ **Real Estate Price Estimator** — XGBoost regressor on the 8-feature California Housing dataset.
- 💼 **Salary Predictor** — Single-feature Linear Regression mapping years of experience to salary.
- 🧬 **DNA Sequence Classifier** *(in progress)* — Classifies DNA sequences by organism class using k-mer counts plus GC/AT content, comparing a majority baseline, Logistic Regression and XGBoost with stratified cross-validation.

---

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Data & EDA | NumPy, Pandas, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Serving | FastAPI, Uvicorn, Pydantic, Jinja2 |
| Frontend | HTML, CSS, Vanilla JavaScript, Bootstrap 5 |
| Image processing | Pillow |
| Serialization | joblib, pickle, Keras `.keras` / `.h5` |
| Training environment | Jupyter Notebook, Kaggle Notebooks (GPU) |

---

## 📂 Repository Structure

Most labs share the same layout:

```text
<Lab Name>/
├── app.py              # FastAPI app (UI + prediction endpoint)
├── data/               # Raw dataset (when not downloaded in the notebook)
├── model/              # Saved model artifacts (House Price uses models/)
├── notebook/           # Training notebook: EDA → training → evaluation → saving
├── templates/          # index.html UI (Iris and Loan use template/)
├── slides/             # Presentation deck (some labs)
├── ss.png              # Screenshot of the UI (some labs)
├── testCase.txt        # Sample inputs for manual testing (some labs)
└── readme.md           # Lab documentation
```

Top level:

```text
Artificial-Intelligence-Labs/
├── Deep Learning/
│   ├── Chest X-Ray Classifier COVID-19 Pneumonia & Normal Detection/
│   ├── CIFAR-10 Object Recognition/
│   ├── Malaria Cell Classification System/
│   └── MNIST Digit Classifier/
│
├── Machine Learning/
│   ├── Diabetes Diagnostic Tool/
│   ├── DNA Sequence Classifier/
│   ├── House Price Predictor/
│   ├── Iris Species Classifier/
│   ├── Loan Approval Prediction/
│   ├── Real State Price Estimator/
│   ├── Salary Predictor/
│   └── SMS Spam Filter/
│
├── Genetic Algorithms.py   # Placeholder for upcoming Genetic Algorithms work (currently empty)
├── main.py                 # Placeholder (currently empty)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher (the Loan Approval app uses `str | None` type hints)
- Git

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/RealHaroon/Artificial-Intelligence-Labs.git
    cd Artificial-Intelligence-Labs
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv venv

    # Windows
    venv\Scripts\activate
    # Linux / macOS / WSL
    source venv/bin/activate
    ```

3. **Install dependencies.** There is no single root `requirements.txt`; this command covers every lab:

    ```bash
    pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn \
                tensorflow pillow fastapi uvicorn jinja2 python-multipart jupyter
    ```

    The Diabetes lab also ships its own pinned `requirements.txt`.

4. **Explore the notebooks:**

    ```bash
    jupyter notebook
    ```

---

## 🌐 Running a Lab

Each deployed lab is a standalone FastAPI app. Run it from inside the lab folder:

```bash
cd "Machine Learning/Iris Species Classifier"
uvicorn app:app --reload
```

Then open:

- **Web UI:** http://127.0.0.1:8000
- **Interactive API docs (Swagger):** http://127.0.0.1:8000/docs

| Lab | Prediction endpoint | Input |
|-----|---------------------|-------|
| Chest X-Ray | `POST /predict` | Image upload (`file`) |
| Malaria | `POST /predict` | Image upload (`file`) |
| CIFAR-10 | `POST /predict` | Image upload (`file`) |
| MNIST | `POST /predict` | JSON `{ "image": "<base64 PNG>" }` |
| Diabetes | `POST /predict-diabetes` | JSON (8 clinical features) |
| Loan Approval | `POST /predict-loan` | JSON (11 applicant fields) |
| Iris | `POST /predict` | JSON (4 flower measurements) |
| SMS Spam | `POST /predict` | JSON `{ "message": "..." }` |
| House Price | `POST /predict` | HTML form (5 features) |
| Real Estate | `POST /predict` | JSON (8 features) |
| Salary | `POST /predict` | JSON `{ "years_experience": 5.5 }` |

---

## ⚠️ Known Caveats

- **Paths:** Every app resolves its model and template paths relative to its own `app.py`, so labs run on any machine. Training notebooks may still save to the original author's absolute paths; adjust the save path before retraining.
- **Medical labs are educational only.** The Chest X-Ray, Malaria and Diabetes models are not clinically validated and must not be used for diagnosis.
- **Large files:** Model files (`.keras`, `.h5`, `.pkl`) and some datasets are committed directly, so the first clone may take a while.

---

## 👨‍💻 Author

**Muhammad Haroon Abbas Khan**
*BS Computer Science, University of Sindh, Laar Campus Badin*

Data Scientist and AI/ML developer focused on bridging the gap between raw data and deployed applications: building end-to-end data pipelines, designing deep learning architectures, and serving models through modern API frameworks backed by relational databases such as Oracle and PostgreSQL.

**Certifications:**

- 🏆 Oracle Cloud Infrastructure (OCI) AI Foundations Certified
- 🏆 Oracle Cloud Infrastructure (OCI) Generative AI Professional Certified

**Connect:** [GitHub @RealHaroon](https://github.com/RealHaroon)
