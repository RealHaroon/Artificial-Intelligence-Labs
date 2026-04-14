
# 🧠 Artificial Intelligence Labs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-blue?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-009688?logo=fastapi)

Welcome to the **Artificial Intelligence Labs** repository! This repository serves as a practical, hands-on portfolio showcasing end-to-end Machine Learning (ML) and Deep Learning (DL) pipelines. 

The primary focus here is to move beyond isolated Jupyter Notebooks. These labs demonstrate the complete AI lifecycle: from raw data ingestion and Exploratory Data Analysis (EDA), through model training and hyperparameter tuning, all the way to model serialization and REST API deployment.

---

## 📑 Table of Contents
1. [Core Methodology](#-core-methodology)
2. [Project Breakdown](#-project-breakdown)
3. [Tech Stack & Tools](#-tech-stack--tools)
4. [Repository Structure](#-repository-structure)
5. [Getting Started](#-getting-started)
6. [API Deployment](#-api-deployment)
7. [Author](#-author)

---

## 🏗️ Core Methodology
Each project in this repository generally adheres to the following workflow:
1. **Data Preprocessing:** Handling missing values, outlier detection, and data normalization/standardization.
2. **Feature Engineering:** Encoding categorical variables and selecting optimal features.
3. **Model Development:** Training baseline models and scaling up to complex algorithms (Ensemble methods, CNNs).
4. **Evaluation:** Utilizing accuracy, precision, recall, F1-score, and confusion matrices to validate model performance.
5. **Deployment:** Wrapping the optimized, serialized model (Pickle/Joblib or H5) into a FastAPI backend to serve real-time predictions via HTTP endpoints.

---

## 🔬 Project Breakdown

### 🧠 Deep Learning & Computer Vision
* 🫁 **Chest X-Ray Classifier (COVID-19 & Pneumonia):** A Convolutional Neural Network (CNN) trained on medical imaging to detect and classify lung infections.
* 🦠 **Malaria Cell Classification System:** An automated diagnostic tool using deep learning to identify parasitized cells in blood smear images.
* 🖼️ **CIFAR-10 Object Recognition:** A multi-class image classification model predicting objects across 10 distinct categories.
* 🔢 **MNIST Digit Classifier:** A foundational computer vision project recognizing handwritten digits with high accuracy.

### 📊 Machine Learning (Classification & Regression)
* 🏥 **Diabetes Diagnostic Tool:** A predictive classification pipeline to determine the likelihood of diabetes based on medical predictor variables.
* 🏠 **House Price Predictor & Real Estate Estimator:** End-to-end regression pipelines utilizing feature scaling to predict property values.
* 💰 **Loan Approval Prediction:** A financial risk assessment model classifying whether a loan should be approved based on applicant history.
* 🌸 **Iris Species Classifier:** A classic machine learning implementation establishing a solid baseline for multi-class prediction.
* 📱 **SMS Spam Filter:** Natural Language Processing (NLP) techniques applied to classify text messages as spam or legitimate.
* 💼 **Salary Predictor:** A regression model analyzing income trends based on experience and feature data.

---

## 🛠️ Tech Stack & Tools

* **Programming:** Python
* **Data Manipulation & EDA:** NumPy, Pandas, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Deep Learning:** TensorFlow, Keras
* **Web Framework / API:** FastAPI, Uvicorn
* **Environment & Version Control:** Jupyter Notebook, Git
* **Infrastructure / DevOps:** Docker, Windows Subsystem for Linux (WSL)

---

## 📂 Repository Structure

```text
Artificial-Intelligence-Labs/
│
├── Chest X-Ray Classifier COVID-19 Pneumonia & Normal Detection/
├── CIFAR-10 Object Recognition/
├── Diabetes Diagnostic Tool/
├── House Price Predictor/
├── Iris Species Classifier/
├── Loan Approval Prediction/
├── Malaria Cell Classification System/
├── MNIST Digit Classifier/
├── Real State Price Estimator/
├── Salary Predictor/
├── SMS Spam Filter/
│
├── main.py                   # FastAPI application entry point (if centralized)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
````

-----

## 🚀 Getting Started

### Prerequisites

  * Python 3.8 or higher installed.
  * Git installed on your local machine.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/RealHaroon/Artificial-Intelligence-Labs.git](https://github.com/RealHaroon/Artificial-Intelligence-Labs.git)
    cd Artificial-Intelligence-Labs
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv

    # On Windows:
    venv\Scripts\activate
    # On Linux/WSL/macOS:
    source venv/bin/activate 
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the Notebooks:**
    Launch Jupyter to view the training processes:

    ```bash
    jupyter notebook
    ```

-----

## 🌐 API Deployment

Many of these models are designed to be production-ready. To spin up the FastAPI server and test the model endpoints locally:

1.  Ensure your virtual environment is active.
2.  Run the Uvicorn server:
    ```bash
    uvicorn main:app --reload
    ```
3.  Open your browser and navigate to `http://127.0.0.1:8000`.
4.  **Interactive Docs:** FastAPI automatically generates interactive API documentation. Visit `http://127.0.0.1:8000/docs` to test the endpoints directly from your browser.

-----

## 👨‍💻 Author

**Muhammad Haroon Abbas Khan** *BS Computer Science, University of Sindh, Laar Campus Badin*

I am a Data Scientist and AI/ML developer focused on bridging the gap between raw data and deployed applications. I specialize in building end-to-end data pipelines, designing deep learning architectures, and serving intelligent web applications using robust relational databases (like Oracle and PostgreSQL) and modern API frameworks.

**Certifications:**

  * 🏆 Oracle Cloud Infrastructure (OCI) AI Foundations Certified
  * 🏆 Oracle Cloud Infrastructure (OCI) Generative AI Professional Certified

**Connect with me:**

  * **GitHub:** [@RealHaroon](https://www.google.com/search?q=https://github.com/RealHaroon)

<!-- end list -->

```
```
