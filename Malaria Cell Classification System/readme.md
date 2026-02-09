
# Malaria Cell Classification with CNN (Binary Image Classification)

A simple Convolutional Neural Network (CNN) that classifies thin blood smear cell images into **Parasitized** (infected) vs **Uninfected** using TensorFlow/Keras. The goal is to demonstrate CNN basics (convolutions, feature maps, pooling) and a complete ML workflow from dataset loading to deployment with a FastAPI endpoint.

## Objective
- Train a lightweight CNN (no transfer learning) to detect malaria-infected cells from images.
- Keep the pipeline beginner-friendly: small input size (64×64), limited epochs, and clear evaluation using medical-relevant metrics.

## Dataset
- Kaggle: **Cell Images for Detecting Malaria** (two folders/classes: `Parasitized/` and `Uninfected/`).  
  Link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria [web:31]

## Model (What & Why)
- **CNN** is better than a basic ANN/MLP for images because it preserves spatial structure and learns local patterns (edges/textures) using convolution filters.
- Architecture used (simple + fast):
  - 3 × (Conv2D + MaxPooling2D)
  - Flatten → Dense(128) → Dropout
  - Output: `sigmoid` (binary probability)

## Training Pipeline (Creation Process)
1. Load images directly from folders using Keras directory loader.
2. Resize to **64×64**, normalize to **[0,1]** by dividing by 255.
3. Split:
   - Train (80%)
   - Validation/Test made from remaining 20% (then split roughly 50/50 by batches)
4. Train with Adam + Binary Cross-Entropy and EarlyStopping on validation AUC.
5. Save artifacts for inference:
   - `malaria_cnn.keras` (native Keras format)
   - `malaria_savedmodel/` (exported SavedModel directory)
   - `class_names.json` (label mapping)

## Evaluation (Why it’s good)
Test results (threshold = 0.5):
- **Accuracy:** 0.9532
- **ROC-AUC:** 0.9897
- Confusion Matrix: TN=1333, FP=54, FN=75, TP=1297

Why these metrics matter:
- ROC-AUC close to 1.0 indicates strong separation between classes.
- In medical screening, reducing **false negatives** (FN) is especially important because missing an infected case can be more harmful than a false alarm.

## FastAPI Deployment
The project includes a FastAPI app with:
- `GET /` → HTML UI (Jinja2 template)
- `POST /predict` → upload an image and receive `{label, probability}`

Returned probability is `P(Parasitized)` from the model sigmoid output.

## Folder Structure

```bash
Malaria-Cell-Classification-System/
├── __pycache__/
├── model/
│   ├── class_names.json
│   └── malaria_cnn.keras
├── notebook/
│   └── Malaria_Cell_Classification.ipynb
├── templates/
│   └── index.html
├── app.py
└── readme.md
```

## How to Run (Local API)
```bash
cd app
pip install fastapi uvicorn tensorflow pillow jinja2 python-multipart
uvicorn main:app --reload --port 8000
```
Open: http://127.0.0.1:8000

## Disclaimer
**Educational only.** This model is a learning project and must not be used for medical diagnosis or clinical decision-making without proper validation, approvals, and domain expert review.

## Acknowledgements
- Dataset credits go to the Kaggle dataset owners/maintainers.
- Built with TensorFlow/Keras and FastAPI.

#### Author : **M Haroon Abbas** (https://github.com/RealHaroon)

---