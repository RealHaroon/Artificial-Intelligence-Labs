
# Malaria Cell Classification with CNN (Binary Image Classification)

A simple Convolutional Neural Network (CNN) that classifies thin blood smear cell images into **Parasitized** (infected) vs **Uninfected** using TensorFlow/Keras. The goal is to demonstrate CNN basics (convolutions, feature maps, pooling) and a complete ML workflow from dataset loading to deployment with a FastAPI endpoint.

## Objective
- Train a lightweight CNN (no transfer learning) to detect malaria-infected cells from images.
- Keep the pipeline beginner-friendly: small input size (64×64), limited epochs, and clear evaluation using medical-relevant metrics.

## Dataset
- Kaggle: **Cell Images for Detecting Malaria** (two folders/classes: `Parasitized/` and `Uninfected/`).  
  Link: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
- 27,558 images in total, roughly balanced between the two classes.

## Model (What & Why)
- **CNN** is better than a basic ANN/MLP for images because it preserves spatial structure and learns local patterns (edges/textures) using convolution filters.
- Architecture used (simple + fast):
  - 3 × (Conv2D + MaxPooling2D) with 32 → 64 → 128 filters (3×3, `same` padding, ReLU)
  - Flatten → Dense(128, ReLU) → Dropout
  - Output: Dense(1, `sigmoid`) = P(Parasitized)
  - ~1.14M trainable parameters

## Training Pipeline (Creation Process)
1. Load images directly from folders using Keras directory loader.
2. Resize to **64×64**, normalize to **[0,1]** by dividing by 255.
3. Split:
   - Train (80%)
   - Validation/Test made from remaining 20% (then split roughly 50/50 by batches)
4. Train with Adam + Binary Cross-Entropy (up to 8 epochs, Kaggle Tesla T4 GPU) and EarlyStopping on validation AUC.
5. Save artifacts for inference:
   - `malaria_cnn.keras` (native Keras format, used by the app)
   - `malaria_savedmodel/` (exported SavedModel directory — produced in the notebook, not committed to this repo)
   - `class_names.json` (label mapping: `["Uninfected", "Parasitized"]`)

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
- `POST /predict` → upload a PNG/JPG image (form field `file`)

Example response:
```json
{
  "label": "Parasitized",
  "probability_parasitized": 0.9731,
  "threshold": 0.5,
  "disclaimer": "Educational only — not for medical diagnosis."
}
```

`probability_parasitized` is the model's sigmoid output. Non-image uploads return `415`, unreadable images return `400`.

## Folder Structure

```text
Malaria Cell Classification System/
├── model/
│   ├── class_names.json
│   └── malaria_cnn.keras
├── notebook/
│   └── Malaria_Cell_Classification.ipynb
├── slides/
│   └── Malaria Detection CNN.pptx
├── templates/
│   └── index.html
├── ss.png                 # UI screenshot
├── app.py
└── readme.md
```

## How to Run (Local API)

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn tensorflow pillow jinja2 python-multipart numpy
   ```
2. Start the server from inside this folder:
   ```bash
   uvicorn app:app --reload --port 8000
   ```
3. Open http://127.0.0.1:8000 (Swagger docs at `/docs`).

## Disclaimer
**Educational only.** This model is a learning project and must not be used for medical diagnosis or clinical decision-making without proper validation, approvals, and domain expert review.

## Acknowledgements
- Dataset credits go to the Kaggle dataset owners/maintainers.
- Built with TensorFlow/Keras and FastAPI.

#### Author : **M Haroon Abbas** (https://github.com/RealHaroon)

---