# Chest X-Ray Classifier (COVID-19 vs Pneumonia vs Normal) — DenseNet121 Transfer Learning

A deep learning project that classifies chest X-ray images into **COVID19**, **PNEUMONIA**, or **NORMAL** using **transfer learning** with **DenseNet121 (ImageNet pretrained)**. The project includes a full training notebook (Kaggle) and a FastAPI web app + HTML UI for inference.

> Disclaimer: **This model is for educational purposes only and must NOT be used for clinical diagnosis.**

***

## Project Overview

### Goal
Build a multi-class chest X-ray classifier with:
- Strong transfer learning performance
- Robust handling of **class imbalance**
- Proper evaluation beyond accuracy (macro-F1, per-class metrics, ROC-AUC)
- Explainability using **Grad-CAM**
- Deployment using **FastAPI** with a simple browser UI

### Classes
- `COVID19`
- `PNEUMONIA`
- `NORMAL`

***

## Folder Structure

```
Chest-X-Ray-Classifier/
│
├── model/
│   ├── chest_xray_densenet121.keras        # Final saved model
│   └── best_chest_xray.keras               # Best checkpoint (val_loss)
│
├── notebook/
│   └── Chest X-Ray Classifier COVID-19 Pneumonia and Normal Detection
│
├── templates/
│   └── index.html                          # UI page
│
└── app.py                                  # FastAPI backend
```

***

## Dataset

**Kaggle Dataset:** Chest X-ray (Covid-19 & Pneumonia)  
- Structure: `train/` and `test/` folders with subfolders per class  
- Imbalanced distribution: `PNEUMONIA >> NORMAL >> COVID19`

Why imbalance matters:
- A model can get high **accuracy** by focusing on the majority class (PNEUMONIA) while performing poorly on minority classes (COVID19 / NORMAL).

***

## Model Architecture (Transfer Learning)

### Why DenseNet121?
DenseNet121 is a strong CNN backbone because it:
- Reuses features efficiently via dense connections (better gradient flow)
- Performs well on medical imaging tasks where subtle texture patterns matter
- Works well with transfer learning from ImageNet when dataset size is limited

### Network Used
**Base:** `DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))`

**Custom classifier head:**
```
GlobalAveragePooling2D
Dense(128, activation="relu")
Dropout(0.5)
Dense(3, activation="softmax")
```

Input size: **224 × 224 × 3**  
Loss: **categorical_crossentropy**  
Optimizer: **Adam**  
Metric: **accuracy**

***

## Data Preprocessing & Augmentation

### Preprocessing
- Used `preprocess_input` from DenseNet (ImageNet-style normalization)

### Strong Augmentation (Train only)
- Rotation
- Zoom
- Width/Height shift
- Brightness adjustment
- Horizontal flip

Why augmentation?
- Helps generalization, especially when minority class (COVID19) has fewer examples.

***

## Handling Class Imbalance (Key Part)

We used **class weighting** computed from training labels:

- `compute_class_weight` (sklearn) produces higher weights for minority classes
- Keras `model.fit(..., class_weight=class_weights)` forces the model to “care more” about underrepresented classes

This improves performance on COVID19 / NORMAL without oversampling.

***

## Training Strategy (Two-Phase)

### Phase 1: Train Head Only (5 epochs)
- Freeze DenseNet121 base
- Train only the custom classifier head
- Faster and avoids destroying pretrained features

### Phase 2: Fine-tuning (Unfreeze last 30 layers)
- Unfreeze the last ~30 layers of DenseNet121
- Train with **lower learning rate = 1e-4**
- Allows model to adapt high-level features to X-ray patterns

***

## Callbacks Used

- **EarlyStopping** (monitor `val_loss`): prevents overfitting
- **ModelCheckpoint**: saves best model (lowest `val_loss`)
- **ReduceLROnPlateau**: reduces learning rate when validation stalls

***

## Evaluation Metrics (Why Not Only Accuracy?)

Because the dataset is imbalanced, accuracy can hide poor minority-class performance.

We evaluate using:
- **Accuracy**
- **Macro-F1** (best for imbalance: averages F1 across classes equally)
- **Per-class precision/recall/F1**
- **Confusion Matrix**
- **AUC-ROC (One-vs-Rest)**

### Macro-F1 vs Accuracy (Simple Explanation)
- **Accuracy**: “How often the model is correct overall”
- **Macro-F1**: “How balanced the performance is across ALL classes”
  - If COVID19 performance drops, macro-F1 drops significantly (good for fairness across classes)

***

## Results (Your Run)

From your test evaluation:
- Accuracy: **0.9270**
- Macro-F1: **0.9336**
- Macro AUC-ROC: **0.9943**

Per-class highlights:
- COVID19 F1 ≈ **0.9869** (excellent)
- NORMAL precision ≈ **0.7800** (main weakness: more false NORMAL predictions than ideal)
- PNEUMONIA F1 ≈ **0.9436** (strong)

### Pros of Your Scores
- Very strong COVID19 detection
- Excellent macro AUC suggests strong separability
- Macro-F1 is high, meaning performance is balanced overall

### Cons / Limitations
- NORMAL precision is lower, meaning some sick cases may be predicted as NORMAL
- Dataset bias, hospital/device differences, and shortcuts may exist
- Not clinically validated; must not be used for diagnosis

***

## Explainability: Grad-CAM

We generate **Grad-CAM heatmaps** for predictions:
- Highlights which lung regions influenced the model’s decision
- Useful to detect spurious attention (e.g., focusing on corners, text markers, borders)

Interpretation:
- Red/yellow regions = high attention
- Blue/purple = low attention

***

## Saving the Model

The final model is saved as:

- `model/chest_xray_densenet121.keras`

This includes model architecture + weights and can be reloaded for inference.

***

## Deployment (FastAPI + HTML UI)

### FastAPI Backend
- `POST /predict` accepts an image upload
- Returns:
  - predicted class
  - confidence score
  - class probabilities
  - disclaimer

### Web UI
- Drag-and-drop / click upload
- Image preview
- Results badge + probability bars

***

## How to Run Locally

### 1) Install Dependencies
```bash
pip install fastapi uvicorn tensorflow pillow jinja2 python-multipart
```

### 2) Start Server
From the project root:
```bash
uvicorn app:app --reload --port 8000
```

### 3) Open UI
- http://localhost:8000

***

## API Usage

### Health Check
```bash
GET /health
```

### Predict
```bash
POST /predict
Form-data: file=@your_xray.png
```

***

## Disclaimer (Mandatory)

**This model is for educational purposes only and must NOT be used for clinical diagnosis.**

***

## Future Improvements (Optional)
- Use focal loss or balanced softmax for harder imbalance handling
- Add patient-level split (avoid leakage if duplicates exist)
- Add calibration (temperature scaling) for better confidence reliability
- Add lung segmentation to reduce shortcut learning
- Add external validation dataset for real-world robustness

***

## Author
Developed by: **Haroon Khan**  
Project type: Deep Learning Lab (Transfer Learning + Deployment)

If you want, I can tailor this README into a GitHub-ready version with badges, screenshots placeholders, and a “Results” section that includes your plots (confusion matrix, ROC curves, Grad-CAM).