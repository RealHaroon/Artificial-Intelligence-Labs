# 🔍 **CIFAR-10 Multi-Class Object Recognition using CNN**

A deep learning web application that classifies images into **10 object categories**
using a Convolutional Neural Network (CNN) built from scratch with TensorFlow/Keras.
The model is served via a **FastAPI** backend with a modern dark-themed UI.

---

## 📸 Demo

> Upload any image → Get instant prediction with confidence scores for all 10 classes.

![UI Preview](templates/preview.png)

---

## 🗂️ Project Structure
```markdown

CIFAR-10/
│
├── model/
│   ├── cifar10_cnn_best.keras       # Best model saved by ModelCheckpoint
│   └── cifar10_cnn.keras            # Final trained model
│
├── notebook/
│   └── cifar-10.ipynb               # Full training notebook (EDA → Training → Evaluation)
│
├── templates/
│   └── index.html                   # Frontend UI (Drag & Drop, Results, Probability Bars)
│
└── app.py                           # FastAPI backend (serves UI + /predict endpoint)
```

---

## 🧠 Model Architecture

Built from scratch using TensorFlow/Keras — **no transfer learning**.

| Block         | Layers                                      | Output Shape  |
|---------------|---------------------------------------------|---------------|
| Input         | Input Layer                                 | 32 × 32 × 3   |
| Augmentation  | RandomFlip, RandomRotation, RandomZoom      | 32 × 32 × 3   |
| Conv Block 1  | Conv2D(32) × 2 + BatchNorm + MaxPool + Dropout(0.25) | 16 × 16 × 32  |
| Conv Block 2  | Conv2D(64) × 2 + BatchNorm + MaxPool + Dropout(0.30) | 8 × 8 × 64    |
| Conv Block 3  | Conv2D(128) × 2 + BatchNorm + MaxPool + Dropout(0.40)| 4 × 4 × 128   |
| Head          | Flatten → Dense(256) + BatchNorm + Dropout(0.50)     | 256           |
| Output        | Dense(10, softmax)                          | 10            |

- **Loss:** `categorical_crossentropy`
- **Optimizer:** `Adam (lr=1e-3)`
- **Metrics:** `accuracy`
- **Callbacks:** `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`

---

## 📊 Dataset — CIFAR-10

| Property        | Details                          |
|-----------------|----------------------------------|
| Source          | `tf.keras.datasets.cifar10`      |
| Total Images    | 60,000 RGB (32 × 32)             |
| Train / Test    | 50,000 / 10,000                  |
| Classes         | 10 (perfectly balanced)          |
| Samples/Class   | 6,000                            |

**Classes:** ✈️ Airplane · 🚗 Automobile · 🐦 Bird · 🐱 Cat · 🦌 Deer  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🐶 Dog · 🐸 Frog · 🐴 Horse · 🚢 Ship · 🚛 Truck

---

## 📈 Results

| Metric         | Value         |
|----------------|---------------|
| Test Accuracy  | ~85–88%       |
| Hardest Classes| Cat, Dog, Deer|
| Best Classes   | Ship, Automobile, Airplane |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cifar10-cnn.git
cd cifar10-cnn
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn python-multipart pillow tensorflow jinja2
```

### 3. Train the Model (optional — skip if model files already exist)

Open and run all cells in:
```
notebook/cifar-10.ipynb
```
This will save `cifar10_cnn_best.keras` and `cifar10_cnn.keras` to the `model/` folder.

### 4. Run the FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the Web UI

```
http://localhost:8000
```

---

## 🔌 API Reference

### `GET /`
Returns the web UI (HTML page).

---

### `GET /health`
Returns server and model status.

**Response:**
```json
{
  "status": "✅ running",
  "model": "model/cifar10_cnn_best.keras",
  "classes": ["Airplane", "Automobile", "..."]
}
```

---

### `POST /predict`
Upload an image to classify it.

**Request:**
- `Content-Type: multipart/form-data`
- Field: `file` (JPEG / PNG / WEBP)

**Response:**
```json
{
  "predicted_class": "Cat",
  "emoji": "🐱",
  "confidence": 87.34,
  "all_probs": [
    { "class": "Cat",  "emoji": "🐱", "probability": 87.34 },
    { "class": "Dog",  "emoji": "🐶", "probability":  8.12 },
    ...
  ]
}
```

**Error Response (invalid file):**
```json
{
  "detail": "Unsupported file type 'application/pdf'. Please upload a JPEG or PNG image."
}
```

---

## 🖥️ UI Features

- **Drag & Drop** or click-to-browse file upload
- **Live image preview** before prediction
- **Animated probability bars** (sorted highest → lowest)
- **Gold highlight** on top prediction
- **Supported classes grid** with emojis
- Fully **responsive** — works on mobile
- **Dark theme** throughout

---

## 🛠️ Tech Stack

| Layer       | Technology                     |
|-------------|--------------------------------|
| Model       | TensorFlow 2.x / Keras         |
| Backend     | FastAPI + Uvicorn              |
| Frontend    | HTML5 + CSS3 + Vanilla JS      |
| Templating  | Jinja2                         |
| Image Proc  | Pillow (PIL)                   |
| Training    | Kaggle Notebooks (GPU P100)    |

---

## 📓 Notebook Overview

The training notebook (`cifar-10.ipynb`) covers:

1. **Imports** — TensorFlow, Sklearn, Matplotlib, Seaborn
2. **EDA** — class distribution, sample grid, pixel intensity histogram
3. **Preprocessing** — normalization, one-hot encoding, augmentation pipeline
4. **Model** — CNN architecture with layer-by-layer explanations
5. **Training** — with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
6. **Evaluation** — accuracy, confusion matrix, per-class classification report
7. **Visualization** — training/validation accuracy & loss curves, prediction samples
8. **Model Saving** — `.keras` format with reload verification

---

## ⚠️ Notes

- The model expects images to be resized to **32×32** internally — you can upload any size image.
- For Kaggle deployment, use [ngrok](https://ngrok.com/) to expose the local server externally.
- Model falls back from `cifar10_cnn_best.keras` → `cifar10_cnn.keras` automatically if the best checkpoint is missing.

---

## 👤 Author

**Haroon Khan**  
Computer Science Student | Deep Learning & Computer Vision  

🔗 [GitHub](https://github.com/RealHaroon)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
