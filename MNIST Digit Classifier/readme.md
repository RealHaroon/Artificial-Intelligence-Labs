# 🔢 MNIST Digit Classifier — Draw & Predict

A deep learning lab that recognizes handwritten digits (0–9). A simple dense neural network is trained on the MNIST dataset with TensorFlow/Keras and served through **FastAPI** with a browser UI where you **draw a digit on a canvas** and get an instant prediction.

---

## 🎯 Objectives

- Load, visualize and normalize the MNIST dataset.
- Build and train a small fully connected network in Keras.
- Evaluate on the held-out test set.
- Deploy the model behind a FastAPI endpoint that accepts canvas drawings.
- Bridge the gap between "clean" MNIST images and real user drawings with MNIST-style preprocessing.

---

## 📁 Project Structure

```text
MNIST Digit Classifier/
├── app.py                              # FastAPI backend (UI + /predict)
├── model/
│   ├── mnist.keras                     # Trained model (loaded by app.py)
│   └── mnist_model.h5                  # Legacy HDF5 copy of the model
├── notebook/
│   └── MNIST Digit Classifier.ipynb    # Training notebook
├── templates/
│   └── index.html                      # Drawing canvas UI
├── debug/
│   └── last_28x28.png                  # Last preprocessed input (written on each prediction)
└── readme.md
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Source | `tf.keras.datasets.mnist` |
| Images | 70,000 grayscale, 28 × 28 |
| Train / Test | 60,000 / 10,000 |
| Classes | 10 digits (0–9) |

Pixel values are scaled from `[0, 255]` to `[0, 1]`.

---

## 🧠 Model Architecture

```text
Flatten(input_shape=(28, 28))      → 784
Dense(128, activation="relu")
Dropout(0.2)
Dense(10, activation="softmax")
```

- **Parameters:** 101,770
- **Optimizer:** Adam
- **Loss:** SparseCategoricalCrossentropy
- **Epochs:** 5

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test accuracy | **97.78%** |
| Validation accuracy after epoch 1 | 96.0% |

---

## ✏️ Canvas Preprocessing (app.py)

User drawings look very different from MNIST digits, so `preprocess_pil()` mimics how MNIST was built:

1. Convert to grayscale and scale to `[0, 1]`.
2. Auto-invert if the background is light (MNIST is white-on-black).
3. Threshold and crop to the digit's bounding box.
4. Resize the digit to fit a **20 × 20** box, keeping aspect ratio.
5. Paste it into a **28 × 28** canvas with a 4 px margin.
6. Shift by **center of mass** so the digit is centered.

Empty canvases are rejected with `400 No digit drawn.` The last processed image is saved to `debug/last_28x28.png` so you can see what the model actually received.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn tensorflow pillow jinja2 numpy
```

### 2. (Optional) Retrain the model

Run all cells in `notebook/MNIST Digit Classifier.ipynb`. The last cell saves the model to an absolute Windows path; change it to `model/mnist.keras` if you are on a different machine.

### 3. Start the server

From inside this folder:

```bash
uvicorn app:app --reload
```

### 4. Open the UI

- **Web UI:** http://127.0.0.1:8000 — draw a digit and click predict.
- **Swagger docs:** http://127.0.0.1:8000/docs

---

## 📡 API Reference

### `GET /`

Serves the drawing canvas UI.

### `POST /predict`

Accepts a base64-encoded PNG, usually from `canvas.toDataURL()`.

**Request:**

```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg..."
}
```

**Response** (illustrative):

```json
{
  "prediction": 7,
  "confidence": 0.9987,
  "probs": [0.0001, 0.0002, 0.0004, 0.0001, 0.0000, 0.0001, 0.0000, 0.9987, 0.0001, 0.0003]
}
```

**Error:** `400 {"detail": "No digit drawn."}` when the canvas is empty.

---

## ⚠️ Notes

- The model ends in a `softmax` layer, so `app.py` returns its output directly as probabilities (no second softmax).
- A CNN would push accuracy above 99% and is a natural next step for this lab.

---

## 👨‍💻 Author

**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
