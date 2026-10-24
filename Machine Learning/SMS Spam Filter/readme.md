# 📱 SMS Spam Filter — Naive Bayes NLP

An end-to-end Natural Language Processing lab that classifies SMS messages as **Spam** or **Ham** (legitimate). A **Multinomial Naive Bayes** model on **TF-IDF** features is served through **FastAPI** with a chat-style Bootstrap UI.

---

## 🚀 Tech Stack

- **Machine Learning:** `scikit-learn` (`TfidfVectorizer`, `MultinomialNB`), `pandas`
- **Backend:** `FastAPI`, `uvicorn`, `pydantic`
- **Frontend:** HTML5, Bootstrap 5, Vanilla JavaScript (Fetch API)
- **Serialization:** `joblib`

---

## 📁 Project Structure

```text
SMS Spam Filter/
├── app.py                      # FastAPI backend (UI + /predict)
├── model/
│   ├── spam_model.joblib       # Trained MultinomialNB
│   └── vectorizer.joblib       # Fitted TfidfVectorizer
├── notebook/
│   └── notebook.ipynb          # Training notebook
├── templates/
│   └── index.html              # Chat-style UI
├── ss.png                      # UI screenshot
├── test.txt                    # Sample spam / ham messages
└── readme.md
```

---

## 📊 Dataset

**SMS Spam Collection** (Kaggle / UCI) — 5,572 English SMS messages labeled `ham` or `spam`. The notebook downloads it directly from a public GitHub mirror, so no local data file is needed.

---

## 🔬 ML Pipeline

1. **Load** the CSV and keep only the label (`v1`) and message (`v2`) columns.
2. **Encode labels:** `ham → 0`, `spam → 1`.
3. **Split:** 80% train / 20% test (`random_state=42`).
4. **Vectorize** with `TfidfVectorizer(stop_words="english")` — fit on train, transform test.
5. **Train** `MultinomialNB`.
6. **Evaluate** with accuracy and a classification report.
7. **Save** both the model and the vectorizer (the API needs the same vocabulary).

---

## 📈 Results (Test Set, 1,115 messages)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Ham | 0.96 | 1.00 | 0.98 |
| Spam | **1.00** | 0.75 | 0.86 |
| **Accuracy** | | | **96.68%** |

- **Spam precision of 1.00:** a message flagged as spam is always spam — no legitimate messages get blocked.
- **Spam recall of 0.75:** about a quarter of spam slips through. Class balancing or a tuned decision threshold would raise recall.

---

## ✨ Features

- **TF-IDF vectorization** with English stop-word removal.
- **Confidence score** returned with every prediction.
- **Interactive UI** that changes color for safe vs. spam messages.
- **Swagger docs** generated at `/docs`.

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn jinja2 joblib scikit-learn pandas
```

### 2. Start the server

From inside this folder:

```bash
uvicorn app:app --reload
```

- **Web UI:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs

---

## 📡 API Reference

### `POST /predict`

**Request:**

```json
{
  "message": "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
}
```

**Response** (illustrative):

```json
{
  "label": "SPAM",
  "confidence": "98.45%"
}
```

Legitimate messages return `"label": "HAM (Clean)"`. More examples are in `test.txt`.

---

## 👨‍💻 Author

**M Haroon Abbas** — Data Scientist & ML Engineer
[GitHub @RealHaroon](https://github.com/RealHaroon)
