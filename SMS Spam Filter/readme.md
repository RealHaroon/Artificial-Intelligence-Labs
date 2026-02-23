# 🚫 SMS Spam Detector - AI Lab 

An end-to-end Natural Language Processing (NLP) project that classifies SMS messages as either "Spam" or "Ham" (legitimate). This project serves a **Multinomial Naive Bayes** machine learning model through a lightning-fast **FastAPI** backend, complete with a responsive chat-style frontend UI.

## 🚀 Tech Stack
* **Machine Learning:** `scikit-learn` (MultinomialNB, TfidfVectorizer), `pandas`
* **Backend:** `FastAPI`, `uvicorn`, `pydantic`
* **Frontend:** `HTML5`, `Bootstrap 5`, `Vanilla JavaScript` (Fetch API)
* **Serialization:** `joblib`

## ✨ Features
* **TF-IDF Vectorization:** Converts raw text into numerical features while automatically filtering out uninformative English stop-words.
* **Real-time Inference:** The Naive Bayes algorithm processes text and returns predictions in milliseconds.
* **Confidence Scoring:** Returns a probability percentage alongside the prediction so users can see exactly how confident the AI is.
* **Interactive UI:** A clean, single-page web interface that dynamically changes color based on whether a message is safe or spam.
* **API Documentation:** Auto-generated Swagger UI available at `/docs`.

## 🛠️ Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RealHaroon/Artificial-Intelligence-Labs/tree/main/SMS%20Spam%20Filter](https://github.com/RealHaroon/Artificial-Intelligence-Labs/tree/main/SMS%20Spam%20Filter)
   cd sms-spam-detector```

2. **Install the required dependencies:**

```bash
pip install -r requirements.txt

```


3. **Run the FastAPI server:**
```bash
uvicorn app:app --reload

```


4. **Access the application:**
* **Web UI:** Open `http://127.0.0.1:8000` in your browser.
* **API Docs:** Open `http://127.0.0.1:8000/docs` to test the endpoints.



## 📡 API Endpoint Details

### `POST /predict`

Accepts a JSON payload containing a raw text message and returns the classification along with a confidence score.

**Request Body:**

```json
{
  "message": "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
}

```

**Response:**

```json
{
  "label": "SPAM",
  "confidence": "98.45%"
}

```

## 👨‍💻 Author

**M Haroon Abbas**  | Data Scientist and ML Eng

GitHub: [@RealHaroon](https://github.com/RealHaroon)

