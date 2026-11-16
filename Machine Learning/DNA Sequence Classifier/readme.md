# 🧬 DNA Sequence Classifier — k-mers + XGBoost

A machine learning lab that predicts the **organism class** of a DNA sequence (**Bacteria**, **Human**, **Plant** or **Virus**). Sequences are turned into **k-mer frequency** features plus simple biological statistics, models are compared with stratified cross-validation, and the final XGBoost model is served through **FastAPI** with a browser UI where you paste a sequence.

> **Key finding:** On this synthetic dataset, no model beats random guessing (~25% accuracy for 4 balanced classes). The sequences carry no signal that separates the classes. See [Results](#-results) for why this matters.

---

## 📁 Project Structure

```text
DNA Sequence Classifier/
├── main.py                         # FastAPI backend (UI + /health + /predict)
├── data/
│   └── synthetic_dna_dataset.csv   # 3,000 labeled sequences
├── model/
│   └── model.joblib                # Bundle: k-mer vectorizer + XGBoost + class names + CV accuracy
├── notebook/
│   ├── notebook.ipynb              # Main notebook: EDA → model comparison → final model → save
│   └── DNA_Classifier.ipynb        # Early prototype of the model comparison
├── static/
│   └── index.html                  # Paste-a-sequence UI
├── requirements.txt
└── readme.md
```

---

## 📊 Dataset

`data/synthetic_dna_dataset.csv` — **3,000 synthetic sequences**, each exactly 100 bases long, with no missing values or duplicates.

| Column | Description |
|--------|-------------|
| `Sequence` | DNA string (A, C, G, T) |
| `GC_Content`, `AT_Content` | Percentage of G+C and A+T bases |
| `Sequence_Length` | Always 100 in this dataset |
| `Num_A`, `Num_T`, `Num_C`, `Num_G` | Base counts |
| `kmer_3_freq` | Pre-computed 3-mer frequency score |
| `Mutation_Flag` | 0 or 1 |
| `Class_Label` | **Target:** Bacteria (761), Human (749), Plant (747), Virus (743) |
| `Disease_Risk` | Alternative target: High / Medium / Low |

EDA shows the classes are nearly identical on every feature: mean GC content is 49.8–50.7% for all four, and the mutation rate is ~50% in each.

---

## 🔬 ML Pipeline

1. **k-mer features:** `CountVectorizer(analyzer="char", ngram_range=(3, 4))` counts every 3- and 4-letter substring. Counts are normalized by the sequence's total so length doesn't dominate.
2. **Numeric features:** appended to the sparse k-mer matrix.
3. **Model comparison** with 5-fold `StratifiedKFold`:
   - Majority-class baseline (`DummyClassifier`)
   - Logistic Regression
   - XGBoost
4. **Final model:** XGBoost (`n_estimators=300, max_depth=5, learning_rate=0.1`), trained on all data using only features the API can compute from raw input — k-mers + `GC_Content`, `Sequence_Length`, `Mutation_Flag` (`kmer_3_freq` is dropped because a user can't supply it).
5. **Save** a single bundle with the vectorizer, model, class names and CV accuracy.

---

## 📈 Results

5-fold cross-validated accuracy on `Class_Label`:

| Model | CV Accuracy |
|-------|-------------|
| Baseline (always predict majority class) | 0.254 |
| Logistic Regression | 0.254 |
| XGBoost (all features) | 0.248 |
| **XGBoost (API features, deployed)** | **0.238** |

**Interpretation:** With 4 balanced classes, random guessing scores 0.25. Every model sits at that level, so the features contain **no information about the label**. The synthetic data was most likely generated with labels assigned independently of the sequences. This is an honest negative result, and an important ML lesson: always compare against a baseline before trusting a model.

To get a meaningful classifier, train on real sequences with biological signal (e.g., from NCBI GenBank), where k-mer composition differs between organisms.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

From inside this folder (the app file is `main.py`, not `app.py`):

```bash
uvicorn main:app --reload
```

- **Web UI:** http://127.0.0.1:8000
- **Swagger docs:** http://127.0.0.1:8000/docs

### 3. (Optional) Retrain

Run `notebook/notebook.ipynb`. Two paths need adjusting first:

- The CSV is read from an absolute Linux path; change it to `../data/synthetic_dna_dataset.csv`.
- The model is saved to `../model.joblib`, but `main.py` loads `model/model.joblib`. Change the save path to `../model/model.joblib`.

---

## 📡 API Reference

### `GET /health`

```json
{ "status": "ok", "classes": ["Bacteria", "Human", "Plant", "Virus"], "cv_accuracy": 0.238 }
```

### `POST /predict`

**Request:**

```json
{
  "sequence": "ATGCGTACGTTAGCATGCGTACGTTAGC",
  "mutation_flag": 0
}
```

- `sequence`: 3–10,000 characters. Spaces and newlines are stripped and letters are upper-cased; anything other than A/C/G/T returns `400`.
- `mutation_flag`: `0` or `1` (default `0`).

**Response** (illustrative):

```json
{
  "prediction": "Plant",
  "confidence": 0.3102,
  "probabilities": { "Bacteria": 0.2215, "Human": 0.2398, "Plant": 0.3102, "Virus": 0.2285 },
  "gc_content": 46.43,
  "length": 28,
  "model_cv_accuracy": 0.238
}
```

Probabilities cluster around 0.25 because the model has learned no real signal.

---

## 🔮 Next Steps

- Swap in a real labeled dataset (e.g., sequences from NCBI GenBank).
- Try `Disease_Risk` as the target (EDA suggests it is just as uninformative on this data, but it hasn't been cross-validated yet).
- Add a held-out test set and a confusion matrix once a dataset with signal is in place.

---

## 👨‍💻 Author

**M Haroon Abbas** — [GitHub @RealHaroon](https://github.com/RealHaroon)
