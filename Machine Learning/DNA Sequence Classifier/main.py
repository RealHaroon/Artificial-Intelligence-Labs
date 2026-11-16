"""FastAPI app for the DNA Sequence Classifier.

Run from the project folder:
    uvicorn main:app --reload
Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (API docs)
"""
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix, hstack

BASE = Path(__file__).parent
bundle = joblib.load(BASE / "model/model.joblib")
kmer = bundle["vectorizer"]
model = bundle["model"]
classes = bundle["classes"]
cv_accuracy = bundle["cv_accuracy"]

app = FastAPI(title="DNA Sequence Classifier", version="1.0")


class DNAInput(BaseModel):
    sequence: str = Field(..., min_length=3, max_length=10_000, examples=["ATGCGTACGTTAGC"])
    mutation_flag: int = Field(0, ge=0, le=1)


def clean_sequence(raw: str) -> str:
    seq = "".join(raw.split()).upper()  # remove spaces and newlines
    bad = set(seq) - set("ACGT")
    if bad:
        raise HTTPException(400, f"Only A, C, G, T allowed. Found: {', '.join(sorted(bad))}")
    if len(seq) < 3:
        raise HTTPException(400, "Sequence must have at least 3 bases.")
    return seq


def make_features(seq: str, mutation_flag: int):
    # Same steps as the notebook: k-mer counts, normalised by total count
    x_kmer = kmer.transform([seq])
    total = x_kmer.sum()
    if total > 0:
        x_kmer = x_kmer.multiply(1 / total)

    gc = (seq.count("G") + seq.count("C")) / len(seq) * 100
    # Order must match api_cols in the notebook: GC_Content, Sequence_Length, Mutation_Flag
    x_num = csr_matrix(np.array([[gc, len(seq), mutation_flag]], dtype=float))
    return hstack([x_kmer, x_num]).tocsr(), gc


@app.get("/")
def home():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "classes": classes, "cv_accuracy": round(cv_accuracy, 3)}


@app.post("/predict")
def predict(data: DNAInput):
    seq = clean_sequence(data.sequence)
    X, gc = make_features(seq, data.mutation_flag)

    probs = model.predict_proba(X)[0]
    best = int(probs.argmax())

    return {
        "prediction": classes[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "gc_content": round(gc, 2),
        "length": len(seq),
        "model_cv_accuracy": round(cv_accuracy, 3),
    }