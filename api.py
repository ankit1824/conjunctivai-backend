"""
ConjunctivAI — FastAPI backend
Start: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import io, json, time, re
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import joblib
from PIL import Image

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MASK_THRESH = 10
FEATURE_COLS = [f"f{i}" for i in range(40)]

FEAT_NAMES = (
    [f"RGB {c} {s}" for c in ["R","G","B"] for s in ["mean","std"]] +
    [f"HSV {c} {s}" for c in ["H","S","V"] for s in ["mean","std"]] +
    [f"LAB {c} {s}" for c in ["L","A","B"] for s in ["mean","std"]] +
    [f"YCbCr {c} {s}" for c in ["Y","Cb","Cr"] for s in ["mean","std"]] +
    ["R/G ratio", "R/(R+G+B)"] +
    ["GLCM contrast","GLCM homogeneity","GLCM energy","GLCM correlation"] +
    [f"LBP {i}" for i in range(10)]
)

MODEL_META = {
    "xgboost": {"name": "XGBoost",         "family": "Gradient Boosting", "color": "#F6C90E"},
    "rf":      {"name": "Random Forest",    "family": "Bagged Trees",      "color": "#FF6B6B"},
    "svm":     {"name": "SVM (RBF)",        "family": "Margin-based",      "color": "#4EA8DE"},
    "knn":     {"name": "KNN",              "family": "Instance-based",    "color": "#56CFB2"},
    "gnb":     {"name": "Gaussian NB",      "family": "Probabilistic",     "color": "#C77DFF"},
}

# ─── Load models on startup ───────────────────────────────────────────────────
app = FastAPI(title="ConjunctivAI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOADED_MODELS: dict = {}

@app.on_event("startup")
def load_models():
    for key in MODEL_META:
        p = MODELS_DIR / f"{key}.joblib"
        if p.exists():
            LOADED_MODELS[key] = joblib.load(p)
            print(f"Loaded {key}.joblib")
        else:
            print(f"WARNING: {key}.joblib not found — run train.py first")


# ─── Feature extraction (identical to train.py) ───────────────────────────────
def extract_features(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    img_rgb = np.array(img)
    mask = img_rgb.sum(axis=2) > MASK_THRESH

    if mask.sum() < 50:
        raise HTTPException(status_code=422,
            detail="Image appears to be mostly black — please upload a valid conjunctiva crop.")

    px_rgb = img_rgb[mask].astype(np.float32)
    feat   = []

    conversions = [None, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YCrCb]
    for conv in conversions:
        px = px_rgb if conv is None else cv2.cvtColor(img_rgb, conv).astype(np.float32)[mask]
        for ch in range(3):
            feat += [px[:, ch].mean(), px[:, ch].std()]

    r, g, b = px_rgb[:, 0], px_rgb[:, 1], px_rgb[:, 2]
    feat.append((r / (g + 1e-5)).mean())
    feat.append((r / (r + g + b + 1e-5)).mean())

    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_q = (gray // 4).astype(np.uint8)
    gray_q[~mask] = 0
    glcm = graycomatrix(gray_q, [1], [0, np.pi / 2], levels=64, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feat.append(graycoprops(glcm, prop).mean())

    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=10, density=True)
    feat.extend(hist.tolist())

    return np.array(feat, dtype=np.float32)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(LOADED_MODELS.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not LOADED_MODELS:
        raise HTTPException(503, "Models not loaded. Run train.py first.")

    img_bytes = await file.read()
    t0 = time.time()
    feat = extract_features(img_bytes)
    feat_time = (time.time() - t0) * 1000

    results = []
    votes   = []
    for key, meta in MODEL_META.items():
        if key not in LOADED_MODELS:
            continue
        model = LOADED_MODELS[key]
        t1 = time.time()
        prob  = float(model.predict_proba(feat.reshape(1, -1))[0][1])
        label = int(model.predict(feat.reshape(1, -1))[0])
        infer_ms = (time.time() - t1) * 1000
        votes.append(label)

        # Feature importance for tree models
        fi = None
        if key in ("xgboost", "rf"):
            raw_model = model
            imp = raw_model.feature_importances_
            top5_idx = np.argsort(imp)[-5:][::-1]
            fi = [{"feature": FEAT_NAMES[i], "importance": round(float(imp[i]), 4)}
                  for i in top5_idx]

        results.append({
            "key":        key,
            "name":       meta["name"],
            "family":     meta["family"],
            "color":      meta["color"],
            "label":      label,
            "label_text": "Anemic" if label == 1 else "Non-Anemic",
            "probability": round(prob, 4),
            "infer_ms":   round(infer_ms, 2),
            "feature_importance": fi,
        })

    consensus = 1 if sum(votes) >= 3 else 0
    return {
        "consensus":       consensus,
        "consensus_text":  "Anemic" if consensus == 1 else "Non-Anemic",
        "votes_anemic":    sum(votes),
        "feat_time_ms":    round(feat_time, 1),
        "models":          results,
    }


@app.get("/metrics")
def get_metrics():
    p = ARTIFACT_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found. Run train.py first.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/roc_data")
def get_roc_data():
    p = ARTIFACT_DIR / "roc_data.json"
    if not p.exists():
        raise HTTPException(404, "roc_data.json not found. Run train.py first.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/artifacts/{filename}")
def serve_artifact(filename: str):
    p = ARTIFACT_DIR / filename
    if not p.exists():
        raise HTTPException(404, f"{filename} not found.")
    return FileResponse(p, media_type="image/png")
