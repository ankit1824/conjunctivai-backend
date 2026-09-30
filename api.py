"""
ConjunctivAI — FastAPI backend (memory-optimised for Render Free)
Start: uvicorn api:app --host 0.0.0.0 --port 8000
"""

import io, json, time
from pathlib import Path

import numpy as np
import cv2
import joblib
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR     = Path(__file__).parent
MODELS_DIR   = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MASK_THRESH  = 10

FEAT_NAMES = (
    [f"RGB {c} {s}"   for c in ["R","G","B"]     for s in ["mean","std"]] +
    [f"HSV {c} {s}"   for c in ["H","S","V"]     for s in ["mean","std"]] +
    [f"LAB {c} {s}"   for c in ["L","A","B"]     for s in ["mean","std"]] +
    [f"YCbCr {c} {s}" for c in ["Y","Cb","Cr"]   for s in ["mean","std"]] +
    ["R/G ratio", "R/(R+G+B)"] +
    ["GLCM contrast","GLCM homogeneity","GLCM energy","GLCM correlation"] +
    [f"LBP {i}" for i in range(10)]
)

MODEL_META = {
    "xgboost": {"name": "XGBoost",      "family": "Gradient Boosting", "color": "#F6C90E"},
    "rf":      {"name": "Random Forest","family": "Bagged Trees",      "color": "#FF6B6B"},
    "svm":     {"name": "SVM (RBF)",    "family": "Margin-based",      "color": "#4EA8DE"},
    "knn":     {"name": "KNN",          "family": "Instance-based",    "color": "#56CFB2"},
    "gnb":     {"name": "Gaussian NB",  "family": "Probabilistic",     "color": "#C77DFF"},
}

# ── Pure-numpy GLCM (replaces scikit-image) ───────────────────────────────────
def _glcm_props(gray_q: np.ndarray, levels: int = 64):
    """Compute GLCM features without scikit-image."""
    h, w = gray_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    # horizontal co-occurrence
    i_idx = gray_q[:, :-1].ravel().astype(int)
    j_idx = gray_q[:, 1:].ravel().astype(int)
    np.add.at(glcm, (i_idx, j_idx), 1)
    # vertical co-occurrence
    i_idx = gray_q[:-1, :].ravel().astype(int)
    j_idx = gray_q[1:, :].ravel().astype(int)
    np.add.at(glcm, (i_idx, j_idx), 1)
    # symmetrise + normalise
    glcm += glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total

    ix = np.arange(levels)
    iy = np.arange(levels)
    II, JJ = np.meshgrid(ix, iy, indexing='ij')
    diff2 = (II - JJ) ** 2

    contrast    = float((glcm * diff2).sum())
    homogeneity = float((glcm / (1 + diff2)).sum())
    energy      = float((glcm ** 2).sum())

    mu_i = (glcm * II).sum()
    mu_j = (glcm * JJ).sum()
    sig_i = np.sqrt(((glcm * (II - mu_i) ** 2)).sum())
    sig_j = np.sqrt(((glcm * (JJ - mu_j) ** 2)).sum())
    if sig_i * sig_j > 1e-10:
        correlation = float(((glcm * (II - mu_i) * (JJ - mu_j)).sum()) / (sig_i * sig_j))
    else:
        correlation = 0.0

    return contrast, homogeneity, energy, correlation


# ── Pure-numpy LBP (replaces scikit-image) ───────────────────────────────────
def _lbp_uniform(gray: np.ndarray) -> np.ndarray:
    """8-point radius-1 uniform LBP without scikit-image."""
    h, w = gray.shape
    pad = np.pad(gray.astype(np.int32), 1, mode='edge')
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        nbr = pad[1+dy:h+1+dy, 1+dx:w+1+dx]
        code |= ((nbr >= gray.astype(np.int32)) << bit).astype(np.uint8)
    # count transitions to determine uniformity
    lbp_out = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            c = int(code[i, j])
            # number of 0→1 or 1→0 transitions in circular bit string
            bits = [(c >> b) & 1 for b in range(8)]
            trans = sum(bits[b] != bits[(b+1) % 8] for b in range(8))
            lbp_out[i, j] = c if trans <= 2 else 9  # 9 = non-uniform bin
    return lbp_out


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    img_rgb = np.array(img)
    mask = img_rgb.sum(axis=2) > MASK_THRESH

    if mask.sum() < 50:
        raise HTTPException(status_code=422,
            detail="Image appears mostly black — upload a valid conjunctiva crop.")

    px_rgb = img_rgb[mask].astype(np.float32)
    feat = []

    conversions = [None, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YCrCb]
    for conv in conversions:
        px = px_rgb if conv is None else cv2.cvtColor(img_rgb, conv).astype(np.float32)[mask]
        for ch in range(3):
            feat += [px[:, ch].mean(), px[:, ch].std()]

    r, g, b = px_rgb[:, 0], px_rgb[:, 1], px_rgb[:, 2]
    feat.append(float((r / (g + 1e-5)).mean()))
    feat.append(float((r / (r + g + b + 1e-5)).mean()))

    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_q = (gray // 4).astype(np.uint8)
    gray_q[~mask] = 0
    contrast, homogeneity, energy, correlation = _glcm_props(gray_q, levels=64)
    feat += [contrast, homogeneity, energy, correlation]

    lbp = _lbp_uniform(gray)
    hist, _ = np.histogram(lbp[mask], bins=10, range=(0, 10), density=True)
    feat.extend(hist.tolist())

    return np.array(feat, dtype=np.float32)


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ConjunctivAI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LOADED_MODELS: dict = {}

@app.on_event("startup")
def load_models():
    for key in MODEL_META:
        p = MODELS_DIR / f"{key}.joblib"
        if p.exists():
            LOADED_MODELS[key] = joblib.load(p)
            print(f"Loaded {key}")
        else:
            print(f"WARNING: {key}.joblib not found")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(LOADED_MODELS.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not LOADED_MODELS:
        raise HTTPException(503, "Models not loaded.")
    img_bytes = await file.read()
    t0   = time.time()
    feat = extract_features(img_bytes)
    feat_time = (time.time() - t0) * 1000

    results, votes = [], []
    for key, meta in MODEL_META.items():
        if key not in LOADED_MODELS:
            continue
        model = LOADED_MODELS[key]
        t1    = time.time()
        prob  = float(model.predict_proba(feat.reshape(1, -1))[0][1])
        label = int(model.predict(feat.reshape(1, -1))[0])
        infer_ms = (time.time() - t1) * 1000
        votes.append(label)

        fi = None
        if hasattr(model, "feature_importances_"):
            imp    = model.feature_importances_
            top5   = np.argsort(imp)[-5:][::-1]
            fi     = [{"feature": FEAT_NAMES[i], "importance": round(float(imp[i]), 4)} for i in top5]

        results.append({
            "key": key, "name": meta["name"], "family": meta["family"],
            "color": meta["color"], "label": label,
            "label_text": "Anemic" if label == 1 else "Non-Anemic",
            "probability": round(prob, 4),
            "infer_ms": round(infer_ms, 2),
            "feature_importance": fi,
        })

    consensus = 1 if sum(votes) >= 3 else 0
    return {
        "consensus": consensus,
        "consensus_text": "Anemic" if consensus == 1 else "Non-Anemic",
        "votes_anemic": sum(votes),
        "feat_time_ms": round(feat_time, 1),
        "models": results,
    }


@app.get("/metrics")
def get_metrics():
    p = ARTIFACT_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/roc_data")
def get_roc_data():
    p = ARTIFACT_DIR / "roc_data.json"
    if not p.exists():
        raise HTTPException(404, "roc_data.json not found.")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/artifacts/{filename}")
def serve_artifact(filename: str):
    p = ARTIFACT_DIR / filename
    if not p.exists():
        raise HTTPException(404, f"{filename} not found.")
    return FileResponse(p, media_type="image/png")
