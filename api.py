"""
ConjunctivAI — FastAPI backend
uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import io, json, time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import cv2
import joblib
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

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
    "xgboost": {"name": "XGBoost",       "family": "Gradient Boosting", "color": "#F6C90E"},
    "rf":      {"name": "Random Forest", "family": "Bagged Trees",      "color": "#FF6B6B"},
    "svm":     {"name": "SVM (RBF)",     "family": "Margin-based",      "color": "#4EA8DE"},
    "knn":     {"name": "KNN",           "family": "Instance-based",    "color": "#56CFB2"},
    "gnb":     {"name": "Logistic Reg",  "family": "Probabilistic",     "color": "#C77DFF"},
}

# ── GLCM (pure numpy) ─────────────────────────────────────────────────────────
def _glcm_props(gray_q: np.ndarray, levels: int = 64):
    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (gray_q[:, :-1].ravel().astype(int), gray_q[:, 1:].ravel().astype(int)), 1)
    np.add.at(glcm, (gray_q[:-1, :].ravel().astype(int), gray_q[1:, :].ravel().astype(int)), 1)
    glcm += glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    ix = np.arange(levels)
    II, JJ = np.meshgrid(ix, ix, indexing='ij')
    diff2 = (II - JJ) ** 2
    contrast    = float((glcm * diff2).sum())
    homogeneity = float((glcm / (1.0 + diff2)).sum())
    energy      = float((glcm ** 2).sum())
    mu_i  = (glcm * II).sum()
    mu_j  = (glcm * JJ).sum()
    sig_i = np.sqrt((glcm * (II - mu_i) ** 2).sum())
    sig_j = np.sqrt((glcm * (JJ - mu_j) ** 2).sum())
    corr  = float(((glcm * (II - mu_i) * (JJ - mu_j)).sum()) / (sig_i * sig_j + 1e-10))
    return contrast, homogeneity, energy, corr

# ── LBP (vectorised, no pixel loops) ─────────────────────────────────────────
def _lbp_uniform(gray: np.ndarray) -> np.ndarray:
    h, w   = gray.shape
    pad    = np.pad(gray.astype(np.int32), 1, mode='edge')
    center = gray.astype(np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    code = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        code |= ((pad[1+dy:h+1+dy, 1+dx:w+1+dx] >= center) << bit).astype(np.uint8)
    bits       = np.stack([(code >> b) & 1 for b in range(8)], axis=0).astype(np.uint8)
    transitions = (bits != np.roll(bits, -1, axis=0)).sum(axis=0)
    return np.where(transitions <= 2, code.astype(np.int32), 9)

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img_bytes: bytes) -> np.ndarray:
    img     = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    img_rgb = np.array(img)
    mask    = img_rgb.sum(axis=2) > MASK_THRESH
    if mask.sum() < 50:
        raise HTTPException(422, "Image appears mostly black — upload a valid conjunctiva crop.")
    px_rgb = img_rgb[mask].astype(np.float32)
    feat   = []
    for conv in [None, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YCrCb]:
        px = px_rgb if conv is None else cv2.cvtColor(img_rgb, conv).astype(np.float32)[mask]
        for ch in range(3):
            feat += [px[:, ch].mean(), px[:, ch].std()]
    r, g, b = px_rgb[:,0], px_rgb[:,1], px_rgb[:,2]
    feat += [float((r / (g + 1e-5)).mean()), float((r / (r + g + b + 1e-5)).mean())]
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_q = (gray // 4).astype(np.uint8); gray_q[~mask] = 0
    feat  += list(_glcm_props(gray_q, levels=64))
    lbp    = _lbp_uniform(gray)
    hist, _ = np.histogram(lbp[mask], bins=10, range=(0, 10), density=True)
    feat   += hist.tolist()
    return np.array(feat, dtype=np.float32)

# ── Model store ───────────────────────────────────────────────────────────────
MODELS: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    for key in MODEL_META:
        p = MODELS_DIR / f"{key}.joblib"
        try:
            MODELS[key] = joblib.load(p)
            print(f"OK  {key}")
        except Exception as e:
            print(f"ERR {key}: {e}")
    yield
    MODELS.clear()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ConjunctivAI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.options("/{path:path}")
async def options_handler(path: str):
    return Response(status_code=200, headers={
        "Access-Control-Allow-Origin":  "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODELS:
        raise HTTPException(503, "Models not loaded — retry in a moment.")
    img_bytes = await file.read()
    t0   = time.time()
    feat = extract_features(img_bytes)
    feat_ms = (time.time() - t0) * 1000

    results, votes = [], []
    for key, meta in MODEL_META.items():
        if key not in MODELS:
            continue
        model = MODELS[key]
        t1    = time.time()
        prob  = float(model.predict_proba(feat.reshape(1, -1))[0][1])
        label = int(model.predict(feat.reshape(1, -1))[0])
        infer_ms = (time.time() - t1) * 1000
        votes.append(label)
        fi = None
        if hasattr(model, "feature_importances_"):
            imp  = model.feature_importances_
            top5 = np.argsort(imp)[-5:][::-1]
            fi   = [{"feature": FEAT_NAMES[i], "importance": round(float(imp[i]), 4)} for i in top5]
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
        "feat_time_ms": round(feat_ms, 1),
        "models": results,
    }

@app.get("/metrics")
def get_metrics():
    p = ARTIFACT_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(404, "metrics.json not found")
    return JSONResponse(json.loads(p.read_text()))

@app.get("/roc_data")
def get_roc_data():
    p = ARTIFACT_DIR / "roc_data.json"
    if not p.exists():
        raise HTTPException(404, "roc_data.json not found")
    return JSONResponse(json.loads(p.read_text()))

@app.get("/artifacts/{filename}")
def serve_artifact(filename: str):
    p = ARTIFACT_DIR / filename
    if not p.exists():
        raise HTTPException(404, f"{filename} not found")
    return FileResponse(p, media_type="image/png")
