"""
ConjunctivAI — Feature extraction + model training
Run once: python train.py --data_dir /path/to/Conjuctiva
Outputs: manifest.csv, features.csv, models/*.joblib, artifacts/*.png, metrics.json
"""

import os, re, time, json, warnings, argparse
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image
from pathlib import Path
from collections import defaultdict

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import xgboost as xgb
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
SEED   = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MASK_THRESH = 10   # pixels with RGB sum <= this treated as background
BASE_DIR = Path(__file__).parent

# ─── 1. Build manifest (grouped, leakage-free split) ─────────────────────────
def build_manifest(data_dir: Path) -> pd.DataFrame:
    print("📂 Building manifest …")
    rows = []
    for cls_dir in sorted(data_dir.glob("*/*")):           # Training/Anemic etc.
        label = 1 if cls_dir.name == "Anemic" else 0
        for fpath in cls_dir.glob("*.png"):
            base = re.sub(r"_aug\d+$", "", fpath.stem)    # strip _augN
            rows.append({"path": str(fpath), "base_id": base, "label": label})
    df = pd.DataFrame(rows)

    # Assign split by unique base_id (grouped, stratified)
    np.random.seed(SEED)
    split_map = {}
    for label in [0, 1]:
        ids = sorted(df[df.label == label]["base_id"].unique())
        np.random.shuffle(ids)
        n = len(ids)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])
        for i, bid in enumerate(ids):
            if   i < n_train:          split_map[bid] = "train"
            elif i < n_train + n_val:  split_map[bid] = "val"
            else:                      split_map[bid] = "test"

    df["split"] = df["base_id"].map(split_map)
    df.to_csv(BASE_DIR / "manifest.csv", index=False)
    print(f"   Total images : {len(df)}")
    for s in ["train","val","test"]:
        sub = df[df.split==s]
        print(f"   {s:5s}: {len(sub):5d} imgs | {sub.label.sum()} anemic, {(sub.label==0).sum()} non-anemic")
    return df


# ─── 2. Feature extraction (40-dim, foreground-only) ─────────────────────────
def extract_features(img_path: str) -> np.ndarray:
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    mask = img_rgb.sum(axis=2) > MASK_THRESH
    if mask.sum() < 50:                        # degenerate image
        return np.zeros(40, dtype=np.float32)
    px_rgb = img_rgb[mask].astype(np.float32)

    feat = []

    # Colour stats: mean + std per channel in 4 colour spaces (24 dims)
    conversions = [None, cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2LAB, cv2.COLOR_RGB2YCrCb]
    for conv in conversions:
        if conv is None:
            px = px_rgb
        else:
            cs = cv2.cvtColor(img_rgb, conv).astype(np.float32)
            px = cs[mask]
        for ch in range(3):
            feat += [px[:, ch].mean(), px[:, ch].std()]

    # Redness ratios (2 dims)
    r, g, b = px_rgb[:, 0], px_rgb[:, 1], px_rgb[:, 2]
    feat.append((r / (g + 1e-5)).mean())
    feat.append((r / (r + g + b + 1e-5)).mean())

    # GLCM texture (4 dims) — 64-level quantisation for speed
    gray   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_q = (gray // 4).astype(np.uint8)
    gray_q[~mask] = 0
    glcm = graycomatrix(gray_q, [1], [0, np.pi / 2], levels=64, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        feat.append(graycoprops(glcm, prop).mean())

    # LBP histogram (10 dims)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp[mask], bins=10, density=True)
    feat.extend(hist.tolist())

    return np.array(feat, dtype=np.float32)


def build_features(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    feat_path = BASE_DIR / "features.csv"
    if feat_path.exists() and not force:
        print("⚡ Loading cached features.csv …")
        return pd.read_csv(feat_path)

    print("🔬 Extracting features (this takes ~4 min on CPU) …")
    t0 = time.time()
    feat_cols = [f"f{i}" for i in range(40)]
    records   = []
    total     = len(df)
    for idx, row in df.iterrows():
        feat = extract_features(row["path"])
        rec  = {"path": row["path"], "base_id": row["base_id"],
                "label": row["label"], "split": row["split"]}
        rec.update({f"f{i}": feat[i] for i in range(40)})
        records.append(rec)
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (total - idx - 1)
            print(f"   {idx+1}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    fdf = pd.DataFrame(records)
    fdf.to_csv(feat_path, index=False)
    print(f"✅ Features saved → features.csv  ({time.time()-t0:.1f}s)")
    return fdf


# ─── 3. Train 5 models ────────────────────────────────────────────────────────
FEATURE_COLS = [f"f{i}" for i in range(40)]

MODEL_CONFIGS = [
    {
        "name": "XGBoost",
        "key":  "xgboost",
        "needs_scale": False,
        "estimator": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                        random_state=SEED, n_jobs=1),
        "params": {
            "max_depth":     [3, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators":  [100, 300],
        },
    },
    {
        "name": "Random Forest",
        "key":  "rf",
        "needs_scale": False,
        "estimator": RandomForestClassifier(random_state=SEED, n_jobs=-1),
        "params": {
            "n_estimators": [100, 300],
            "max_depth":    [None, 10, 20],
        },
    },
    {
        "name": "SVM (RBF)",
        "key":  "svm",
        "needs_scale": True,
        "estimator": SVC(kernel="rbf", probability=True, random_state=SEED),
        "params": {
            "svc__C":     [0.1, 1, 10],
            "svc__gamma": ["scale", "auto"],
        },
    },
    {
        "name": "KNN",
        "key":  "knn",
        "needs_scale": True,
        "estimator": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 11],
        },
    },
    {
        "name": "Gaussian NB",
        "key":  "gnb",
        "needs_scale": True,
        "estimator": GaussianNB(),
        "params": {},   # no tuning
    },
]


def train_models(fdf: pd.DataFrame):
    print("\n🏋️  Training models …")
    train_df = fdf[fdf.split == "train"]
    X_train  = train_df[FEATURE_COLS].values
    y_train  = train_df["label"].values
    groups   = train_df["base_id"].values   # keep source IDs together in CV

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for cfg in MODEL_CONFIGS:
        t0   = time.time()
        name = cfg["name"]
        key  = cfg["key"]
        print(f"\n   ▶ {name} …")

        if cfg["needs_scale"]:
            scaler = StandardScaler()
            # Rename params to pipeline format if not already done
            raw_params = cfg["params"]
            est_key    = cfg["estimator"].__class__.__name__.lower()
            pipe_params = {}
            for k, v in raw_params.items():
                if not k.startswith(est_key):
                    pipe_params[f"{est_key}__{k}"] = v
                else:
                    pipe_params[k] = v
            pipe = Pipeline([("scaler", scaler), (est_key, cfg["estimator"])])
            if pipe_params:
                gs = GridSearchCV(pipe, pipe_params, cv=cv, scoring="f1",
                                  n_jobs=-1, refit=True)
                gs.fit(X_train, y_train, groups=groups)
                best_model = gs.best_estimator_
                print(f"     Best params: {gs.best_params_}")
            else:
                pipe.fit(X_train, y_train)
                best_model = pipe
        else:
            if cfg["params"]:
                gs = GridSearchCV(cfg["estimator"], cfg["params"], cv=cv,
                                  scoring="f1", n_jobs=-1, refit=True)
                gs.fit(X_train, y_train, groups=groups)
                best_model = gs.best_estimator_
                print(f"     Best params: {gs.best_params_}")
            else:
                cfg["estimator"].fit(X_train, y_train)
                best_model = cfg["estimator"]

        train_time = time.time() - t0

        # Save model
        model_path = BASE_DIR / "models" / f"{key}.joblib"
        joblib.dump(best_model, model_path)

        results[key] = {"name": name, "model": best_model, "train_time": train_time}
        print(f"     Done in {train_time:.1f}s → saved {key}.joblib")

    return results


# ─── 4. Evaluate ─────────────────────────────────────────────────────────────
def evaluate(fdf: pd.DataFrame, trained: dict) -> dict:
    print("\n📊 Evaluating on test set …")
    test_df = fdf[fdf.split == "test"]
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values

    all_metrics = {}
    roc_data    = {}

    for key, info in trained.items():
        model = info["model"]
        t0    = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        inf_time = (time.time() - t0) * 1000 / len(y_test)   # ms per image

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics = {
            "name":         info["name"],
            "accuracy":     round(accuracy_score(y_test, y_pred),   4),
            "precision":    round(precision_score(y_test, y_pred),  4),
            "recall":       round(recall_score(y_test, y_pred),     4),
            "f1":           round(f1_score(y_test, y_pred),         4),
            "specificity":  round(tn / (tn + fp),                   4),
            "roc_auc":      round(roc_auc_score(y_test, y_prob),    4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "train_time_s": round(info["train_time"], 2),
            "infer_time_ms": round(inf_time, 3),
        }
        all_metrics[key] = metrics
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[key] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                         "name": info["name"], "auc": metrics["roc_auc"]}

        print(f"   {info['name']:20s}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['roc_auc']:.4f}")

    # Save metrics
    with open(BASE_DIR / "artifacts" / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(BASE_DIR / "artifacts" / "roc_data.json", "w") as f:
        json.dump(roc_data, f, indent=2)

    return all_metrics, roc_data


# ─── 5. Plots ─────────────────────────────────────────────────────────────────
COLORS = {
    "xgboost": "#F6C90E",
    "rf":      "#FF6B6B",
    "svm":     "#4EA8DE",
    "knn":     "#56CFB2",
    "gnb":     "#C77DFF",
}

def make_plots(all_metrics: dict, roc_data: dict, fdf: pd.DataFrame, trained: dict):
    print("\n🎨 Generating plots …")
    art = BASE_DIR / "artifacts"

    # — ROC overlay —
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot([0, 1], [0, 1], "--", color="#484f58", lw=1.2)
    for key, rd in roc_data.items():
        ax.plot(rd["fpr"], rd["tpr"],
                label=f"{rd['name']} (AUC={rd['auc']:.3f})",
                color=COLORS[key], lw=2)
    ax.set_xlabel("False Positive Rate", color="#8b949e")
    ax.set_ylabel("True Positive Rate",  color="#8b949e")
    ax.set_title("ROC Curves — All 5 Models", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.legend(fontsize=9, facecolor="#161b22", labelcolor="#e6edf3",
              edgecolor="#30363d")
    fig.tight_layout()
    fig.savefig(art / "roc_curves.png", dpi=140, bbox_inches="tight")
    plt.close()

    # — Metrics comparison bar chart —
    metric_keys = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    models = list(all_metrics.keys())
    x = np.arange(len(metric_keys))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for i, key in enumerate(models):
        vals = [all_metrics[key][m] for m in metric_keys]
        ax.bar(x + i * width, vals, width, label=all_metrics[key]["name"],
               color=COLORS[key], alpha=0.88)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_keys], color="#8b949e")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color="#8b949e")
    ax.set_title("Model Performance — Test Set", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values(): sp.set_color("#30363d")
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d")
    fig.tight_layout()
    fig.savefig(art / "metrics_bar.png", dpi=140, bbox_inches="tight")
    plt.close()

    # — Confusion matrices —
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), facecolor="#0d1117")
    test_df = fdf[fdf.split == "test"]
    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["label"].values
    for ax, (key, info) in zip(axes, trained.items()):
        y_pred = info["model"].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_facecolor("#0d1117")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Anemic", "Anemic"], color="#8b949e", fontsize=9)
        ax.set_yticklabels(["Non-Anemic", "Anemic"], color="#8b949e", fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual",    color="#8b949e")
        ax.set_title(all_metrics[key]["name"], color=COLORS[key], fontsize=11)
        for sp in ax.spines.values(): sp.set_color("#30363d")
    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()
    fig.savefig(art / "confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close()

    # — Feature importance (XGBoost + RF) —
    feat_names = (
        [f"RGB_{c}_{s}" for c in ["R","G","B"] for s in ["mean","std"]] +
        [f"HSV_{c}_{s}" for c in ["H","S","V"] for s in ["mean","std"]] +
        [f"LAB_{c}_{s}" for c in ["L","A","B"] for s in ["mean","std"]] +
        [f"YCbCr_{c}_{s}" for c in ["Y","Cb","Cr"] for s in ["mean","std"]] +
        ["R/G_ratio", "R/(R+G+B)"] +
        ["GLCM_contrast","GLCM_homogeneity","GLCM_energy","GLCM_correlation"] +
        [f"LBP_{i}" for i in range(10)]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0d1117")
    for ax, key, title in zip(axes, ["xgboost", "rf"],
                               ["XGBoost Feature Importance", "Random Forest Feature Importance"]):
        model = trained[key]["model"]
        imp = (model.feature_importances_ if key == "xgboost"
               else model.feature_importances_)
        top_idx = np.argsort(imp)[-15:]
        ax.set_facecolor("#0d1117")
        ax.barh([feat_names[i] for i in top_idx], imp[top_idx],
                color=COLORS[key], alpha=0.85)
        ax.set_title(title, color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#30363d")
    fig.patch.set_facecolor("#0d1117")
    fig.tight_layout()
    fig.savefig(art / "feature_importance.png", dpi=140, bbox_inches="tight")
    plt.close()

    print("   Saved: roc_curves.png, metrics_bar.png, confusion_matrices.png, feature_importance.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../Conjuctiva",
                        help="Path to Conjuctiva folder (contains Training/, Validation/, Testing/)")
    parser.add_argument("--force_extract", action="store_true",
                        help="Re-extract features even if features.csv exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists(), f"Data dir not found: {data_dir}"

    manifest = build_manifest(data_dir)
    fdf      = build_features(manifest, force=args.force_extract)
    trained  = train_models(fdf)
    metrics, roc = evaluate(fdf, trained)
    make_plots(metrics, roc, fdf, trained)

    print("\n✅ All done. Run the API with: uvicorn api:app --reload")
