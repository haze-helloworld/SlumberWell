"""SlumberWell FastAPI Backend
Serves the trained sleep-stage classifier and exposes REST endpoints.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("slumberwell")

BASE_DIR    = Path(__file__).resolve().parent.parent
ARTIFACTS   = BASE_DIR / "artifacts"
FRONTEND    = BASE_DIR / "frontend"

MODEL_PATH    = ARTIFACTS / "best_model.joblib"
CONFIG_PATH   = ARTIFACTS / "preprocessing_config.json"
METRICS_PATH  = ARTIFACTS / "evaluation_metrics.json"
FEATURES_PATH = ARTIFACTS / "feature_names.json"
LABELS_PATH   = ARTIFACTS / "label_mapping.json"
SC4001_PATH   = ARTIFACTS / "sc4001_demo.json"

log.info("Loading model from %s", MODEL_PATH)
MODEL = joblib.load(MODEL_PATH)

with open(CONFIG_PATH)   as f: CFG           = json.load(f)
with open(METRICS_PATH)  as f: METRICS       = json.load(f)
with open(FEATURES_PATH) as f: FEATURE_NAMES = json.load(f)
with open(LABELS_PATH)   as f: LABEL_MAPPING  = json.load(f)
with open(SC4001_PATH)   as f: SC4001_DEMO   = json.load(f)

STAGE_NAMES  = ["W", "N1", "N2", "N3", "REM"]
INT_TO_LABEL = {int(k): v for k, v in LABEL_MAPPING.items()}
STAGE_COLOURS = {"W": "#64748b", "N1": "#818cf8", "N2": "#38bdf8", "N3": "#2563eb", "REM": "#f472b6"}

log.info("Model and artifacts loaded.")

app = FastAPI(title="SlumberWell API", description="Sleep-stage classification", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class EpochFeaturesRequest(BaseModel):
    features: list[float] = Field(..., min_length=16, max_length=40)

class BatchFeaturesRequest(BaseModel):
    epochs: list[list[float]]

class PredictionResponse(BaseModel):
    stage: str
    stage_int: int
    probabilities: dict[str, float]
    inference_ms: float

class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict[str, Any]
    hypnogram: list[str]
    inference_ms_total: float

def _summarize(stages, epoch_len_sec=30):
    stages_arr = np.asarray(stages)
    n = len(stages_arr)
    mins = epoch_len_sec / 60.0
    trt  = n * mins
    asleep = stages_arr != "W"
    tst    = float(asleep.sum() * mins)
    onset  = int(np.argmax(asleep)) if asleep.any() else None
    rem_idx = np.where(stages_arr == "REM")[0]
    rem_lat = None
    if onset is not None and len(rem_idx):
        rem_lat = float((rem_idx[0] - onset) * mins)
    counts = {s: int((stages_arr == s).sum()) for s in STAGE_NAMES}
    return {
        "total_recording_min": round(trt, 2),
        "total_sleep_min":     round(tst, 2),
        "sleep_efficiency_pct": round(100 * tst / trt, 2) if trt else 0.0,
        "sleep_onset_latency_min": None if onset is None else round(float(onset * mins), 2),
        "rem_latency_min": None if rem_lat is None else round(rem_lat, 2),
        "stage_counts": counts,
        "stage_percent": {s: round(100 * c / n, 2) if n else 0.0 for s, c in counts.items()},
    }

# Baseline physiological statistics for single-epoch reference normalization
BASELINE_STATS = {
    "mean": (0.0, 0.20), "std": (16.08, 8.94), "var": (338.4, 417.7), "rms": (16.08, 8.94),
    "min": (-61.46, 30.82), "max": (62.89, 31.14), "range": (124.35, 59.83),
    "delta_power": (224.27, 298.65), "theta_power": (22.12, 26.37), "alpha_power": (9.96, 12.33), "beta_power": (15.30, 38.99),
    "rel_delta": (0.765, 0.153), "rel_theta": (0.104, 0.056), "rel_alpha": (0.058, 0.054), "rel_beta": (0.072, 0.082),
    "spectral_entropy": (0.644, 0.115)
}
BASE_NAMES = [
    "mean", "std", "var", "rms", "min", "max", "range",
    "delta_power", "theta_power", "alpha_power", "beta_power",
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "spectral_entropy"
]

def _prepare_features(feat_list):
    n_expected = getattr(MODEL, "n_features_in_", len(FEATURE_NAMES))
    if len(feat_list) == n_expected:
        return np.array(feat_list, dtype=float).reshape(1, -1)
    
    # If 16 features were provided and model expects 40
    raw = list(feat_list[:16])
    # Check if raw needs baseline z-score (e.g. if std or variance are non-normalized)
    if CFG.get("normalize_per_recording", False):
        z16 = []
        for i, val in enumerate(raw):
            name = BASE_NAMES[i]
            mu, sigma = BASELINE_STATS.get(name, (0.0, 1.0))
            z16.append((val - mu) / (sigma + 1e-8))
    else:
        z16 = raw

    if n_expected == 40:
        f_dict = {BASE_NAMES[i]: z16[i] for i in range(16)}
        expanded = list(z16)
        for k in ["std", "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "spectral_entropy"]:
            val = f_dict[k]
            expanded.extend([val, val, val, val]) # prev1, next1, prev2, next2
        return np.array(expanded, dtype=float).reshape(1, -1)
    return np.array(z16, dtype=float).reshape(1, -1)

def _calc_probas(X, pred_int):
    if hasattr(MODEL, "predict_proba"):
        try:
            return MODEL.predict_proba(X)[0]
        except Exception:
            pass
    if hasattr(MODEL, "decision_function"):
        df = MODEL.decision_function(X)[0]
        exp_df = np.exp(df - np.max(df))
        return exp_df / exp_df.sum()
    arr = np.zeros(len(STAGE_NAMES))
    arr[pred_int] = 1.0
    return arr

def _predict_one(features):
    X = _prepare_features(features)
    t0 = time.perf_counter()
    pred_int  = int(MODEL.predict(X)[0])
    proba_arr = _calc_probas(X, pred_int)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    stage = INT_TO_LABEL[pred_int]
    proba = {INT_TO_LABEL[i]: round(float(p), 4) for i, p in enumerate(proba_arr)}
    return PredictionResponse(stage=stage, stage_int=pred_int, probabilities=proba, inference_ms=round(elapsed_ms, 3))

@app.get("/api/health")
def health():
    return {"status": "ok", "model": METRICS.get("model_name"), "version": "1.0.0"}

@app.get("/api/info")
def info():
    return {
        "model_name": METRICS.get("model_name"),
        "hyperparameters": METRICS.get("hyperparameters"),
        "feature_names": FEATURE_NAMES,
        "stage_names": STAGE_NAMES,
        "stage_colours": STAGE_COLOURS,
        "config": CFG,
        "split": METRICS.get("split"),
        "library_versions": METRICS.get("library_versions"),
    }

@app.get("/api/metrics")
def metrics():
    test = METRICS.get("test", {})
    return {
        "model_name": METRICS.get("model_name"),
        "test": test,
        "validation": METRICS.get("validation"),
        "split": METRICS.get("split"),
        "class_names": METRICS.get("class_names", STAGE_NAMES),
    }

@app.get("/api/metrics/comparison")
def model_comparison():
    return {"models": METRICS.get("model_comparison_validation", [])}

@app.get("/api/metrics/tuning")
def tuning_trials():
    return {"trials": METRICS.get("tuning_trials", [])}

@app.post("/api/predict")
def predict_single(req: EpochFeaturesRequest):
    try:
        return _predict_one(req.features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/predict/batch")
def predict_batch(req: BatchFeaturesRequest):
    if not req.epochs:
        raise HTTPException(status_code=400, detail="epochs list is empty")
    t0 = time.perf_counter()
    preds = []
    stages = []
    for ep in req.epochs:
        single_res = _predict_one(ep)
        preds.append(single_res)
        stages.append(single_res.stage)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        predictions=preds,
        summary=_summarize(stages),
        hypnogram=stages,
        inference_ms_total=round(elapsed_ms, 3)
    )

@app.get("/api/demo")
def demo_prediction():
    """Return the real SC4001 whole-night hypnogram and sleep summary."""
    return {
        "subject":     SC4001_DEMO["subject"],
        "description": SC4001_DEMO["description"],
        "hypnogram":     SC4001_DEMO["hypnogram"],
        "hypno_encoded": SC4001_DEMO["hypno_encoded"],
        "summary":       SC4001_DEMO["summary"],
        "stage_colours": STAGE_COLOURS,
    }

if FRONTEND.exists():
    try:
        static_dir = FRONTEND / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    except Exception:
        pass

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str = ""):
        index = FRONTEND / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Frontend not found"}

log.info("SlumberWell API ready. Docs at http://127.0.0.1:8000/docs")
