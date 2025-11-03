
import os
import json
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field


# Configuration 

MODEL_PATH = os.environ.get("MODEL_PATH", "model_final.joblib")
THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "thresholds_final.json")
DEFAULT_MODE = os.environ.get("PRED_MODE", "balanced")  # "recall" | "balanced" | "precision" pour choisir le type de prediciton que l'on veut
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", 0.5))


# charge le modele

try:
    bundle = joblib.load(MODEL_PATH)
    PIPE = bundle["pipeline"]
    FEATURES: List[str] = list(bundle["feature_names"])
except Exception as e:
    raise RuntimeError(f"Failed to load model bundle at {MODEL_PATH}: {e}")

# charge les seuils si présent
THRESHOLDS: Optional[Dict[str, Any]] = None
if os.path.exists(THRESHOLDS_PATH):
    try:
        with open(THRESHOLDS_PATH, "r") as f:
            THRESHOLDS = json.load(f)
    except Exception:
        THRESHOLDS = None


# Utilitaires

def pick_threshold(mode: str, thresholds: Optional[Dict[str, Any]]) -> float:
    mode = (mode or "").lower()
    if thresholds is None:
        return DEFAULT_THRESHOLD
    if mode == "recall":
        return float(thresholds.get("best_fbeta_threshold", DEFAULT_THRESHOLD))
    if mode in ("precision", "confident", "confidence"):
        return float(thresholds.get("target_recall_threshold", DEFAULT_THRESHOLD))
    return DEFAULT_THRESHOLD

def compute_engineered(payload: Dict[str, Any], row: Dict[str, float]) -> None:
    # Calcul de PTS/GP et PTS/MIN seulement si atendues dans les features
    if "PTS/GP" in FEATURES:
        gp = float(payload.get("GP", row.get("GP", 0.0)))
        pts = float(payload.get("PTS", row.get("PTS", 0.0)))
        row["PTS/GP"] = (pts / gp) if gp > 0 else 0.0
    if "PTS/MIN" in FEATURES:
        minu = float(payload.get("MIN", row.get("MIN", 0.0)))
        pts = float(payload.get("PTS", row.get("PTS", 0.0)))
        row["PTS/MIN"] = (pts / minu) if minu > 0 else 0.0


# FastAPI

app = FastAPI(
    title="Classifieur pour prédire si un joueur restera plus de 5 ans en NBA",
    description=(
        "API REST unitaire pour prédire si un joueur restera en NBA au moins 5 ans. "
        "Entrée: stats d'un joueur (JSON). Sortie: probabilité, prédiction 0/1, seuil utilisé."
    ),
    version="1.0.0",
)

class PredictResponse(BaseModel):
    model_path: str
    mode: str
    threshold: float
    probability: float
    prediction: int
    missing_filled: list[str]
    n_features: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "thresholds_path": THRESHOLDS_PATH if THRESHOLDS is not None else None,
        "default_mode": DEFAULT_MODE,
        "default_threshold": DEFAULT_THRESHOLD,
        "n_features": len(FEATURES),
        "feature_sample": FEATURES[:12] + (["..."] if len(FEATURES) > 12 else []),
    }

@app.get("/schema")
def schema():
    return {"features": FEATURES, "count": len(FEATURES)}

@app.post("/predict", response_model=PredictResponse)
def predict(
    payload: Dict[str, Any],
    mode: str = Query(DEFAULT_MODE, description="recall | balanced | precision"),
    threshold: Optional[float] = Query(None, description="Optionnel: override du seuil"),
):
    # Alignement de la payload sur le schéma attendu
    row: Dict[str, float] = {k: float(payload.get(k, 0.0)) for k in FEATURES}
    missing = [k for k in FEATURES if k not in payload]

    # calcul des features apres feature engineering attendues dans le schema
    compute_engineered(payload, row)

    X = pd.DataFrame([row], columns=FEATURES)

    # probabilité
    if hasattr(PIPE.named_steps["model"], "predict_proba"):
        prob = float(PIPE.predict_proba(X)[0, 1])
    elif hasattr(PIPE.named_steps["model"], "decision_function"):
        score = float(PIPE.decision_function(X)[0])
        # application d'une sigmoide, pour calculer une pseudo-proba
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        # fallback sur la prediction sinon
        prob = float(PIPE.predict(X)[0])

    # selection du seuil
    thr = float(threshold) if threshold is not None else pick_threshold(mode, THRESHOLDS)
    pred = int(prob >= thr)

    return PredictResponse(
        model_path=MODEL_PATH,
        mode=mode,
        threshold=thr,
        probability=prob,
        prediction=pred,
        missing_filled=missing,
        n_features=len(FEATURES),
    )
