from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime

from src.api.model_loader import (
    load_model_from_registry,
    load_registry
)
from src.api.feature_builder import (
    build_feature_vector,
    build_features_from_datetime,
)

# 🔥 unified inference
from src.inference.predictor import predict as unified_predict

# 🔥 drift logging
from src.monitoring.prediction_logger import log_prediction

app = FastAPI(title="Energy Forecast API", version="0.6.1")


# ======================
# Schemas
# ======================
class PredictRequest(BaseModel):
    state: str
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class PredictFromDatetimeRequest(BaseModel):
    state: str
    datetime: datetime


# ======================
# Health
# ======================
@app.get("/health")
def health():
    return {"status": "ok"}


def _available_states(full_cfg: Dict[str, Any]):
    block = full_cfg.get("states") or full_cfg.get("state_models") or {}
    if not isinstance(block, dict):
        return []
    return list(block.keys())


# ======================
# Model Info
# ======================
@app.get("/model-info/{state}")
def model_info(state: str):

    state = state.strip().upper()

    try:
        _, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    full_cfg = load_registry()

    return {
        "state": state,
        "loader": state_cfg.get("loader") or full_cfg.get("loader"),
        "model_type": state_cfg.get("model_type"),
        "feature_names": state_cfg.get("feature_names"),
        "available_states": _available_states(full_cfg),
    }


# ======================================================
# PREDICT (FEATURE-BASED)
# ======================================================
@app.post("/predict")
def predict(req: PredictRequest):

    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_type = state_cfg.get("model_type", "xgboost")
    feature_names = state_cfg.get("feature_names", [])

    try:
        # ===============================
        # Prophet
        # ===============================
        if model_type == "prophet":
            y = unified_predict(
                model=model,
                model_type=model_type,
                data=req.features  # kullanılmaz
            )
            pred = float(y[0])

        # ===============================
        # Baseline
        # ===============================
        elif model_type == "baseline":
            y = unified_predict(
                model=model,
                model_type=model_type,
                data=req.features
            )
            pred = float(y[0])

        # ===============================
        # Feature-based
        # ===============================
        else:
            missing = [f for f in feature_names if f not in req.features]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Eksik feature'lar: {missing}"
                )

            x = build_feature_vector(req.features, feature_names)

            y = unified_predict(
                model=model,
                model_type=model_type,
                data=x
            )
            pred = float(y[0])

        # 🔥 LOG PREDICTION
        log_prediction(
            state=state,
            model_type=model_type,
            prediction=pred,
            features=req.features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    return {
        "state": state,
        "model_type": model_type,
        "prediction": pred
    }


# ======================================================
# PREDICT FROM DATETIME
# ======================================================
@app.post("/predict-from-datetime")
def predict_from_datetime(req: PredictFromDatetimeRequest):

    state = req.state.strip().upper()

    try:
        model, state_cfg = load_model_from_registry(state=state, force=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_type = state_cfg.get("model_type", "xgboost")

    if model_type == "baseline":
        raise HTTPException(
            status_code=400,
            detail="Baseline model datetime-based prediction desteklemez."
        )

    try:
        # ===============================
        # Prophet
        # ===============================
        if model_type == "prophet":
            y = unified_predict(
                model=model,
                model_type=model_type,
                data=req.datetime
            )
            pred = float(y[0])

            log_features = {"datetime": req.datetime.isoformat()}

        # ===============================
        # Feature-based
        # ===============================
        else:
            features = build_features_from_datetime(state, req.datetime)
            feature_names = state_cfg.get("feature_names", [])
            x = build_feature_vector(features, feature_names)

            y = unified_predict(
                model=model,
                model_type=model_type,
                data=x
            )
            pred = float(y[0])
            log_features = features

        # 🔥 LOG PREDICTION
        log_prediction(
            state=state,
            model_type=model_type,
            prediction=pred,
            features=log_features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict hatası: {str(e)}")

    return {
        "state": state,
        "datetime": req.datetime.isoformat(),
        "model_type": model_type,
        "prediction": pred
    }