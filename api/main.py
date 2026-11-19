"""
api/main.py - FastAPI service for Coronary Heart Disease Risk Prediction
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import PredictionInput, PredictionResponse

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger("chd-api")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "chd_pipeline.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "model_metadata.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")

# In-memory model cache
state: Dict[str, Any] = {
    "pipeline": None,
    "metadata": None,
    "feature_importance": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML model pipeline and metadata once
    logger.info("Starting CHD Risk Prediction Service...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Train model first.")

    try:
        state["pipeline"] = joblib.load(MODEL_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            state["metadata"] = json.load(f)
        if os.path.exists(IMPORTANCE_PATH):
            with open(IMPORTANCE_PATH, "r", encoding="utf-8") as f:
                state["feature_importance"] = json.load(f)
        logger.info("Successfully loaded calibrated ML pipeline and metadata into memory.")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        raise e

    yield

    # Shutdown
    logger.info("Shutting down CHD Risk Prediction Service.")
    state.clear()


app = FastAPI(
    title="Coronary Heart Disease (CHD) Risk Prediction API",
    description="Production-quality calibrated 10-year coronary heart disease risk predictor based on the Framingham Heart Study.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
def health():
    """Health check endpoint to verify service and model availability."""
    is_ready = state["pipeline"] is not None and state["metadata"] is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_name": state["metadata"].get("model_name", "unknown") if is_ready else None
    }


@app.get("/model-info", status_code=status.HTTP_200_OK, tags=["Model"])
def model_info():
    """Returns complete model metadata, performance metrics, features, and importance."""
    if not state["metadata"]:
        raise HTTPException(status_code=503, detail="Model metadata not loaded.")
    return {
        "metadata": state["metadata"],
        "feature_importance": state.get("feature_importance", [])
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict(payload: PredictionInput):
    """
    Accepts validated patient clinical features and returns:
    - Calibrated 10-year coronary heart disease risk probability
    - Standard clinical risk tier (low <10%, moderate 10-20%, high >=20%)
    - Screening binary prediction based on optimized decision threshold
    """
    pipeline = state["pipeline"]
    meta = state["metadata"]
    if not pipeline or not meta:
        raise HTTPException(status_code=503, detail="Prediction model is not initialized.")

    try:
        # Construct DataFrame strictly matching the feature ordering from metadata
        feature_order = meta.get("features", [
            "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
            "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
            "sysBP", "diaBP", "BMI", "heartRate", "glucose"
        ])

        payload_dict = payload.model_dump()
        input_df = pd.DataFrame([{col: payload_dict.get(col, None) for col in feature_order}])

        # Probability prediction via calibrated pipeline
        prob = float(pipeline.predict_proba(input_df)[0, 1])

        # Threshold from metadata
        threshold = float(meta.get("decision_threshold", 0.12))
        binary_pred = 1 if prob >= threshold else 0

        # Clinical Risk Tiering (standard Framingham / ACC / AHA risk stratification):
        # <10% = Low Risk
        # 10% - 20% = Borderline / Moderate Risk
        # >=20% = High Risk (often guideline indication for primary statin therapy)
        if prob < 0.10:
            risk_level = "low"
            risk_label = "Low Risk (<10%)"
            explanation = (
                f"The estimated 10-year coronary heart disease risk is {prob:.1%}. "
                "This falls within the low cardiovascular risk tier (<10% 10-year risk)."
            )
        elif prob < 0.20:
            risk_level = "moderate"
            risk_label = "Moderate Risk (10% - 20%)"
            explanation = (
                f"The estimated 10-year coronary heart disease risk is {prob:.1%}. "
                "This indicates moderate/borderline cardiovascular risk (10-20% 10-year risk) and suggests lifestyle vigilance."
            )
        else:
            risk_level = "high"
            risk_label = "High Risk (>=20%)"
            explanation = (
                f"The estimated 10-year coronary heart disease risk is {prob:.1%}. "
                "This represents high cardiovascular risk (>=20% 10-year risk), warranting close medical review of lipids, blood pressure, and lifestyle."
            )

        logger.info(f"Prediction performed: prob={prob:.4f}, risk_level={risk_level}, threshold={threshold}")

        return PredictionResponse(
            probability=round(prob, 4),
            risk_level=risk_level,
            risk_label=risk_label,
            prediction=binary_pred,
            threshold=threshold,
            explanation=explanation,
            disclaimer="Educational demo. Not medical advice. Not clinically validated."
        )

    except Exception as e:
        logger.error(f"Prediction inference error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

