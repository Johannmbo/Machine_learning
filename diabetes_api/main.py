from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier
import joblib
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diabetes-api")

app = FastAPI(title="Diabetes Prediction API", version="1.0.0")

# --- Load artifacts once at startup, not per-request ---
model = CatBoostClassifier()
model.load_model("catboost_diabetes_model.cbm")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

logger.info(f"Model loaded. Expected features: {feature_columns}")


# --- Input schema (mirrors the Pima Diabetes dataset columns) ---
class PatientInput(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: int = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 2,
                "Glucose": 130,
                "BloodPressure": 70,
                "SkinThickness": 25,
                "Insulin": 100,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.45,
                "Age": 35
            }
        }


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_label: str


@app.get("/health")
def health_check():
    """Basic liveness check — used by load balancers/k8s probes."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    try:
        # Enforce the exact column order the scaler/model were fit on
        input_df = pd.DataFrame([patient.dict()])[feature_columns]

        # Same preprocessing as training: scale, then predict
        scaled = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled, columns=feature_columns)

        pred = int(model.predict(scaled_df)[0])
        proba = float(model.predict_proba(scaled_df)[:, 1][0])

        return PredictionOutput(
            prediction=pred,
            probability=round(proba, 4),
            risk_label="High risk" if pred == 1 else "Low risk"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
def predict_batch(patients: list[PatientInput]):
    """Batch endpoint — avoids N separate HTTP round trips."""
    input_df = pd.DataFrame([p.dict() for p in patients])[feature_columns]
    scaled = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled, columns=feature_columns)

    preds = model.predict(scaled_df)
    probas = model.predict_proba(scaled_df)[:, 1]

    return [
        {"prediction": int(p), "probability": round(float(pr), 4)}
        for p, pr in zip(preds, probas)
    ]