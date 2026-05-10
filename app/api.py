from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from model.recommender import CollaborativeFilteringRecommender, train_from_csv

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/app/artifacts"))
DEFAULT_DATASET = DATA_DIR / "sample_interactions.csv"
MODEL_PATH = ARTIFACT_DIR / "recommender.joblib"
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

app = FastAPI(title="Mini-project 3 RecSys Deployment", version="1.0.0")
_model: Optional[CollaborativeFilteringRecommender] = None


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["U001"])
    n: int = Field(10, ge=1, le=50)
    include_seen: bool = False


class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    rating: float = Field(..., ge=1, le=5)
    comment: str = ""


def load_or_train_model() -> CollaborativeFilteringRecommender:
    global _model
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _model is not None:
        return _model
    if MODEL_PATH.exists():
        _model = CollaborativeFilteringRecommender.load(MODEL_PATH)
    elif DEFAULT_DATASET.exists():
        _model = train_from_csv(DEFAULT_DATASET, MODEL_PATH)
    else:
        raise RuntimeError("No model artifact or dataset found. Add data/sample_interactions.csv or artifacts/recommender.joblib.")
    return _model


@app.on_event("startup")
def startup() -> None:
    load_or_train_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        model = load_or_train_model()
        return {"status": "ok", "model_loaded": True, "metrics": model.metrics_}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return load_or_train_model().metrics_


@app.get("/users")
def users() -> Dict[str, List[str]]:
    model = load_or_train_model()
    if model.user_item_ is None:
        return {"users": []}
    return {"users": list(map(str, model.user_item_.index[:500]))}


@app.post("/recommend")
def recommend(req: RecommendationRequest) -> Dict[str, Any]:
    model = load_or_train_model()
    try:
        recs = model.recommend(req.user_id, req.n, req.include_seen)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"user_id": req.user_id, "recommendations": recs}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([req.model_dump()])
    exists = FEEDBACK_PATH.exists()
    row.to_csv(FEEDBACK_PATH, mode="a", header=not exists, index=False)
    return {"status": "saved", "message": "Feedback stored. Use retraining to incorporate it into the model."}


@app.post("/train")
async def train(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = DATA_DIR / "uploaded_interactions.csv"
    upload_path.write_bytes(await file.read())
    global _model
    try:
        _model = train_from_csv(upload_path, MODEL_PATH)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "trained", "metrics": _model.metrics_}


@app.get("/risk-assessment")
def risk_assessment() -> Dict[str, Any]:
    model = load_or_train_model()
    m = model.metrics_
    risks = [
        {"risk": "Cold-start users/items", "impact": "New users and items have little historical data.", "mitigation": "Add onboarding preferences, popularity fallback, and metadata/content-based recommendations."},
        {"risk": "Popularity bias", "impact": f"Current popularity-bias ratio is {m.get('popularity_bias', 'n/a')}.", "mitigation": "Diversify ranking, cap repeated popular items, monitor long-tail coverage."},
        {"risk": "Privacy", "impact": "Interaction logs can identify user preferences.", "mitigation": "Minimize stored identifiers, pseudonymize user IDs, restrict access, document retention policy."},
        {"risk": "Model drift", "impact": "Recommendations can degrade as user behavior changes.", "mitigation": "Schedule retraining, compare RMSE/MAE/precision over time, collect explicit feedback."},
        {"risk": "Explainability", "impact": "Users may not trust unexplained suggestions.", "mitigation": "Show predicted rating, item popularity, and reason fields in the GUI."},
    ]
    return {"risks": risks, "improvement_proposal": "Incorporate saved feedback into a retraining job and add A/B testing for ranking strategies."}
