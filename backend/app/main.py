from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .recommender import ContentBasedBookRecommender

#nacitame cesty k suborom
BOOKS_MODEL_PATH = os.environ.get("BOOKS_MODEL_PATH", "/app/data/books_model.csv")
EXPERIMENTS_PATH = os.environ.get(
    "EXPERIMENTS_PATH", "/app/results/experiment_results.csv"
)
#pripadny novy csv file sa ulozi sem
ACTIVE_UPLOAD_PATH = "/app/data/uploaded_books_model.csv"

#instancia web appky 
app = FastAPI(
    title="Mini-project 3: Goodbooks Content-Based Recommender",
    description="Production deployment of Mini-project 1: TF-IDF over cleaned book tags + cosine similarity with optional average-rating reranking."
)
#cbf model
model = ContentBasedBookRecommender(BOOKS_MODEL_PATH, EXPERIMENTS_PATH)

#definovanie requestov
#mode je sposob recommendation (cold start, user profile..)
#slidery pre hybrid recc, pocet reccs
class RecommendRequest(BaseModel):
    mode: str = Field("title", description="title | genre | profile | global | auto")
    title: str | None = None
    genre: str | None = None
    favorite_titles: list[str] = []
    n: int = Field(10, ge=1, le=30)
    hybrid: bool = Field(
        True,
        description="For similarity modes, rerank by cosine similarity + normalized average_rating",
    )
    similarity_weight: float = Field(0.8, ge=0, le=1)
    rating_weight: float = Field(0.2, ge=0, le=1)

#forma feedbacku (info o knihe + useful/not useful)
class FeedbackRequest(BaseModel):
    mode: str
    user_input: str = ""
    title: str
    decision: str = Field(..., description="useful | not_useful")
    comment: str = ""

#pomocne endpointy pre app
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return model.model_info


@app.get("/titles")
def titles(q: str = "", limit: int = 25) -> dict[str, list[str]]:
    return {"titles": model.search_titles(q, limit)}


@app.get("/experiments")
def experiments() -> dict[str, Any]:
    if model.experiments_df.empty:
        return {"experiments": []}
    return {"experiments": model.experiments_df.to_dict(orient="records")}

#recc endpoint
#kontrola weights, aby neboli nulove
#podla mode sa spusti metoda modelu
@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    total = req.similarity_weight + req.rating_weight
    if total <= 0:
        raise HTTPException(400, "At least one score weight must be > 0")
    sw, rw = req.similarity_weight / total, req.rating_weight / total

    if req.mode == "global":
        return {
            "strategy": "Cold start: top-rated books",
            "items": model.global_top_rated(req.n),
        }
    if req.mode == "genre":
        if not req.genre:
            raise HTTPException(400, "genre is required")
        return {
            "strategy": f"Cold start: top-rated books for tag '{req.genre}'",
            "items": model.genre_top_rated(req.genre, req.n),
        }
    if req.mode == "profile":
        if not req.favorite_titles:
            raise HTTPException(400, "favorite_titles is required")
        return {
            "strategy": "User Profile Recommender",
            "items": model.user_profile(req.favorite_titles, req.n, req.hybrid, sw, rw),
        }
    if req.mode == "auto":
        user_input = (
            req.favorite_titles
            if req.favorite_titles
            else (req.title or req.genre or None)
        )
        return model.final(user_input, req.n, req.hybrid, sw, rw)
    if not req.title:
        raise HTTPException(400, "title is required")
    return {
        "strategy": f"Content-based filtering for '{req.title}'",
        "items": model.by_title(req.title, req.n, req.hybrid, sw, rw),
    }

#ulozenie feedbacku do feedback.csv
@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, Any]:
    if req.decision not in {"useful", "not_useful"}:
        raise HTTPException(400, "decision must be useful or not_useful")
    return model.save_feedback(
        req.mode, req.user_input, req.title, req.decision, req.comment
    )

#pocas behu je mozne uploadnut vlastny "model" (csv file)
#ak csv subor nebude mat required columns tak nebude funogvat
@app.post("/upload-books-model")
async def upload_books_model(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file")
    content = await file.read()
    Path(ACTIVE_UPLOAD_PATH).write_bytes(content)
    global model
    model = ContentBasedBookRecommender(ACTIVE_UPLOAD_PATH, EXPERIMENTS_PATH)
    return {
        "uploaded": True,
        "active_path": ACTIVE_UPLOAD_PATH,
        "model_info": model.model_info,
    }

#resetovanie modelu
@app.post("/reset-sample-model")
def reset_sample_model() -> dict[str, Any]:
    global model
    model = ContentBasedBookRecommender(BOOKS_MODEL_PATH, EXPERIMENTS_PATH)
    return {"reset": True, "model_info": model.model_info}
