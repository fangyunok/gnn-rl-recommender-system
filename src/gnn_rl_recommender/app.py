import os
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .service import RecommenderService

app = FastAPI(
    title="GNN-RL Personalized Recommender",
    version="0.1.0",
    description="LightGCN candidate retrieval and reinforcement-learning reranking API",
)
service = RecommenderService(os.getenv("MODEL_DIR", "artifacts"))


class RecommendationItem(BaseModel):
    item_id: int
    score: float
    category: int
    reason: str


class RecommendationResponse(BaseModel):
    user_id: int
    items: list[RecommendationItem]
    model_version: str = "lightgcn-policy-v1"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "artifact_loaded": service.loaded_from_artifact,
        "num_users": service.data.num_users,
        "num_items": service.data.num_items,
    }


@app.get("/v1/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(
    user_id: int, limit: int = Query(default=10, ge=1, le=50)
) -> RecommendationResponse:
    try:
        items = service.recommend(user_id, limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecommendationResponse(
        user_id=user_id,
        items=[RecommendationItem(**asdict(item)) for item in items],
    )
