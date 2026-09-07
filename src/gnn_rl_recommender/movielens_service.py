from __future__ import annotations

from pathlib import Path

import torch

from .data import normalized_bipartite_graph
from .lightgcn import LightGCN
from .movielens import load_movielens_1m
from .reranker import PolicyReranker
from .service import Recommendation


class MovieLensRecommenderService:
    """Online inference service backed by the trained MovieLens checkpoint."""

    dataset_name = "movielens-1m"

    def __init__(
        self,
        artifact_dir: str = "artifacts",
        dataset_dir: str = "data/raw/ml-1m",
        policy_weight: float = 0.2,
        candidate_k: int = 100,
    ):
        checkpoint = Path(artifact_dir) / "movielens_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint}; run scripts/experiment_movielens.py before real-data serving."
            )
        self.split = load_movielens_1m(Path(dataset_dir))
        self.data = self.split.train
        self.graph = normalized_bipartite_graph(self.data)
        self.gnn = LightGCN(self.data.num_users, self.data.num_items, dim=32, layers=2)
        self.reranker = PolicyReranker()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.gnn.load_state_dict(payload["gnn"])
        self.reranker.load_state_dict(payload["reranker"])
        self.gnn.eval()
        self.reranker.eval()
        self.policy_weight = policy_weight
        self.candidate_k = candidate_k
        self.loaded_from_artifact = True
        self.user_id_map = {
            original: internal for internal, original in enumerate(self.split.user_original_ids)
        }

    @torch.no_grad()
    def recommend(self, user_id: int, limit: int = 10) -> list[Recommendation]:
        if user_id not in self.user_id_map:
            raise ValueError("unknown MovieLens user_id")
        internal_user = self.user_id_map[user_id]
        users, items = self.gnn.propagate(self.graph)
        raw_relevance = items @ users[internal_user]
        raw_relevance[list(self.split.train_seen[internal_user])] = -torch.inf
        relevance, candidate_ids = torch.topk(
            raw_relevance, min(self.candidate_k, self.data.num_items)
        )
        categories = self.data.item_categories[candidate_ids]
        counts = torch.bincount(
            categories, minlength=int(self.data.item_categories.max()) + 1
        )
        novelty = 1.0 / (1.0 + counts[categories].float())
        policy_scores = self.reranker.logits(
            users[internal_user], items[candidate_ids], relevance, novelty
        )
        policy_scores = (policy_scores - policy_scores.mean()) / policy_scores.std().clamp_min(1e-6)
        relevance = (relevance - relevance.mean()) / relevance.std().clamp_min(1e-6)
        combined = relevance + self.policy_weight * policy_scores
        order = torch.argsort(combined, descending=True)[:limit]
        return [
            Recommendation(
                item_id=self.split.item_original_ids[int(candidate_ids[index])],
                score=round(float(combined[index]), 6),
                category=int(categories[index]),
                reason=f"LightGCN召回 + Policy重排（权重{self.policy_weight}）",
            )
            for index in order
        ]

