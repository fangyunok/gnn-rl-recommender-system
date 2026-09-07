from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import InteractionData, make_synthetic_data, normalized_bipartite_graph
from .lightgcn import LightGCN
from .reranker import PolicyReranker


@dataclass
class Recommendation:
    item_id: int
    score: float
    category: int
    reason: str


class RecommenderService:
    dataset_name = "synthetic"

    def __init__(self, artifact_dir: str = "artifacts"):
        self.data: InteractionData = make_synthetic_data()
        self.graph = normalized_bipartite_graph(self.data)
        self.gnn = LightGCN(self.data.num_users, self.data.num_items)
        self.reranker = PolicyReranker()
        self.loaded_from_artifact = False
        checkpoint = Path(artifact_dir) / "model.pt"
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.gnn.load_state_dict(payload["gnn"])
            self.reranker.load_state_dict(payload["reranker"])
            self.loaded_from_artifact = True
        self.gnn.eval()
        self.reranker.eval()

    @torch.no_grad()
    def recommend(self, user_id: int, limit: int = 10) -> list[Recommendation]:
        if not 0 <= user_id < self.data.num_users:
            raise ValueError(f"user_id must be between 0 and {self.data.num_users - 1}")
        relevance, candidate_ids = self.gnn.candidates(self.graph, user_id, max(limit * 3, 20))
        users, items = self.gnn.propagate(self.graph)
        categories = self.data.item_categories[candidate_ids]
        frequency = torch.bincount(categories, minlength=int(self.data.item_categories.max()) + 1)
        novelty = 1.0 / (1.0 + frequency[categories].float())
        logits = self.reranker.logits(users[user_id], items[candidate_ids], relevance, novelty)
        # An untrained demo checkpoint remains deterministic and rewards both signals.
        combined = logits + relevance + 0.2 * novelty
        order = torch.argsort(combined, descending=True)[:limit]
        return [
            Recommendation(
                item_id=int(candidate_ids[i]),
                score=round(float(combined[i]), 6),
                category=int(categories[i]),
                reason="LightGCN相关性召回 + 强化学习策略重排",
            )
            for i in order
        ]
