from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
        bundle_path = Path(artifact_dir) / "serving_bundle.pt"
        if bundle_path.exists():
            self._load_bundle(bundle_path, policy_weight, candidate_k)
            return
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
        with torch.no_grad():
            self.user_embeddings, self.item_embeddings = self.gnn.propagate(self.graph)
        self.policy_weight = policy_weight
        self.candidate_k = candidate_k
        self.loaded_from_artifact = True
        self.user_id_map = {
            original: internal for internal, original in enumerate(self.split.user_original_ids)
        }

    def _load_bundle(self, path: Path, policy_weight: float, candidate_k: int) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.user_embeddings = payload["user_embeddings"]
        self.item_embeddings = payload["item_embeddings"]
        self.reranker = PolicyReranker()
        self.reranker.load_state_dict(payload["reranker"])
        self.reranker.eval()
        self.item_categories = payload["item_categories"]
        self.item_original_ids = payload["item_original_ids"].tolist()
        original_users = payload["user_original_ids"].tolist()
        self.user_id_map = {original: internal for internal, original in enumerate(original_users)}
        values = payload["seen_values"]
        offsets = payload["seen_offsets"]
        self.train_seen = {
            user: set(values[offsets[user] : offsets[user + 1]].tolist())
            for user in range(len(original_users))
        }
        self.data = SimpleNamespace(
            num_users=len(original_users), num_items=len(self.item_original_ids)
        )
        self.policy_weight = policy_weight
        self.candidate_k = candidate_k
        self.loaded_from_artifact = True
        self.uses_serving_bundle = True

    @torch.no_grad()
    def recommend(self, user_id: int, limit: int = 10) -> list[Recommendation]:
        if user_id not in self.user_id_map:
            raise ValueError("unknown MovieLens user_id")
        internal_user = self.user_id_map[user_id]
        raw_relevance = self.item_embeddings @ self.user_embeddings[internal_user]
        train_seen = self.train_seen if hasattr(self, "train_seen") else self.split.train_seen
        raw_relevance[list(train_seen[internal_user])] = -torch.inf
        relevance, candidate_ids = torch.topk(
            raw_relevance, min(self.candidate_k, self.data.num_items)
        )
        item_categories = (
            self.item_categories if hasattr(self, "item_categories") else self.data.item_categories
        )
        categories = item_categories[candidate_ids]
        counts = torch.bincount(
            categories, minlength=int(item_categories.max()) + 1
        )
        novelty = 1.0 / (1.0 + counts[categories].float())
        policy_scores = self.reranker.logits(
            self.user_embeddings[internal_user],
            self.item_embeddings[candidate_ids],
            relevance,
            novelty,
        )
        policy_scores = (policy_scores - policy_scores.mean()) / policy_scores.std().clamp_min(1e-6)
        relevance = (relevance - relevance.mean()) / relevance.std().clamp_min(1e-6)
        combined = relevance + self.policy_weight * policy_scores
        order = torch.argsort(combined, descending=True)[:limit]
        return [
            Recommendation(
                item_id=(
                    self.item_original_ids
                    if hasattr(self, "item_original_ids")
                    else self.split.item_original_ids
                )[int(candidate_ids[index])],
                score=round(float(combined[index]), 6),
                category=int(categories[index]),
                reason=f"LightGCN召回 + Policy重排（权重{self.policy_weight}）",
            )
            for index in order
        ]
