from pathlib import Path

import torch

from gnn_rl_recommender.movielens_service import MovieLensRecommenderService
from gnn_rl_recommender.reranker import PolicyReranker


def test_bundle_serving_without_raw_dataset(tmp_path: Path) -> None:
    policy = PolicyReranker(embedding_dim=32)
    bundle = {
        "user_embeddings": torch.ones(2, 32),
        "item_embeddings": torch.arange(128, dtype=torch.float32).reshape(4, 32) / 128,
        "reranker": policy.state_dict(),
        "item_categories": torch.tensor([0, 1, 0, 1]),
        "user_original_ids": torch.tensor([10, 20]),
        "item_original_ids": torch.tensor([100, 200, 300, 400]),
        "seen_values": torch.tensor([0, 1]),
        "seen_offsets": torch.tensor([0, 1, 2]),
    }
    torch.save(bundle, tmp_path / "serving_bundle.pt")
    service = MovieLensRecommenderService(
        artifact_dir=str(tmp_path), dataset_dir=str(tmp_path / "does-not-exist"), candidate_k=3
    )
    recommendations = service.recommend(10, 2)
    assert service.uses_serving_bundle
    assert len(recommendations) == 2
    assert all(item.item_id != 100 for item in recommendations)

