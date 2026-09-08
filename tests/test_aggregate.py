import json
from pathlib import Path

from gnn_rl_recommender.aggregation import aggregate


def test_multiseed_aggregation(tmp_path: Path) -> None:
    paths = []
    for seed, recall in [(1, 0.2), (2, 0.4)]:
        payload = {
            "popularity": {name: 0.1 for name in ["recall_at_k", "ndcg_at_k", "coverage_at_k", "diversity_at_k"]},
            "lightgcn": {name: recall for name in ["recall_at_k", "ndcg_at_k", "coverage_at_k", "diversity_at_k"]},
            "policy_weight_sweep": {"0.2": {name: recall for name in ["recall_at_k", "ndcg_at_k", "coverage_at_k", "diversity_at_k"]}},
            "training": {"seed": seed},
        }
        path = tmp_path / f"seed_{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    result = aggregate(paths, "0.2")
    assert result["seeds"] == [1, 2]
    assert result["summary"]["lightgcn"]["recall_at_k_mean"] == 0.3
    assert result["summary"]["lightgcn"]["recall_at_k_std"] == 0.141421
