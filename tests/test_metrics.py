from gnn_rl_recommender.metrics import evaluate_rankings


def test_ranking_metrics_exact_values() -> None:
    metrics = evaluate_rankings(
        rankings={0: [1, 2], 1: [3, 4]},
        targets={0: 1, 1: 9},
        item_categories=[0, 0, 1, 0, 1, 1, 0, 0, 1, 1],
        num_items=10,
    )
    assert metrics.recall_at_k == 0.5
    assert metrics.ndcg_at_k == 0.5
    assert metrics.coverage_at_k == 0.4
    assert metrics.diversity_at_k == 1.0

