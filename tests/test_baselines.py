import torch

from gnn_rl_recommender.baselines import build_popularity_rankings
from gnn_rl_recommender.data import InteractionData
from gnn_rl_recommender.movielens import MovieLensSplit


def test_popularity_baseline_filters_seen_items() -> None:
    data = InteractionData(
        num_users=2,
        num_items=4,
        user_ids=torch.tensor([0, 0, 1, 1, 1]),
        item_ids=torch.tensor([0, 1, 0, 0, 2]),
        item_categories=torch.tensor([0, 0, 1, 1]),
    )
    split = MovieLensSplit(
        train=data,
        train_seen={0: {0, 1}, 1: {0, 2}},
        validation_item={0: 2, 1: 1},
        test_item={0: 3, 1: 3},
        user_original_ids=[1, 2],
        item_original_ids=[10, 11, 12, 13],
    )
    rankings = build_popularity_rankings(split, [0, 1], top_k=2)
    assert all(item not in split.train_seen[user] for user, items in rankings.items() for item in items)
    assert rankings[0][0] == 2
    assert rankings[1][0] == 1
