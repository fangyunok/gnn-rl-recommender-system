import torch

from gnn_rl_recommender.data import make_synthetic_data, normalized_bipartite_graph
from gnn_rl_recommender.lightgcn import LightGCN
from gnn_rl_recommender.movielens import MovieLensSplit
from gnn_rl_recommender.sampling import sample_negatives


def test_all_negative_sampling_strategies_filter_seen_items() -> None:
    data = make_synthetic_data(num_users=5, num_items=20, interactions_per_user=4)
    split = MovieLensSplit(
        train=data,
        train_seen={
            user: set(data.item_ids[data.user_ids == user].tolist()) for user in range(5)
        },
        validation_item={},
        test_item={},
        user_original_ids=list(range(5)),
        item_original_ids=list(range(20)),
    )
    graph = normalized_bipartite_graph(data)
    model = LightGCN(5, 20, dim=4, layers=1)
    users = torch.tensor([0, 1, 2, 3, 4] * 4)
    for strategy in ["uniform", "popularity", "hard"]:
        negatives = sample_negatives(
            split,
            users,
            model,
            graph,
            torch.Generator().manual_seed(42),
            strategy,
            hard_pool=3,
        )
        assert all(
            int(item) not in split.train_seen[int(user)]
            for user, item in zip(users, negatives)
        )
