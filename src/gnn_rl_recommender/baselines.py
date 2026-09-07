import torch

from .movielens import MovieLensSplit


def build_popularity_rankings(
    split: MovieLensSplit, users: list[int], top_k: int
) -> dict[int, list[int]]:
    """Recommend globally popular unseen items as a non-personalized baseline."""
    counts = torch.bincount(split.train.item_ids, minlength=split.train.num_items)
    popular_items = torch.argsort(counts, descending=True).tolist()
    rankings: dict[int, list[int]] = {}
    for user in users:
        seen = split.train_seen[user]
        rankings[user] = [item for item in popular_items if item not in seen][:top_k]
    return rankings

