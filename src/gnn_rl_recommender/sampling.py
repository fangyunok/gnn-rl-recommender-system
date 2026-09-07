import torch

from .lightgcn import LightGCN
from .movielens import MovieLensSplit


@torch.no_grad()
def sample_negatives(
    split: MovieLensSplit,
    users: torch.Tensor,
    model: LightGCN,
    graph: torch.Tensor,
    generator: torch.Generator,
    strategy: str,
    hard_pool: int,
) -> torch.Tensor:
    """Sample unseen negatives, optionally mining the highest-scoring item from a pool."""
    sample_count = len(users)
    if strategy == "uniform":
        candidates = torch.randint(
            0, split.train.num_items, (sample_count, 1), generator=generator
        )
    else:
        popularity = torch.bincount(
            split.train.item_ids, minlength=split.train.num_items
        ).float()
        probability = popularity.clamp_min(1).pow(0.75)
        probability /= probability.sum()
        pool_size = hard_pool if strategy == "hard" else 1
        candidates = torch.multinomial(
            probability,
            sample_count * pool_size,
            replacement=True,
            generator=generator,
        ).reshape(sample_count, pool_size)

    repeated_users = users.unsqueeze(1).expand_as(candidates)
    invalid = torch.tensor(
        [
            int(item) in split.train_seen[int(user)]
            for user, item in zip(repeated_users.reshape(-1), candidates.reshape(-1))
        ],
        dtype=torch.bool,
    ).reshape_as(candidates)

    if strategy == "hard":
        user_embeddings, item_embeddings = model.propagate(graph)
        scores = (
            user_embeddings[users].unsqueeze(1) * item_embeddings[candidates]
        ).sum(dim=-1)
        scores[invalid] = -torch.inf
        negatives = candidates.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
    else:
        negatives = candidates.squeeze(1)

    chosen_invalid = torch.tensor(
        [
            int(item) in split.train_seen[int(user)]
            for user, item in zip(users, negatives)
        ],
        dtype=torch.bool,
    )
    while chosen_invalid.any():
        replacements = torch.randint(
            0, split.train.num_items, (int(chosen_invalid.sum()),), generator=generator
        )
        negatives[chosen_invalid] = replacements
        chosen_invalid = torch.tensor(
            [
                int(item) in split.train_seen[int(user)]
                for user, item in zip(users, negatives)
            ],
            dtype=torch.bool,
        )
    return negatives

