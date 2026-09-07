from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class InteractionData:
    num_users: int
    num_items: int
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    item_categories: torch.Tensor


def make_synthetic_data(
    num_users: int = 64,
    num_items: int = 120,
    num_categories: int = 8,
    interactions_per_user: int = 12,
    seed: int = 42,
) -> InteractionData:
    """Create reproducible implicit feedback with category-level user preferences."""
    rng = np.random.default_rng(seed)
    categories = np.arange(num_items) % num_categories
    users: list[int] = []
    items: list[int] = []
    for user in range(num_users):
        preferred = {user % num_categories, (user * 3 + 1) % num_categories}
        weights = np.array([4.0 if int(c) in preferred else 0.5 for c in categories])
        weights /= weights.sum()
        sampled = rng.choice(
            num_items, size=min(interactions_per_user, num_items), replace=False, p=weights
        )
        users.extend([user] * len(sampled))
        items.extend(sampled.tolist())
    return InteractionData(
        num_users=num_users,
        num_items=num_items,
        user_ids=torch.tensor(users, dtype=torch.long),
        item_ids=torch.tensor(items, dtype=torch.long),
        item_categories=torch.tensor(categories, dtype=torch.long),
    )


def normalized_bipartite_graph(data: InteractionData) -> torch.Tensor:
    """Build D^-1/2 A D^-1/2 for the user-item bipartite graph."""
    users = data.user_ids
    items = data.item_ids + data.num_users
    row = torch.cat([users, items])
    col = torch.cat([items, users])
    degree = torch.bincount(row, minlength=data.num_users + data.num_items).float().clamp_min(1)
    values = degree[row].rsqrt() * degree[col].rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (len(degree), len(degree)),
        check_invariants=False,
    ).coalesce()
