from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import InteractionData


@dataclass
class MovieLensSplit:
    train: InteractionData
    train_seen: dict[int, set[int]]
    validation_item: dict[int, int]
    test_item: dict[int, int]
    user_original_ids: list[int]
    item_original_ids: list[int]


def load_movielens_1m(dataset_dir: Path) -> MovieLensSplit:
    """Load MovieLens-1M and leave each user's final two events out by timestamp."""
    ratings_path = dataset_dir / "ratings.dat"
    movies_path = dataset_dir / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            f"MovieLens-1M not found under {dataset_dir}. Run scripts/download_movielens.py first."
        )

    raw_ratings: list[tuple[int, int, int]] = []
    with ratings_path.open(encoding="latin-1") as handle:
        for line in handle:
            user_id, item_id, _rating, timestamp = line.rstrip().split("::")
            raw_ratings.append((int(user_id), int(item_id), int(timestamp)))

    original_users = sorted({row[0] for row in raw_ratings})
    original_items = sorted({row[1] for row in raw_ratings})
    user_map = {value: index for index, value in enumerate(original_users)}
    item_map = {value: index for index, value in enumerate(original_items)}

    genre_names: dict[int, str] = {}
    with movies_path.open(encoding="latin-1") as handle:
        for line in handle:
            item_id, _title, genres = line.rstrip().split("::")
            genre_names[int(item_id)] = genres.split("|")[0]
    genres = sorted(set(genre_names.values()))
    genre_map = {value: index for index, value in enumerate(genres)}
    item_categories = torch.tensor(
        [genre_map[genre_names[item_id]] for item_id in original_items], dtype=torch.long
    )

    histories: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(original_users))}
    for raw_user, raw_item, timestamp in raw_ratings:
        histories[user_map[raw_user]].append((timestamp, item_map[raw_item]))

    train_users: list[int] = []
    train_items: list[int] = []
    train_seen: dict[int, set[int]] = {}
    validation_item: dict[int, int] = {}
    test_item: dict[int, int] = {}
    for user, history in histories.items():
        ordered = [item for _, item in sorted(history)]
        if len(ordered) < 3:
            continue
        train = ordered[:-2]
        train_seen[user] = set(train)
        validation_item[user] = ordered[-2]
        test_item[user] = ordered[-1]
        train_users.extend([user] * len(train))
        train_items.extend(train)

    interaction_data = InteractionData(
        num_users=len(original_users),
        num_items=len(original_items),
        user_ids=torch.tensor(train_users, dtype=torch.long),
        item_ids=torch.tensor(train_items, dtype=torch.long),
        item_categories=item_categories,
    )
    return MovieLensSplit(
        train=interaction_data,
        train_seen=train_seen,
        validation_item=validation_item,
        test_item=test_item,
        user_original_ids=original_users,
        item_original_ids=original_items,
    )

