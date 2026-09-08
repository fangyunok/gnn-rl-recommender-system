from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gnn_rl_recommender.data import normalized_bipartite_graph
from gnn_rl_recommender.lightgcn import LightGCN
from gnn_rl_recommender.movielens import load_movielens_1m


def export_bundle(checkpoint: Path, dataset_dir: Path, output: Path) -> None:
    split = load_movielens_1m(dataset_dir)
    graph = normalized_bipartite_graph(split.train)
    model = LightGCN(split.train.num_users, split.train.num_items, dim=32, layers=2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["gnn"])
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(graph)

    seen_values: list[int] = []
    seen_offsets = [0]
    for user in range(split.train.num_users):
        seen_values.extend(sorted(split.train_seen[user]))
        seen_offsets.append(len(seen_values))

    bundle = {
        "user_embeddings": user_embeddings,
        "item_embeddings": item_embeddings,
        "reranker": payload["reranker"],
        "item_categories": split.train.item_categories,
        "user_original_ids": torch.tensor(split.user_original_ids, dtype=torch.long),
        "item_original_ids": torch.tensor(split.item_original_ids, dtype=torch.long),
        "seen_values": torch.tensor(seen_values, dtype=torch.long),
        "seen_offsets": torch.tensor(seen_offsets, dtype=torch.long),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)
    print(f"Exported {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/movielens_model.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/serving_bundle.pt"))
    args = parser.parse_args()
    export_bundle(args.checkpoint, args.dataset_dir, args.output)

