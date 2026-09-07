from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from gnn_rl_recommender.data import make_synthetic_data, normalized_bipartite_graph
from gnn_rl_recommender.lightgcn import LightGCN
from gnn_rl_recommender.reranker import PolicyReranker


def train(output_dir: Path, epochs: int, seed: int) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = make_synthetic_data(seed=seed)
    graph = normalized_bipartite_graph(data)
    gnn = LightGCN(data.num_users, data.num_items)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=2e-2)
    positive_pairs = set(zip(data.user_ids.tolist(), data.item_ids.tolist()))
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    for _ in range(epochs):
        negatives = torch.randint(0, data.num_items, data.item_ids.shape, generator=generator)
        for index, (user, negative) in enumerate(zip(data.user_ids.tolist(), negatives.tolist())):
            while (user, negative) in positive_pairs:
                negative = int(torch.randint(0, data.num_items, (1,), generator=generator))
            negatives[index] = negative
        optimizer.zero_grad()
        loss = gnn.bpr_loss(graph, data.user_ids, data.item_ids, negatives)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    reranker = PolicyReranker()
    rl_optimizer = torch.optim.Adam(reranker.parameters(), lr=5e-3)
    users, items = gnn.propagate(graph)
    for _ in range(max(epochs // 2, 1)):
        user_id = random.randrange(data.num_users)
        relevance, candidate_ids = gnn.candidates(graph, user_id, 20)
        categories = data.item_categories[candidate_ids]
        novelty = 1.0 / (1.0 + torch.bincount(categories, minlength=8)[categories].float())
        rewards = torch.sigmoid(relevance.detach()) + 0.2 * novelty
        logits = reranker.logits(users[user_id].detach(), items[candidate_ids].detach(), relevance.detach(), novelty)
        loss = reranker.reinforce_loss(logits, rewards)
        rl_optimizer.zero_grad()
        loss.backward()
        rl_optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"gnn": gnn.state_dict(), "reranker": reranker.state_dict()}, output_dir / "model.pt")
    metrics = {"final_bpr_loss": round(final_loss, 6), "epochs": epochs, "seed": seed}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output_dir, args.epochs, args.seed), indent=2))
