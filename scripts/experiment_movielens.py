from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from gnn_rl_recommender.baselines import build_popularity_rankings
from gnn_rl_recommender.data import normalized_bipartite_graph
from gnn_rl_recommender.lightgcn import LightGCN
from gnn_rl_recommender.metrics import evaluate_rankings
from gnn_rl_recommender.movielens import MovieLensSplit, load_movielens_1m
from gnn_rl_recommender.reranker import PolicyReranker
from gnn_rl_recommender.sampling import sample_negatives


def train_lightgcn(
    split: MovieLensSplit,
    graph: torch.Tensor,
    epochs: int,
    samples: int,
    seed: int,
    model: LightGCN | None = None,
    label: str = "LightGCN",
    negative_sampling: str = "uniform",
    hard_pool: int = 5,
) -> tuple[LightGCN, list[float]]:
    torch.manual_seed(seed)
    if model is None:
        model = LightGCN(split.train.num_users, split.train.num_items, dim=32, layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    interaction_count = len(split.train.user_ids)
    for epoch in range(epochs):
        indices = torch.randint(0, interaction_count, (samples,), generator=generator)
        users = split.train.user_ids[indices]
        positives = split.train.item_ids[indices]
        negatives = sample_negatives(
            split, users, model, graph, generator, negative_sampling, hard_pool
        )
        optimizer.zero_grad()
        loss = model.bpr_loss(graph, users, positives, negatives)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        print(f"{label} epoch {epoch + 1}/{epochs}: loss={losses[-1]:.5f}")
    return model, losses


@torch.no_grad()
def candidate_features(
    split: MovieLensSplit,
    user: int,
    candidate_k: int,
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = item_embeddings @ user_embeddings[user]
    seen = list(split.train_seen[user])
    scores[seen] = -torch.inf
    relevance, ids = torch.topk(scores, min(candidate_k, split.train.num_items - len(seen)))
    categories = split.train.item_categories[ids]
    counts = torch.bincount(categories, minlength=int(split.train.item_categories.max()) + 1)
    novelty = 1.0 / (1.0 + counts[categories].float())
    return user_embeddings[user], item_embeddings[ids], relevance, novelty, ids


def train_policy(
    model: LightGCN,
    policy: PolicyReranker,
    graph: torch.Tensor,
    split: MovieLensSplit,
    epochs: int,
    candidate_k: int,
    max_users: int,
    seed: int,
) -> list[float]:
    random.seed(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)
    users = list(split.validation_item)
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(graph)
        user_embeddings = user_embeddings.detach()
        item_embeddings = item_embeddings.detach()
    losses: list[float] = []
    for epoch in range(epochs):
        random.shuffle(users)
        epoch_losses: list[float] = []
        for user in users[: min(max_users, len(users))]:
            user_emb, item_emb, relevance, novelty, ids = candidate_features(
                split, user, candidate_k, user_embeddings, item_embeddings
            )
            target = split.validation_item[user]
            rewards = 0.15 * novelty
            rewards = rewards + (ids == target).float()
            logits = policy.logits(user_emb, item_emb, relevance, novelty)
            loss = policy.reinforce_loss(logits, rewards)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        print(f"Policy epoch {epoch + 1}/{epochs}: loss={mean_loss:.5f}")
    return losses


@torch.no_grad()
def build_rankings(
    model: LightGCN,
    policy: PolicyReranker | None,
    graph: torch.Tensor,
    split: MovieLensSplit,
    users: list[int],
    top_k: int,
    candidate_k: int,
    policy_weight: float = 1.0,
) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    user_embeddings, item_embeddings = model.propagate(graph)
    for user in users:
        user_emb, item_emb, relevance, novelty, ids = candidate_features(
            split, user, candidate_k, user_embeddings, item_embeddings
        )
        scores = relevance
        if policy is not None:
            policy_scores = policy.logits(user_emb, item_emb, relevance, novelty)
            policy_scores = (policy_scores - policy_scores.mean()) / policy_scores.std().clamp_min(1e-6)
            relevance_normalized = (relevance - relevance.mean()) / relevance.std().clamp_min(1e-6)
            scores = relevance_normalized + policy_weight * policy_scores
        order = torch.argsort(scores, descending=True)[:top_k]
        rankings[user] = ids[order].tolist()
    return rankings


@torch.no_grad()
def build_sweep_rankings(
    model: LightGCN,
    policy: PolicyReranker,
    graph: torch.Tensor,
    split: MovieLensSplit,
    users: list[int],
    top_k: int,
    candidate_k: int,
    weights: list[float],
) -> tuple[dict[int, list[int]], dict[str, dict[int, list[int]]]]:
    """Evaluate all interpolation weights while reusing graph and candidate computation."""
    baseline: dict[int, list[int]] = {}
    sweeps: dict[str, dict[int, list[int]]] = {str(weight): {} for weight in weights}
    user_embeddings, item_embeddings = model.propagate(graph)
    for user in users:
        user_emb, item_emb, relevance, novelty, ids = candidate_features(
            split, user, candidate_k, user_embeddings, item_embeddings
        )
        baseline_order = torch.argsort(relevance, descending=True)[:top_k]
        baseline[user] = ids[baseline_order].tolist()
        policy_scores = policy.logits(user_emb, item_emb, relevance, novelty)
        policy_scores = (policy_scores - policy_scores.mean()) / policy_scores.std().clamp_min(1e-6)
        relevance_normalized = (relevance - relevance.mean()) / relevance.std().clamp_min(1e-6)
        for weight in weights:
            order = torch.argsort(
                relevance_normalized + weight * policy_scores, descending=True
            )[:top_k]
            sweeps[str(weight)][user] = ids[order].tolist()
    return baseline, sweeps


def main(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = load_movielens_1m(args.data_dir)
    graph = normalized_bipartite_graph(split.train)
    model = LightGCN(split.train.num_users, split.train.num_items, dim=32, layers=2)
    policy = PolicyReranker()
    if args.load_checkpoint:
        payload = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["gnn"])
        policy.load_state_dict(payload["reranker"])
        gnn_losses = [float("nan")]
        policy_losses = [float("nan")]
    else:
        if args.resume_checkpoint:
            payload = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["gnn"])
        model, gnn_losses = train_lightgcn(
            split,
            graph,
            args.gnn_epochs,
            args.samples_per_epoch,
            args.seed,
            model,
            negative_sampling=args.negative_sampling,
            hard_pool=args.hard_pool,
        )
        policy_losses = train_policy(
            model,
            policy,
            graph,
            split,
            args.rl_epochs,
            args.candidate_k,
            args.rl_users,
            args.seed,
        )
    evaluation_users = sorted(split.test_item)[: args.eval_users]
    baseline, sweep_rankings = build_sweep_rankings(
        model,
        policy,
        graph,
        split,
        evaluation_users,
        args.top_k,
        args.candidate_k,
        args.policy_weights,
    )
    categories = split.train.item_categories.tolist()
    popularity = build_popularity_rankings(split, evaluation_users, args.top_k)
    mf_metrics = None
    if args.mf_epochs > 0:
        mf = LightGCN(split.train.num_users, split.train.num_items, dim=32, layers=0)
        mf, _ = train_lightgcn(
            split,
            graph,
            args.mf_epochs,
            args.samples_per_epoch,
            args.seed + 100,
            mf,
            "BPR-MF",
        )
        mf_rankings = build_rankings(
            mf, None, graph, split, evaluation_users, args.top_k, args.candidate_k
        )
        mf_metrics = evaluate_rankings(
            mf_rankings, split.test_item, categories, split.train.num_items
        ).as_dict()
    policy_sweep = {}
    for weight in args.policy_weights:
        policy_sweep[str(weight)] = evaluate_rankings(
            sweep_rankings[str(weight)], split.test_item, categories, split.train.num_items
        ).as_dict()
    results = {
        "dataset": "MovieLens-1M",
        "split": "per-user chronological leave-two-out",
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "popularity": evaluate_rankings(
            popularity, split.test_item, categories, split.train.num_items
        ).as_dict(),
        "bpr_mf": mf_metrics,
        "lightgcn": evaluate_rankings(
            baseline, split.test_item, categories, split.train.num_items
        ).as_dict(),
        "policy_weight_sweep": policy_sweep,
        "training": {
            "gnn_epochs": args.gnn_epochs + (args.previous_epochs if args.resume_checkpoint else 0),
            "rl_epochs": args.rl_epochs,
            "rl_users_per_epoch": args.rl_users,
            "final_gnn_loss": None if args.load_checkpoint else round(gnn_losses[-1], 6),
            "final_policy_loss": None if args.load_checkpoint else round(policy_losses[-1], 6),
            "seed": args.seed,
            "negative_sampling": args.negative_sampling,
            "hard_pool": args.hard_pool,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gnn": model.state_dict(),
            "reranker": policy.state_dict(),
            "training": results["training"],
        },
        args.output_dir / "movielens_model.pt",
    )
    (args.output_dir / "movielens_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--gnn-epochs", type=int, default=10)
    parser.add_argument("--rl-epochs", type=int, default=3)
    parser.add_argument("--mf-epochs", type=int, default=0)
    parser.add_argument("--rl-users", type=int, default=1000)
    parser.add_argument("--samples-per-epoch", type=int, default=100_000)
    parser.add_argument(
        "--negative-sampling", choices=["uniform", "popularity", "hard"], default="uniform"
    )
    parser.add_argument("--hard-pool", type=int, default=5)
    parser.add_argument("--eval-users", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--policy-weights", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--previous-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
