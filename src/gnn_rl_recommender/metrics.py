from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RankingMetrics:
    recall_at_k: float
    ndcg_at_k: float
    coverage_at_k: float
    diversity_at_k: float
    evaluated_users: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "recall_at_k": round(self.recall_at_k, 6),
            "ndcg_at_k": round(self.ndcg_at_k, 6),
            "coverage_at_k": round(self.coverage_at_k, 6),
            "diversity_at_k": round(self.diversity_at_k, 6),
            "evaluated_users": self.evaluated_users,
        }


def evaluate_rankings(
    rankings: dict[int, list[int]],
    targets: dict[int, int],
    item_categories: list[int],
    num_items: int,
) -> RankingMetrics:
    hits: list[float] = []
    ndcgs: list[float] = []
    diversities: list[float] = []
    recommended: set[int] = set()
    for user, ranking in rankings.items():
        target = targets[user]
        hits.append(float(target in ranking))
        ndcgs.append(1.0 / math.log2(ranking.index(target) + 2) if target in ranking else 0.0)
        recommended.update(ranking)
        if len(ranking) < 2:
            diversities.append(0.0)
        else:
            unequal = sum(
                item_categories[ranking[i]] != item_categories[ranking[j]]
                for i in range(len(ranking))
                for j in range(i + 1, len(ranking))
            )
            pairs = len(ranking) * (len(ranking) - 1) / 2
            diversities.append(unequal / pairs)
    return RankingMetrics(
        recall_at_k=float(np.mean(hits)),
        ndcg_at_k=float(np.mean(ndcgs)),
        coverage_at_k=len(recommended) / num_items,
        diversity_at_k=float(np.mean(diversities)),
        evaluated_users=len(rankings),
    )

