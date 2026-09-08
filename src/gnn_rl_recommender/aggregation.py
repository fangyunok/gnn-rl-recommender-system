from __future__ import annotations

import json
from pathlib import Path

import numpy as np

METRICS = ("recall_at_k", "ndcg_at_k", "coverage_at_k", "diversity_at_k")


def aggregate(paths: list[Path], policy_weight: str) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    methods = {
        "popularity": [run["popularity"] for run in runs],
        "lightgcn": [run["lightgcn"] for run in runs],
        "lightgcn_plus_policy": [run["policy_weight_sweep"][policy_weight] for run in runs],
    }
    summary: dict[str, dict[str, float]] = {}
    for method, values in methods.items():
        summary[method] = {}
        for metric in METRICS:
            samples = np.array([value[metric] for value in values], dtype=float)
            summary[method][f"{metric}_mean"] = round(float(samples.mean()), 6)
            summary[method][f"{metric}_std"] = round(float(samples.std(ddof=1)), 6)
    return {
        "seeds": [run["training"]["seed"] for run in runs],
        "number_of_runs": len(runs),
        "policy_weight": float(policy_weight),
        "summary": summary,
        "source_files": [str(path) for path in paths],
    }

