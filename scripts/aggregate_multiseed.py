from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_rl_recommender.aggregation import aggregate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--policy-weight", default="0.2")
    parser.add_argument("--output", type=Path, default=Path("results/multiseed_summary.json"))
    args = parser.parse_args()
    result = aggregate(args.paths, args.policy_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
