from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def deploy(repo_id: str, project_dir: Path) -> str:
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=project_dir,
        commit_message="Deploy real-model GNN-RL recommender",
        ignore_patterns=[
            ".git/**",
            ".venv/**",
            ".tools/**",
            ".cache/**",
            "data/raw/**",
            "artifacts/hard_*/**",
            "artifacts/movielens_model*.pt",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
        ],
    )
    return f"https://huggingface.co/spaces/{repo_id}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face namespace/space-name")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(deploy(args.repo_id, args.project_dir))

