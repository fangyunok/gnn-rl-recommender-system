param(
    [ValidateSet("synthetic", "movielens")]
    [string]$Dataset = "synthetic",
    [int]$Port = 8000
)

$env:DATASET = $Dataset
$env:MODEL_DIR = if ($env:MODEL_DIR) { $env:MODEL_DIR } else { "artifacts" }
$env:MOVIELENS_DIR = if ($env:MOVIELENS_DIR) { $env:MOVIELENS_DIR } else { "data/raw/ml-1m" }
$env:POLICY_WEIGHT = if ($env:POLICY_WEIGHT) { $env:POLICY_WEIGHT } else { "0.1" }

python -m uvicorn gnn_rl_recommender.app:app --host 0.0.0.0 --port $Port
