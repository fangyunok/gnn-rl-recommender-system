# GNN + RL 个性化推荐系统

[![CI](https://github.com/fangyunok/gnn-rl-recommender-system/actions/workflows/ci.yml/badge.svg)](https://github.com/fangyunok/gnn-rl-recommender-system/actions/workflows/ci.yml)

面向推荐算法与强化学习岗位的端到端项目：使用 **LightGCN** 学习用户—物品二部图表示，
完成候选召回；使用 **Policy Gradient** 策略同时优化相关性与多样性，完成候选重排；最终通过
FastAPI、Docker 与 GitHub Actions 形成可复现的训练—评测—服务闭环。

## 系统流程

```text
隐式反馈 -> 用户-物品二部图 -> LightGCN/BPR -> Top-K 候选
                                             |
用户状态 + 候选相关性 + 新颖性 ------------> RL Policy -> 推荐列表 -> REST API
```

## 当前能力

- 可复现的合成隐式反馈数据，便于任何机器快速跑通闭环。
- LightGCN 多层消息传播与 BPR pairwise ranking loss。
- REINFORCE 重排策略，将相关性和类目新颖性组成长期奖励的最小实现。
- `GET /health` 与 `GET /v1/recommendations/{user_id}` 在线接口。
- pytest、Ruff、Docker Compose 和 GitHub Actions。
- MovieLens-1M 按用户时间顺序 leave-two-out 切分与真实离线实验入口。
- Recall@K、NDCG@K、Catalog Coverage、Intra-list Diversity 四类指标。

> 合成数据仅用于快速验证工程链路；正式指标来自 MovieLens-1M。离线结果不等同于真实线上收益。

### 实验 001 结果（6,040 名用户）

| 方法 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|
| LightGCN | 0.034272 | 0.016964 | 0.066919 | 0.611093 |
| LightGCN + Policy（权重 0.2） | **0.034603** | **0.017536** | 0.066919 | **0.695695** |

完整配置、权重扫描和局限性见 [实验 001 报告](docs/EXPERIMENT_001_MOVIELENS.md)。

## MovieLens-1M 真实数据实验

```powershell
python scripts\download_movielens.py
python scripts\experiment_movielens.py --gnn-epochs 10 --rl-epochs 3 --eval-users 1000
```

数据按每位用户的时间戳排序：倒数第二次交互只用于训练策略，最后一次交互只用于测试。
输出写入 `artifacts/movielens_metrics.json`，模型权重不会提交到 GitHub。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts\train.py --epochs 30
pytest -q
uvicorn gnn_rl_recommender.app:app --reload
```

打开 `http://127.0.0.1:8000/docs`，或请求：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/v1/recommendations/3?limit=5'
```

Docker 与云服务器部署见 [部署手册](docs/DEPLOYMENT.md)。

## 仓库结构

```text
src/gnn_rl_recommender/  # 数据、LightGCN、策略重排与 API
scripts/train.py         # 可复现训练入口
tests/                   # API 自动化测试
artifacts/               # 本地模型与指标（不提交权重）
docs/DEPLOYMENT.md       # 完整部署流程
.github/workflows/       # CI 与 Docker 构建验证
```

## API 示例

`GET /v1/recommendations/3?limit=5`

```json
{
  "user_id": 3,
  "items": [{"item_id": 27, "score": 0.42, "category": 3,
    "reason": "LightGCN相关性召回 + 强化学习策略重排"}],
  "model_version": "lightgcn-policy-v1"
}
```

## 技术边界

- 合成数据仅用于验证工程链路，不用于宣称业务指标。
- 服务无模型文件时使用固定随机初始化完成演示；训练后自动加载 `artifacts/model.pt`。
- 真实生产系统还需要特征平台、在线召回、实验平台、模型监控和数据漂移治理。
