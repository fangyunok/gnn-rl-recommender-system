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

> 当前版本是工程基线，合成数据结果不代表真实线上收益。下一阶段将接入 MovieLens-1M，增加
> Recall@K、NDCG@K、Coverage、Diversity 离线对照实验，并升级为序列决策环境。

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

