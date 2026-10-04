# GNN + RL 个性化推荐系统

[![CI](https://github.com/fangyunok/gnn-rl-recommender-system/actions/workflows/ci.yml/badge.svg)](https://github.com/fangyunok/gnn-rl-recommender-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Image-ghcr.io-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**召回 → 重排 → 服务** 一体化的端到端个性化推荐系统：**LightGCN** 学习用户—物品二部图表示完成候选召回，**Policy Gradient** 策略在相关性与多样性之间做重排，FastAPI、Docker 与 GitHub Actions 串起可复现的训练—评测—服务闭环。

> 生产镜像：`ghcr.io/fangyunok/gnn-rl-recommender-system:latest`

---

## 关键结果

在 MovieLens-1M（6,040 名用户、约 100 万条评分）按用户时间顺序 leave-two-out 切分下，三随机种子的全量评估结果：

| 问题 | 结论 | 证据 |
|---|---|---|
| 图协同过滤能否超越热门推荐 | Recall@10 由 `0.0351` 提升至 `0.0462`，NDCG@10 由 `0.0173` 提升至 `0.0233` | MovieLens-1M 全量用户、seed 7/42/2026 |
| 图传播是否带来增益 | 相对同预算 BPR-MF，Recall@10 提升约 64.0%、NDCG@10 提升约 79.6% | LightGCN 层数 0 vs 2 消融 |
| RL 重排是否只牺牲相关性换多样性 | Diversity@10 由 `0.5496` 提升至 `0.6034`，Recall/NDCG 基本持平 | 相同候选集上的策略对照 |
| 结果是否稳定 | 三次独立运行均报告 mean±std，并保留全部种子产物 | [实验 005](docs/EXPERIMENT_005_MULTISEED.md) |
| 能否一键运行 | 内置真实模型 serving bundle、FastAPI 接口、Docker 镜像与 CI | [部署手册](docs/DEPLOYMENT.md) |

## 系统架构

```text
隐式反馈 ──▶ 用户-物品二部图 ──▶ LightGCN / BPR ──▶ Top-K 候选
                                                      │
用户状态 + 候选相关性 + 新颖性 ─────────────────────▶ RL Policy ──▶ 推荐列表 ──▶ REST API
```

离线链路与在线链路的分层设计见 [架构说明](docs/ARCHITECTURE.md)。

## 核心能力

- **图召回**：LightGCN 多层消息传播 + BPR pairwise ranking loss，两层传播、32 维 embedding。
- **困难负采样**：按物品流行度 `0.75` 次幂构造候选池，用当前模型分数挑选最难负样本。
- **策略重排**：REINFORCE 重排策略，将相关性与类目新颖性组成长期奖励，按 Pareto trade-off 选择权重。
- **严格防泄漏**：按用户时间顺序 leave-two-out，训练图 / 策略奖励 / 最终测试三者隔离。
- **四类评测指标**：Recall@K、NDCG@K、Catalog Coverage、Intra-list Diversity。
- **在线服务**：`GET /health` 与 `GET /v1/recommendations/{user_id}`，启动阶段一次性传播并缓存 embedding。
- **双数据集模式**：`synthetic` 轻量冒烟模式与 `movielens` 真实权重模式一键切换。
- **工程闭环**：pytest 自动化测试、Ruff 静态检查、Docker Compose 与 GitHub Actions。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

pytest -q                          # 运行自动化测试
python scripts\train.py --epochs 5 # 合成数据快速训练
uvicorn gnn_rl_recommender.app:app --reload
```

打开 `http://127.0.0.1:8000/docs`，或直接请求：

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/v1/recommendations/3?limit=5'
```

Docker 默认加载内置真实模型 bundle，无需下载原始数据即可启动：

```bash
docker compose up --build -d
curl "http://127.0.0.1:8000/v1/recommendations/3?limit=5"
```

## 实验结果

### 三随机种子结果（实验 005，6,040 名用户）

| 方法 | Recall@10（mean±std） | NDCG@10（mean±std） | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.035099±0 | 0.017322±0 | 0.053157±0 | 0.590430±0 |
| Hard-negative LightGCN | 0.046082±0.001755 | 0.023175±0.001053 | 0.337291±0.015226 | 0.549648±0.010757 |
| LightGCN + Policy（权重 0.2） | **0.046247±0.002413** | **0.023318±0.001141** | **0.336391±0.016019** | **0.603379±0.014963** |

相对 Popularity，最终模型平均 Recall@10 提升约 31.8%、平均 NDCG@10 提升约 34.6%。

### 实验演进

| 实验 | 主题 | 报告 |
|---|---|---|
| 001 | LightGCN + Policy 端到端闭环与权重扫描 | [报告](docs/EXPERIMENT_001_MOVIELENS.md) |
| 002 | 增量训练与 Popularity 对照 | [报告](docs/EXPERIMENT_002_EXTENDED_TRAINING.md) |
| 003 | BPR-MF 消融：图传播的贡献 | [报告](docs/EXPERIMENT_003_BPR_MF_ABLATION.md) |
| 004 | Hard Negative Mining 带来的提升 | [报告](docs/EXPERIMENT_004_HARD_NEGATIVE.md) |
| 005 | 三随机种子稳定性 | [报告](docs/EXPERIMENT_005_MULTISEED.md) |

原始机器可读指标见 `results/*.json`。

## MovieLens-1M 真实数据实验

```powershell
python scripts\download_movielens.py
python scripts\experiment_movielens.py --gnn-epochs 10 --rl-epochs 3 --eval-users 1000
```

数据按每位用户的时间戳排序：倒数第二次交互用于训练策略，最后一次交互仅用于测试。输出写入 `artifacts/movielens_metrics.json`。

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

## 项目结构

```text
src/gnn_rl_recommender/  # 数据、LightGCN、策略重排与 API
scripts/                 # 训练、下载、导出与实验入口
tests/                   # 12 个自动化测试（API / 模型 / 指标 / 采样 / 聚合）
artifacts/               # 本地模型与 serving bundle
docs/                    # 架构、部署、实验报告与面试指南
.github/workflows/       # CI 静态检查、测试与 Docker 构建验证
```

## 部署与 CI

- 本地、Docker 与云服务器三种部署方式见 [部署手册](docs/DEPLOYMENT.md)。
- GitHub Actions 在每次 push 与 PR 时运行静态检查、单元测试，并构建 Docker 镜像做真实模型冒烟验证，确保仓库始终处于可部署状态。
- 生产镜像固定使用 CPU-only PyTorch，GPU 训练环境与 CPU 推理镜像分离。

## Roadmap

- **规模化召回**：将 item embedding 写入 ANN 向量索引，支撑百万级物品的低延迟召回。
- **模型热更新**：以版本化双缓冲方式加载新权重，实现无中断的模型发布。
- **在线实验体系**：接入 A/B 实验与漂移监控，用真实曝光/反馈数据校准离线策略奖励。
- **多样性指标升级**：以多标签 Jaccard 距离替代主类型不相等衡量。
- **超参自动化**：对 hard pool、流行度指数与策略权重进行系统化网格搜索。

## License

MIT
