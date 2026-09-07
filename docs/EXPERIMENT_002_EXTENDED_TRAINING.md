# 实验 002：增量训练与 Popularity 对照

## 问题

实验 001 中个性化模型的 Recall@10 略低于 Popularity。实验 002 保留 15 epoch 检查点，继续训练
15 epoch，并把每轮 Policy 训练用户从 1,000 增至 2,000，以判断训练不足是否是主要原因。

## 结果（全量 6,040 用户）

| 方法 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|
| Popularity | **0.035099** | 0.017322 | 0.053157 | 0.590430 |
| 30 epoch LightGCN | 0.033940 | 0.016879 | 0.061522 | 0.617079 |
| LightGCN + Policy（权重 0.1） | **0.035099** | **0.017572** | **0.061522** | **0.669367** |

策略重排在 Recall 与 Popularity 持平的情况下：

- NDCG@10 相对提高约 1.44%；
- Coverage@10 相对提高约 15.74%；
- Diversity@10 相对提高约 13.37%。

## 诊断

LightGCN 的 BPR loss 从 0.561323 降至 0.352942，但 Recall 没有同步改善。这说明继续拟合当前
pairwise objective 不一定提高时间留出的 Top-K 泛化指标，潜在原因包括均匀负采样过于简单、流行度
偏置、时间漂移，以及训练目标与最终排序指标不一致。

Policy 权重 0.1 是本轮更合理的工作点。权重增至 0.5 时 Diversity 达到 0.810085，但 Recall
降至 0.034106，因此不选择该配置。

## 下一步

1. 增加 popularity-aware hard negative sampling。
2. 增加 BPR-MF 消融，区分图传播与矩阵分解的贡献。
3. 使用多个随机种子报告均值和标准差。
4. 将主类型距离升级为多标签 Jaccard diversity。

