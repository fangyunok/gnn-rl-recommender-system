# 实验 003：BPR-MF 消融

## 目的

LightGCN 与传统矩阵分解都可以使用 user/item embedding 和 BPR loss。为验证提升是否来自图消息传播，
本实验将 LightGCN 层数设为 0，得到 BPR-MF 对照；两者保持 32 维 embedding、30 epoch、
每轮 100,000 个训练三元组和相同时间切分。

## 结果

| 方法 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|
| Popularity | **0.035099** | 0.017322 | 0.053157 | 0.590430 |
| BPR-MF | 0.020695 | 0.009396 | **0.198597** | 0.785486 |
| LightGCN | 0.033940 | 0.016879 | 0.061522 | 0.617079 |
| LightGCN + Policy（0.1） | **0.035099** | **0.017572** | 0.061522 | 0.669367 |

相对 BPR-MF，LightGCN 的 Recall@10 提高约 64.0%，NDCG@10 提高约 79.6%，说明在当前设置下
用户—物品图的邻域传播显著改善了 Top-K 相关性。BPR-MF 的 Coverage 和 Diversity 更高，但命中率
明显偏低，体现“推荐得散”不等同于“推荐得准”。

策略重排进一步把 LightGCN 的 Recall 提升到与 Popularity 持平，同时获得高于 Popularity 的
NDCG、Coverage 和 Diversity。

## 边界

- 本实验只控制了核心结构与训练轮数，尚未为 BPR-MF 单独搜索最优学习率和正则强度。
- 单随机种子结果不能代表稳定置信区间；后续需运行多种子实验。
- MF loss 仍在下降，更多训练可能继续改善结果，因此结论限定在当前统一预算下。

