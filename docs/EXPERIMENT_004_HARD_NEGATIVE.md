# 实验 004：Hard Negative Mining

## 动机

均匀负采样会产生大量模型很容易区分的负物品，BPR loss 持续下降不一定改善 Top-K 排序。
本实验先按物品流行度的 0.75 次幂抽取 5 个候选负物品，再用当前 LightGCN 分数选择其中最难的
未交互物品。其余数据切分、embedding 维度和评估口径保持不变。

## 配置

| 参数 | 值 |
|---|---:|
| GNN epoch | 20 |
| 每轮三元组 | 100,000 |
| hard-negative pool | 5 |
| Policy epoch / 每轮用户 | 3 / 2,000 |
| 候选集 / 推荐列表 | 100 / 10 |
| 评估用户 | 6,040（全量） |

## 结果

| 方法 | Recall@10 | NDCG@10 | Coverage@10 | Diversity@10 |
|---|---:|---:|---:|---:|
| Popularity | 0.035099 | 0.017322 | 0.053157 | 0.590430 |
| 均匀负采样 LightGCN + Policy | 0.035099 | 0.017572 | 0.061522 | 0.669367 |
| Hard-negative LightGCN | 0.046358 | 0.023338 | **0.335132** | 0.540670 |
| Hard-negative LightGCN + Policy（0.2） | **0.048179** | **0.023828** | 0.333243 | **0.597112** |

相对 Popularity，最终模型的 Recall@10、NDCG@10 分别提高约 37.27% 和 37.56%；Coverage
从 0.0532 提升至 0.3332，Diversity 也略高于 Popularity。

在 hard-negative LightGCN 内部，Policy 重排使 Recall@10 提高约 3.93%，NDCG@10 提高约
2.10%，Diversity@10 提高约 10.44%，代价是 Coverage 轻微下降约 0.56%。

## 关键解释

Hard-negative 训练的最终 loss 为 0.689950，数值高于均匀负采样，但两者采样分布不同，loss
不能直接横向比较。Hard mining 持续选择当前模型最难区分的物品，因此损失更高意味着训练任务
更难；模型优劣必须由相同测试集上的 Recall/NDCG 判断。

## 局限性

- 当前仍是单随机种子，需要多种子重复实验确认方差。
- hard pool=5 和流行度指数 0.75 尚未网格搜索。
- 训练开销高于均匀负采样，因为每轮需要额外传播 embedding 以计算困难度。
- MovieLens 离线收益不能直接解释为真实线上点击率提升。

