# 系统架构

## 离线链路

```text
MovieLens ratings.dat
        |
        v
按用户时间切分 -----> train：构图 / validation：策略奖励 / test：最终评估
        |
        v
用户-物品二部图 -> 归一化邻接矩阵 -> LightGCN 两层传播 -> BPR Loss
                                                |
                                                v
                                  Top-100 候选及相关性特征
                                                |
                           用户向量 + 物品向量 + 相关性 + 新颖性
                                                |
                                                v
                                      Policy Gradient 重排
                                                |
                                                v
                         Recall / NDCG / Coverage / Diversity
```

训练输出包含 GNN 与 Policy 的 `state_dict`。原始数据和权重不进入 Git，仓库只保留下载、训练、
评估代码以及机器可读指标，避免大文件和数据许可问题。

## 在线链路

```text
服务启动 -> 读取数据映射和 checkpoint -> 一次性执行 LightGCN 传播 -> 缓存 embedding

GET /v1/recommendations/{user_id}
        -> 原始 ID 映射
        -> user embedding 与全部 item embedding 点积
        -> 过滤历史物品并取 Top-100
        -> Policy 权重 0.1 重排
        -> Top-10 原始电影 ID + 分数 + 推荐原因
```

整图传播只发生在服务启动阶段，请求阶段不重复运行 GNN，单机即可承载完整推理链路。向更大规模线上系统
演进时，可将 item embedding 放入向量索引，将 user embedding 与历史过滤下沉到低延迟特征服务。

## 关键设计选择

- LightGCN：删除特征变换和激活，仅保留协同过滤需要的邻域聚合，结构简洁且便于消融。
- BPR：隐式反馈下优化正物品分数高于采样负物品。
- Policy Gradient：直接使用奖励优化重排策略，适合组合相关性与多样性等不可微业务目标。
- 时间切分：模拟用过去预测未来，防止随机切分造成未来信息泄漏。
- 权重扫描：通过 Pareto trade-off 在相关性与多样性之间选择工作点，而非默认 RL 权重越高越好。

## 演进路线

- **向量召回**：将全量矩阵乘法替换为 ANN 索引，支撑百万级物品的低延迟候选生成。
- **模型热更新**：以双缓冲或版本化方案加载新权重，实现服务不中断的模型发布。
- **在线闭环**：补齐在线特征一致性、A/B 实验、漂移监控、反馈延迟建模与冷启动策略。
- **奖励校准**：用线上真实反馈校准 Policy 的代理奖励，持续逼近长期收益目标。

