# 简历与面试讲解

## 简历表述（建议版）

**基于 LightGCN 与 Policy Gradient 的个性化推荐系统**

- 基于 MovieLens-1M 构建用户—物品二部图，采用按用户时间顺序 leave-two-out 切分，使用
  LightGCN + BPR 完成 Top-100 候选召回，并通过 BPR-MF 消融验证图传播贡献。
- 设计 popularity-aware hard negative mining 与 Policy Gradient 重排策略；三随机种子、全量
  6,040 用户实验中，LightGCN 召回将 Recall@10 由 `0.035099` 提升到 `0.046082±0.001755`，
  NDCG@10 由 `0.017322` 提升到 `0.023175±0.001053`（相对 Popularity 约 `+31%` / `+34%`，
  该提升来自图召回而非策略网络）；策略重排在此基础上把 Diversity@10 由 `0.549648` 提升到
  `0.603379`，Recall/NDCG 变化在噪声范围内。
- 搭建 FastAPI 在线服务，缓存 GNN embedding 降低请求计算开销，并使用 Docker、自动测试和
  GitHub Actions 完成训练—评估—部署闭环。

**指标口径与贡献归因**：统一使用离线排序指标 Recall@10 与 NDCG@10（报告 mean±std）。在固定
MovieLens 时间切分与三随机种子（seed 7 / 42 / 2026）下，相对 Popularity 基线，最终系统
Recall@10、NDCG@10 平均提升约 `31.8%`、`34.6%`，其中主要贡献来自 LightGCN 图召回
（Popularity → LightGCN 一段）；Policy 重排可验证的收益是把 Diversity@10 由 `0.5496` 提升到
`0.6034`，并在等效相关性下保持 Recall/NDCG 基本不变。

## 两分钟项目介绍

项目解决两个问题：先从大规模物品里找相关候选，再在候选内平衡准确性和多样性。我用
LightGCN 学习用户—物品二部图 embedding，以 BPR loss 训练召回；随后把用户向量、候选向量、
相关性和新颖性输入策略网络，用 REINFORCE 做重排。数据按时间切分，训练图、策略奖励和最终测试
互相隔离。结果上，图传播比相同预算的 BPR-MF 显著提高 Recall/NDCG；在 hard-negative LightGCN
候选上，最终系统相对热门推荐的 Recall@10 / NDCG@10 平均提升约 `31.8%` / `34.6%`，这部分主要
由图召回带来；策略重排本身把 Diversity@10 从 `0.5496` 提到 `0.6034`，Recall/NDCG 基本不变。
最后把模型封装成 FastAPI，在启动阶段缓存图 embedding，并通过 Docker 和 CI 验证部署链路。

## 高频追问

### 为什么使用 LightGCN？

推荐场景的核心信号是用户—物品交互，高阶邻域本身就能传播协同过滤信息。LightGCN 去掉普通
GCN 的变换矩阵和激活，参数少、解释清晰，也便于与零传播层的 BPR-MF 做公平消融。

### RL 的状态、动作和奖励是什么？

- 状态：用户 embedding、候选 item embedding、GNN 相关性和类目新颖性。
- 动作：从候选集合中选择推荐物品。
- 奖励：验证交互命中奖励加新颖性奖励。
- 优化：Categorical policy 上使用 REINFORCE 和 batch baseline 降低方差。

### 为什么测试集不能训练 Policy？

如果用测试交互定义奖励，再在相同交互上报告结果，相当于把答案泄漏给策略。本项目把倒数第二次
交互作为验证反馈，最后一次交互只做最终评估。

### 为什么训练 loss 降低但 Recall 没提高？

BPR 优化采样 pair 的相对分数，不直接等价于全物品 Top-10 Recall。均匀负采样可能产生大量简单
负样本，模型持续降低训练 loss，但对最难混淆候选的排序帮助有限。下一步应尝试 hard negatives、
学习率/正则搜索，并用多随机种子验证。

### 为什么不用 RL 直接召回？

全物品动作空间过大，探索成本和推理成本都很高。工业系统通常先召回缩小空间，再用更复杂策略
精排；本项目在 Top-100 内做 Policy 重排，更符合分阶段架构。

### 当前方案怎么扩展到线上？

离线生成 item embedding 并写入 ANN 索引；user embedding 由特征服务提供；推荐服务执行向量召回、
历史过滤、策略重排，并记录曝光/点击供后续训练。模型使用版本化发布和 A/B 实验验证长期指标。
