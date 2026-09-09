from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gnn_rl_recommender.movielens_service import MovieLensRecommenderService

st.set_page_config(page_title="GNN + RL 推荐系统", page_icon="🎬", layout="wide")


@st.cache_resource
def load_service(policy_weight: float) -> MovieLensRecommenderService:
    return MovieLensRecommenderService(
        artifact_dir=str(PROJECT_ROOT / "artifacts"),
        dataset_dir=str(PROJECT_ROOT / "data" / "raw" / "ml-1m"),
        policy_weight=policy_weight,
    )


st.title("🎬 GNN + RL 个性化推荐系统")
st.caption("MovieLens-1M · Hard-negative LightGCN · Policy Gradient reranking")

metric_columns = st.columns(4)
metric_columns[0].metric("Recall@10", "0.0462 ± 0.0024", "+31.8% vs Popularity")
metric_columns[1].metric("NDCG@10", "0.0233 ± 0.0011", "+34.6% vs Popularity")
metric_columns[2].metric("Coverage@10", "0.3364 ± 0.0160", "+532.8% vs Popularity")
metric_columns[3].metric("Diversity@10", "0.6034 ± 0.0150", "+2.2% vs Popularity")

with st.sidebar:
    st.header("推荐参数")
    user_id = st.number_input("MovieLens 用户 ID", min_value=1, max_value=6040, value=1)
    limit = st.slider("推荐数量", min_value=1, max_value=20, value=10)
    policy_weight = st.slider(
        "Policy 权重", min_value=0.0, max_value=0.5, value=0.2, step=0.05
    )
    st.caption("0 表示只使用 LightGCN；提高权重会增强策略重排影响。")

service = load_service(policy_weight)
recommendations = service.recommend(int(user_id), limit)

st.subheader(f"用户 {int(user_id)} 的 Top-{limit} 推荐")
rows = [
    {
        "排名": rank,
        "电影 ID": recommendation.item_id,
        "综合分数": recommendation.score,
        "主类型编码": recommendation.category,
        "推荐路径": recommendation.reason,
    }
    for rank, recommendation in enumerate(recommendations, start=1)
]
st.dataframe(rows, width="stretch", hide_index=True)

chart_data = {str(row["电影 ID"]): row["综合分数"] for row in rows}
st.bar_chart(chart_data, horizontal=True)

with st.expander("模型与实验说明"):
    st.markdown(
        """
        - LightGCN 在用户—物品二部图上进行两层传播，以 BPR loss 学习召回表示。
        - 每轮从流行度分布抽取负样本池，再选择当前模型最难区分的负物品。
        - Policy 使用用户、物品、相关性与新颖性特征重排 Top-100 候选。
        - 指标来自 seed 7/42/2026、全量 6,040 用户；离线提升不等于线上点击率收益。
        """
    )
