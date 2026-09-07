import torch

from gnn_rl_recommender.data import make_synthetic_data, normalized_bipartite_graph
from gnn_rl_recommender.lightgcn import LightGCN
from gnn_rl_recommender.reranker import PolicyReranker
from gnn_rl_recommender.service import RecommenderService


def test_lightgcn_shapes_and_finite_loss() -> None:
    data = make_synthetic_data(num_users=4, num_items=8, interactions_per_user=3)
    graph = normalized_bipartite_graph(data)
    model = LightGCN(data.num_users, data.num_items, dim=6, layers=2)
    users, items = model.propagate(graph)
    assert users.shape == (4, 6)
    assert items.shape == (8, 6)
    negatives = (data.item_ids + 1) % data.num_items
    loss = model.bpr_loss(graph, data.user_ids, data.item_ids, negatives)
    assert torch.isfinite(loss)


def test_policy_reranker_shapes_and_gradient() -> None:
    policy = PolicyReranker(embedding_dim=4, hidden_dim=8)
    logits = policy.logits(
        torch.ones(4), torch.ones(5, 4), torch.linspace(0, 1, 5), torch.ones(5)
    )
    loss = policy.reinforce_loss(logits, torch.linspace(0, 1, 5))
    loss.backward()
    assert logits.shape == (5,)
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_synthetic_service_is_reproducible() -> None:
    first = RecommenderService("missing-artifacts").recommend(2, 5)
    second = RecommenderService("missing-artifacts").recommend(2, 5)
    assert first == second

