from __future__ import annotations

import torch
from torch import nn


class PolicyReranker(nn.Module):
    """Contextual policy that reranks GNN candidates for accuracy and diversity."""

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def logits(
        self,
        user: torch.Tensor,
        items: torch.Tensor,
        relevance: torch.Tensor,
        novelty: torch.Tensor,
    ) -> torch.Tensor:
        repeated_user = user.expand(items.shape[0], -1)
        features = torch.cat(
            [repeated_user, items, relevance.unsqueeze(-1), novelty.unsqueeze(-1)], dim=-1
        )
        return self.policy(features).squeeze(-1)

    def reinforce_loss(self, logits: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample((32,))
        sampled_rewards = rewards[actions]
        advantage = sampled_rewards - sampled_rewards.mean()
        return -(distribution.log_prob(actions) * advantage.detach()).mean()

