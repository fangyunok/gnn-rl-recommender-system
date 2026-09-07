from __future__ import annotations

import torch
from torch import nn


class LightGCN(nn.Module):
    """LightGCN collaborative-filtering encoder."""

    def __init__(self, num_users: int, num_items: int, dim: int = 32, layers: int = 2):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.embedding = nn.Embedding(num_users + num_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self, graph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = [self.embedding.weight]
        current = embeddings[0]
        for _ in range(self.layers):
            current = torch.sparse.mm(graph, current)
            embeddings.append(current)
        output = torch.stack(embeddings).mean(dim=0)
        return output[: self.num_users], output[self.num_users :]

    def bpr_loss(
        self, graph: torch.Tensor, users: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        user_emb, item_emb = self.propagate(graph)
        pos_score = (user_emb[users] * item_emb[positive]).sum(dim=-1)
        neg_score = (user_emb[users] * item_emb[negative]).sum(dim=-1)
        ranking = -torch.nn.functional.logsigmoid(pos_score - neg_score).mean()
        reg = 1e-4 * (
            self.embedding(users).square().mean()
            + self.embedding(positive + self.num_users).square().mean()
            + self.embedding(negative + self.num_users).square().mean()
        )
        return ranking + reg

    @torch.no_grad()
    def candidates(self, graph: torch.Tensor, user_id: int, top_k: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        users, items = self.propagate(graph)
        scores = items @ users[user_id]
        return torch.topk(scores, k=min(top_k, self.num_items))

