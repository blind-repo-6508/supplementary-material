# src/prediction/models/heads.py
from typing import Dict

import torch
from torch import nn


class GravityDecoder(nn.Module):


    def __init__(self, embed_dim: int, eps: float = 1e-7):
        super().__init__()
        self.mass_fc = nn.Linear(embed_dim, 1, bias=True)
        self.eps = eps

    def forward(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        src, dst = edge_index  # [E]
        z_i = z[src]  # [E, d]
        z_j = z[dst]  # [E, d]

        m_j = self.mass_fc(z_j).squeeze(-1)  # [E]

        dist2 = torch.sum((z_i - z_j) ** 2, dim=-1) + self.eps  # [E]
        log_d = torch.log(dist2)

        logits = m_j - log_d  # [E]
        prob = torch.sigmoid(logits)

        return {
            "logits": logits,
            "prob": prob,
            "mass_j": m_j,
            "dist2": dist2,
        }


class EdgeAttributeHead(nn.Module):


    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        if edge_dim <= 0:
            raise ValueError("edge_dim must be > 0 for EdgeAttributeHead.")

        layers = []
        in_dim = node_dim * 2 + edge_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, edge_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if edge_attr is None:
            raise ValueError("edge_attr is required for EdgeAttributeHead.")

        src, dst = edge_index
        z_i = z[src]
        z_j = z[dst]

        x = torch.cat([z_i, z_j, edge_attr], dim=-1)
        w_hat = self.mlp(x)
        return w_hat