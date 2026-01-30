
import torch
from torch import nn
from torch_geometric.nn.conv import GINConv


class GINEncoder(nn.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        layers = []
        last_dim = in_channels
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(last_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(mlp)
            layers.append(conv)
            last_dim = hidden_dim

        self.convs = nn.ModuleList(layers)

        self.proj = nn.Linear(last_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)

        z = self.proj(h)
        return z