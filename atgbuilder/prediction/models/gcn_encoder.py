# src/prediction/models/gcn_encoder.py
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class OutDegreeGCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr="add", node_dim=0)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)

        src, dst = edge_index
        deg = degree(src, x.size(0), dtype=x.dtype)
        deg_inv = 1.0 / deg.clamp(min=1.0)

        x = x * deg_inv.view(-1, 1)

        out = self.propagate(edge_index, x=x)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:

        return x_j


class GCNEncoder(nn.Module):


    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = OutDegreeGCNConv(in_channels, hidden_channels)
        self.conv2 = OutDegreeGCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        return x  # Z: [N, d]