
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch_geometric.data import Data

from src.prediction.models.gin_encoder import GINEncoder
from src.prediction.models.heads import GravityDecoder, EdgeAttributeHead
from src.prediction.models.moco import MoCoModule
from src.prediction.models.atg_gnn import compute_binary_metrics


class ATGGNNGinConsistency(nn.Module):
    def __init__(
        self,
        activity_dim: int,
        widget_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        dropout: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        use_attr_head: bool = True,
        use_moco: bool = False,
        moco_mode: str = "queue",
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_attr_head = use_attr_head
        self.use_moco = use_moco
        self.moco_mode = moco_mode

        self.encoder = GINEncoder(
            in_channels=activity_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.gravity_head = GravityDecoder(embed_dim)

        self.attr_head: Optional[EdgeAttributeHead] = None
        if use_attr_head and widget_dim is not None and widget_dim > 0:
            self.attr_head = EdgeAttributeHead(
                node_dim=embed_dim,
                edge_dim=widget_dim,
                hidden_dim=hidden_dim,
            )

        self.moco: Optional[MoCoModule] = None
        if use_moco:
            self.moco = MoCoModule(
                embed_dim=embed_dim,
                proj_dim=128,
                queue_size=1024,
                momentum=0.999,
                temperature=0.07,
                mode=self.moco_mode,
            )

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, data: Data, compute_metrics: bool = True) -> Dict[str, Any]:
        device = data.x.device
        x = data.x

        edge_index = data.edge_index

        mp_edge_index = getattr(data, "mp_edge_index", None)
        enc_edge_index = mp_edge_index if (mp_edge_index is not None) else edge_index

        y = getattr(data, "y", None)
        edge_attr = getattr(data, "edge_attr", None)

        z = self.encoder(x, enc_edge_index)

        gravity_out = self.gravity_head(z, edge_index)
        logits: torch.Tensor = gravity_out["logits"]
        prob: torch.Tensor = gravity_out["prob"]

        loss_exist: Optional[torch.Tensor] = None
        metrics: Optional[Dict[str, float]] = None
        if y is not None:
            y_float = y.float()
            loss_exist = self.bce_loss(logits, y_float)
            if compute_metrics:
                metrics = compute_binary_metrics(y_float, prob, threshold=0.5)

        loss_attr: Optional[torch.Tensor] = None
        w_hat: Optional[torch.Tensor] = None
        if self.use_attr_head and self.attr_head is not None and edge_attr is not None:
            w_hat = self.attr_head(z, edge_index, edge_attr)  # [E, edge_dim]

            if y is not None:
                pos_mask = (y == 1)  # [E]
                if pos_mask.any():
                    w_hat_pos = w_hat[pos_mask]  # [E_pos, edge_dim]
                    w_true_pos = edge_attr[pos_mask]  # [E_pos, edge_dim]
                    loss_attr = self.mse_loss(w_hat_pos, w_true_pos)
                else:
                    loss_attr = torch.zeros(1, device=device)

        loss_contrast: Optional[torch.Tensor] = None
        moco_logits = None
        moco_labels = None
        moco_keys = None
        if self.use_moco and self.moco is not None:
            moco_out = self.moco(z)
            loss_contrast = moco_out["loss_contrast"]
            moco_logits = moco_out.get("logits")
            moco_labels = moco_out.get("labels")
            moco_keys = moco_out.get("keys")

        loss_total = torch.zeros(1, device=device)
        if loss_exist is not None:
            loss_total = loss_total + self.alpha * loss_exist
        if loss_attr is not None:
            loss_total = loss_total + self.beta * loss_attr
        if loss_contrast is not None:
            loss_total = loss_total + self.gamma * loss_contrast

        return {
            "loss_total": loss_total,
            "loss_exist": loss_exist,
            "loss_attr": loss_attr,
            "loss_contrast": loss_contrast,
            "logits": logits,
            "prob": prob,
            "z": z,
            "w_hat": w_hat,
            "moco_logits": moco_logits,
            "moco_labels": moco_labels,
            "moco_keys": moco_keys,
            "metrics": metrics,
        }