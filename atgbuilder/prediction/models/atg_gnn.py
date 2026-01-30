from typing import Dict, Any, Optional

import torch
from torch import nn
from torch_geometric.data import Data

from src.prediction.models.gcn_encoder import GCNEncoder
from src.prediction.models.heads import GravityDecoder, EdgeAttributeHead
from src.prediction.models.moco import MoCoModule


def compute_binary_metrics(
    y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    y_true = y_true.detach().cpu()
    y_prob = y_prob.detach().cpu()
    y_pred = (y_prob >= threshold).long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class ATGGNN(nn.Module):
    def __init__(
        self,
        activity_dim: int,
        widget_dim: int,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        dropout: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        use_attr_head: bool = True,
        use_moco: bool = True,
        moco_mode: str = "queue",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.use_attr_head = use_attr_head and widget_dim > 0
        self.use_moco = use_moco
        self.moco_mode = moco_mode

        self.encoder = GCNEncoder(
            in_channels=activity_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            dropout=dropout,
        )
        self.gravity_head = GravityDecoder(embed_dim=embed_dim)

        if self.use_attr_head:
            self.attr_head = EdgeAttributeHead(
                node_dim=embed_dim,
                edge_dim=widget_dim,
                hidden_dim=hidden_dim,
            )
        else:
            self.attr_head = None

        if self.use_moco:
            self.moco = MoCoModule(
                embed_dim=embed_dim,
                proj_dim=128,
                queue_size=4096,
                momentum=0.999,
                temperature=0.2,
                mode=self.moco_mode,
            )
        else:
            self.moco = None

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, data: Data, compute_metrics: bool = True) -> Dict[str, Any]:
        x: torch.Tensor = data.x
        device = x.device

        # labels / attrs
        y = getattr(data, "y", None)
        edge_attr = getattr(data, "edge_attr", None)


        edge_index: torch.Tensor = data.edge_index

        mp_edge_index = getattr(data, "mp_edge_index", None)
        enc_edge_index = mp_edge_index if (mp_edge_index is not None) else edge_index

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
                metrics = compute_binary_metrics(y_float, prob)

        loss_attr: Optional[torch.Tensor] = None
        w_hat: Optional[torch.Tensor] = None
        if self.use_attr_head and self.attr_head is not None and edge_attr is not None:
            w_hat = self.attr_head(z, edge_index, edge_attr)  # [E, edge_dim]

            if y is not None:
                pos_mask = (y == 1)  # [E]
                if pos_mask.any():
                    loss_attr = self.mse_loss(w_hat[pos_mask], edge_attr[pos_mask])
                else:
                    loss_attr = torch.zeros(1, device=device)
            else:
                loss_attr = self.mse_loss(w_hat, edge_attr)
        else:
            loss_attr = torch.zeros(1, device=device)

        loss_contrast = torch.zeros(1, device=device)
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
            "metrics": metrics,
            "node_embeddings": z,
            "w_hat": w_hat,
            "moco_logits": moco_logits,
            "moco_labels": moco_labels,
            "moco_keys": moco_keys,
        }