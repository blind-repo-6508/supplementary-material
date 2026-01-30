
from typing import Dict, Optional, Union, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MoCoModule(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        proj_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2,
        mode: str = "queue",
    ):
        super().__init__()
        assert mode in ("queue", "fixed"), f"Unsupported MoCo mode: {mode}"
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.queue_size = int(queue_size)
        self.mode = mode

        self.q_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # queue: [C, K]
        self.register_buffer("queue", torch.randn(proj_dim, self.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._init_key_encoder()

    @torch.no_grad()
    def _init_key_encoder(self):
        for param_q, param_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.q_proj.parameters(), self.k_proj.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def update_queue(self, keys: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]):

        if self.mode == "fixed" or getattr(self, "fixed_moco", False):
            return

        if keys is None:
            return

        if isinstance(keys, (list, tuple)):
            keys_list = [k for k in keys if k is not None]
            if len(keys_list) == 0:
                return
            keys = torch.cat(keys_list, dim=0)
            if keys.numel() == 0:
                return

        if keys.dim() == 3:
            keys = keys.reshape(-1, keys.size(-1))

        keys = keys.detach()
        if keys.device != self.queue.device:
            keys = keys.to(self.queue.device)

        C = int(self.queue.size(0))

        if keys.dim() == 2 and keys.size(-1) != C and keys.size(0) == C:
            keys = keys.t()

        if keys.dim() != 2 or keys.size(-1) != C:
            raise RuntimeError(
                f"MoCo.update_queue: bad keys shape {tuple(keys.shape)}, expected (*, {C})"
            )

        keys = F.normalize(keys, dim=-1)

        K = int(self.queue.size(1))
        if keys.size(0) > K:
            keys = keys[:K]

        if keys.size(0) == 0:
            return

        self._dequeue_and_enqueue(keys)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # keys: [bs, C]
        bs = int(keys.size(0))
        K = int(self.queue.size(1))

        ptr = int(self.queue_ptr.item())
        end = ptr + bs

        if end <= K:
            self.queue[:, ptr:end] = keys.t()
        else:
            first = K - ptr
            self.queue[:, ptr:] = keys[:first].t()
            remain = bs - first
            self.queue[:, :remain] = keys[first:first + remain].t()

        self.queue_ptr[0] = (ptr + bs) % K

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:

        device = z.device

        if z.size(0) < 2:
            return {
                "loss_contrast": torch.tensor(0.0, device=device),
                "logits": None,
                "labels": None,
                "keys": None,
            }

        # query
        q = self.q_proj(z)  # [N, C]
        q = F.normalize(q, dim=-1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.k_proj(z)  # [N, C]
            k = F.normalize(k, dim=-1)

        l_pos = torch.sum(q * k, dim=-1, keepdim=True)  # [N, 1]

        with torch.no_grad():
            queue = self.queue.detach()  # [C, K]
        l_neg = torch.matmul(q, queue)  # [N, K]

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]
        logits = logits / self.temperature

        labels = torch.zeros(z.size(0), dtype=torch.long, device=device)
        loss_contrast = F.cross_entropy(logits, labels)

        return {
            "loss_contrast": loss_contrast,
            "logits": logits,
            "labels": labels,
            "keys": k.detach(),
        }