
import copy
import logging
import os
from typing import List, Dict, Optional, Any, Tuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# 复用你这个 BaseTrainer（不覆盖、不修改）
from src.prediction.trainers.trainer_base import BaseTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


def _safe_get_logits_or_prob(out: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    logits = None
    for k in ("logits", "logit", "edge_logits", "pred_logits"):
        v = out.get(k, None)
        if v is not None:
            logits = v
            break

    prob = None
    for k in ("prob", "y_prob", "pred_prob", "edge_prob"):
        v = out.get(k, None)
        if v is not None:
            prob = v
            break

    return logits, prob


def _bce_weighted_from_out(
    out: Dict[str, Any],
    y: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:

    logits, prob = _safe_get_logits_or_prob(out)

    y_f = y.float().view(-1)
    w_f = w.float().view(-1)

    if logits is not None:
        logits_f = logits.view(-1)
        loss_raw = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_f, y_f, reduction="none"
        )
        return (loss_raw * w_f).mean()

    if prob is not None:
        prob_f = prob.view(-1).clamp(1e-7, 1.0 - 1e-7)
        loss_raw = torch.nn.functional.binary_cross_entropy(
            prob_f, y_f, reduction="none"
        )
        return (loss_raw * w_f).mean()


def _sample_pu_indices(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    knn_edge_index: Optional[torch.Tensor],
    neg_pos_ratio: int,
    hard_u_ratio: float,
    per_src_balance: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    E = int(edge_index.size(1))

    if y is None or y.numel() != E:
        raise ValueError(f"PU trainer got y={getattr(y, 'shape', None)} E={E}")

    pos_idx = (y == 1).nonzero(as_tuple=False).view(-1)
    u_idx = (y == 0).nonzero(as_tuple=False).view(-1)

    if pos_idx.numel() == 0 or u_idx.numel() == 0:
        sel = pos_idx
        return sel, torch.empty((0,), dtype=torch.long, device=edge_index.device)

    # ---------- build hard-U pool ----------
    hard_u_idx = torch.empty((0,), dtype=torch.long, device=edge_index.device)
    if knn_edge_index is not None and isinstance(knn_edge_index, torch.Tensor) and knn_edge_index.numel() > 0:
        ei_cpu = edge_index.detach().cpu()
        u_cpu = u_idx.detach().cpu()
        knn_cpu = knn_edge_index.detach().cpu()

        max_node = int(ei_cpu.max().item()) if ei_cpu.numel() > 0 else 0
        base = max_node + 1

        u_src = ei_cpu[0, u_cpu]
        u_dst = ei_cpu[1, u_cpu]
        u_key = (u_src * base + u_dst).tolist()

        knn_key = (knn_cpu[0] * base + knn_cpu[1]).tolist()
        knn_set = set(int(k) for k in knn_key)

        hard_mask = [int(k) in knn_set for k in u_key]
        hard_mask_t = torch.tensor(hard_mask, dtype=torch.bool, device=edge_index.device)
        hard_u_idx = u_idx[hard_mask_t]

    # easy-U = U \ hard-U
    if hard_u_idx.numel() > 0:
        hard_set = set(hard_u_idx.detach().cpu().tolist())
        u_cpu = u_idx.detach().cpu().tolist()
        keep = [i not in hard_set for i in u_cpu]
        u_mask = torch.tensor(keep, dtype=torch.bool, device=edge_index.device)
        easy_u_idx = u_idx[u_mask]
    else:
        easy_u_idx = u_idx

    # ---------- sampling ----------
    n_pos = int(pos_idx.numel())
    total_neg = int(neg_pos_ratio * n_pos)

    def _rand_pick(pool: torch.Tensor, k: int) -> torch.Tensor:
        if pool is None or pool.numel() == 0 or k <= 0:
            return torch.empty((0,), dtype=torch.long, device=edge_index.device)
        if pool.numel() <= k:
            return pool
        perm = torch.randperm(pool.numel(), device=edge_index.device)[:k]
        return pool[perm]

    if not per_src_balance:
        n_hard = int(round(hard_u_ratio * total_neg))
        n_easy = total_neg - n_hard

        neg_h = _rand_pick(hard_u_idx, n_hard)
        neg_e = _rand_pick(easy_u_idx, n_easy)

        need = total_neg - int(neg_h.numel() + neg_e.numel())
        if need > 0:
            extra = _rand_pick(easy_u_idx, need)
            neg_e = torch.cat([neg_e, extra], dim=0)
            need2 = total_neg - int(neg_h.numel() + neg_e.numel())
            if need2 > 0:
                extra2 = _rand_pick(hard_u_idx, need2)
                neg_h = torch.cat([neg_h, extra2], dim=0)

        neg_idx = torch.cat([neg_h, neg_e], dim=0)
        sel_idx = torch.cat([pos_idx, neg_idx], dim=0)
        return sel_idx, neg_idx

    # ---------- per-src balance sampling ----------
    src_all = edge_index[0]
    pos_src = src_all[pos_idx]
    uniq_src, cnt = torch.unique(pos_src, return_counts=True)

    neg_list = []
    for s, c in zip(uniq_src.tolist(), cnt.tolist()):
        n_neg_s = int(neg_pos_ratio * int(c))
        if n_neg_s <= 0:
            continue

        s_val = int(s)
        hard_pool_s = hard_u_idx[(src_all[hard_u_idx] == s_val)] if hard_u_idx.numel() > 0 else torch.empty(
            (0,), dtype=torch.long, device=edge_index.device
        )
        easy_pool_s = easy_u_idx[(src_all[easy_u_idx] == s_val)] if easy_u_idx.numel() > 0 else torch.empty(
            (0,), dtype=torch.long, device=edge_index.device
        )

        n_hard_s = int(round(hard_u_ratio * n_neg_s))
        n_easy_s = n_neg_s - n_hard_s

        pick_h = _rand_pick(hard_pool_s, n_hard_s)
        pick_e = _rand_pick(easy_pool_s, n_easy_s)

        need = n_neg_s - int(pick_h.numel() + pick_e.numel())
        if need > 0:
            extra = _rand_pick(easy_pool_s, need)
            pick_e = torch.cat([pick_e, extra], dim=0)
            need2 = n_neg_s - int(pick_h.numel() + pick_e.numel())
            if need2 > 0:
                extra2 = _rand_pick(hard_pool_s, need2)
                pick_h = torch.cat([pick_h, extra2], dim=0)

        neg_list.append(torch.cat([pick_h, pick_e], dim=0))

    neg_idx = torch.cat(neg_list, dim=0) if len(neg_list) > 0 else torch.empty(
        (0,), dtype=torch.long, device=edge_index.device
    )
    sel_idx = torch.cat([pos_idx, neg_idx], dim=0)
    return sel_idx, neg_idx


def _make_pu_batch(batch: Data, sel_idx: torch.Tensor, lambda_u: float, pos_w: float):
    b2 = batch.clone()
    b2.edge_index = batch.edge_index[:, sel_idx]
    b2.y = batch.y[sel_idx]

    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        b2.edge_attr = batch.edge_attr[sel_idx]
    if hasattr(batch, "y_seed") and batch.y_seed is not None:
        b2.y_seed = batch.y_seed[sel_idx]

    w = torch.empty((sel_idx.numel(),), device=batch.edge_index.device, dtype=torch.float32)
    w_pos = torch.full_like(w, float(pos_w))
    w_neg = torch.full_like(w, float(lambda_u))
    w = torch.where(b2.y.view(-1) == 1, w_pos, w_neg)
    return b2, w


def _choose_pkeep(
    pkeeps: List[float],
    probs: List[float],
    device: torch.device,
) -> float:
    if len(pkeeps) == 0:
        return 1.0
    if probs is None or len(probs) != len(pkeeps):
        w = torch.ones(len(pkeeps), device=device, dtype=torch.float32)
    else:
        w = torch.tensor(probs, device=device, dtype=torch.float32).clamp_min(0)
        if float(w.sum().item()) <= 0:
            w = torch.ones(len(pkeeps), device=device, dtype=torch.float32)

    idx = torch.multinomial(w / w.sum(), num_samples=1).item()
    return float(pkeeps[idx])


def _apply_seed_dropout_inplace(
    batch: Data,
    pkeeps: List[float],
    probs: List[float],
    *,
    per_graph: bool = True,
) -> Dict[str, Any]:

    assert hasattr(batch, "y") and batch.y is not None
    device = batch.y.device

    y = batch.y.view(-1).clone()
    y_seed = None
    if hasattr(batch, "y_seed") and batch.y_seed is not None:
        y_seed = batch.y_seed.view(-1).clone()

    E = int(y.numel())
    if E == 0:
        batch.y = y
        if y_seed is not None:
            batch.y_seed = y_seed
        return {"E": 0, "pos_before": 0, "pos_after": 0, "pkeep_hist": {}}

    pos_idx = (y == 1).nonzero(as_tuple=False).view(-1)
    pos_before = int(pos_idx.numel())
    if pos_before == 0:
        batch.y = y
        if y_seed is not None:
            batch.y_seed = y_seed
        return {"E": E, "pos_before": 0, "pos_after": 0, "pkeep_hist": {}}

    pkeep_hist: Dict[str, int] = {}

    if per_graph:
        ei = batch.edge_index
        bb = getattr(batch, "batch", None)
        ptr = getattr(batch, "ptr", None)
        if ei is None or bb is None or ptr is None:
            per_graph = False
        else:
            src = ei[0]
            g_of_edge = bb[src]
            num_g = int(ptr.numel() - 1)

            for gi in range(num_g):
                mask_g_pos = (g_of_edge == gi) & (y == 1)
                idx_g_pos = mask_g_pos.nonzero(as_tuple=False).view(-1)
                if idx_g_pos.numel() == 0:
                    continue

                p_keep = _choose_pkeep(pkeeps, probs, device=device)
                key = f"{p_keep:.1f}"
                pkeep_hist[key] = pkeep_hist.get(key, 0) + 1

                keep = (torch.rand(idx_g_pos.numel(), device=device) < p_keep)
                drop_idx = idx_g_pos[~keep]
                if drop_idx.numel() > 0:
                    y[drop_idx] = 0
                    if y_seed is not None:
                        y_seed[drop_idx] = 0

    if not per_graph:
        p_keep = _choose_pkeep(pkeeps, probs, device=device)
        key = f"{p_keep:.1f}"
        pkeep_hist[key] = pkeep_hist.get(key, 0) + 1

        keep = (torch.rand(pos_idx.numel(), device=device) < p_keep)
        drop_idx = pos_idx[~keep]
        if drop_idx.numel() > 0:
            y[drop_idx] = 0
            if y_seed is not None:
                y_seed[drop_idx] = 0

    batch.y = y
    if y_seed is not None:
        batch.y_seed = y_seed

    pos_after = int((y == 1).sum().item())
    return {"E": E, "pos_before": pos_before, "pos_after": pos_after, "pkeep_hist": pkeep_hist}


class ATGTrainerPULearnWeights(BaseTrainer):


    logger.info("[TRAINER_FILE] %s", os.path.abspath(__file__))

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        batch_size: int = 128,
        log_interval: int = 10,
        save_dir: Optional[str] = None,
        init_alpha: float = 1.0,
        init_beta: float = 0.5,
        init_gamma: float = 0.1,
        learn_alpha: bool = False,
        learn_beta: bool = True,
        learn_gamma: bool = True,
        lambda_seed: float = 0.0,
        lambda_boot: float = 0.0,
        boot_mix: float = 0.5,
        # ===== PU params =====
        pu_neg_pos_ratio: int = 5,
        pu_hard_u_ratio: float = 0.5,
        pu_lambda_u: float = 0.2,
        pu_per_src_balance: bool = True,
        # ===== [SEED_DROPOUT] params =====
        seed_dropout: bool = True,
        seed_dropout_pkeeps: Optional[List[float]] = None,
        seed_dropout_probs: Optional[List[float]] = None,
        seed_dropout_per_graph: bool = True,
    ):
        super().__init__(
            model=model,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            log_interval=log_interval,
            save_dir=save_dir,
            init_alpha=init_alpha,
            init_beta=init_beta,
            init_gamma=init_gamma,
            learn_alpha=learn_alpha,
            learn_beta=learn_beta,
            learn_gamma=learn_gamma,
            lambda_seed=lambda_seed,
            lambda_boot=lambda_boot,
            boot_mix=boot_mix,
        )

        self.pu_neg_pos_ratio = int(pu_neg_pos_ratio)
        self.pu_hard_u_ratio = float(pu_hard_u_ratio)
        self.pu_lambda_u = float(pu_lambda_u)
        self.pu_per_src_balance = bool(pu_per_src_balance)

        logger.info(
            "🧷 [PU_CONFIG] neg_pos_ratio=%d hard_u_ratio=%.3f lambda_u=%.3f per_src_balance=%s",
            self.pu_neg_pos_ratio, self.pu_hard_u_ratio, self.pu_lambda_u, str(self.pu_per_src_balance)
        )

        self.seed_dropout = bool(seed_dropout)
        self.seed_dropout_pkeeps = seed_dropout_pkeeps or [1.0, 0.5, 0.2, 0.0]
        self.seed_dropout_probs = seed_dropout_probs or [0.4, 0.3, 0.2, 0.1]
        self.seed_dropout_per_graph = bool(seed_dropout_per_graph)

        logger.info(
            "🧷 [SEED_DROPOUT] enabled=%s pkeeps=%s probs=%s per_graph=%s",
            str(self.seed_dropout),
            str(self.seed_dropout_pkeeps),
            str(self.seed_dropout_probs),
            str(self.seed_dropout_per_graph),
        )

    def _run_one_epoch(
        self,
        loader: Optional[DataLoader],
        train: bool,
        epoch: int,
        phase: str,
    ) -> Dict[str, float]:

        if loader is None:
            return {
                "loss_total": 0.0,
                "loss_exist": 0.0,
                "loss_attr": 0.0,
                "loss_contrast": 0.0,
                "loss_seed": 0.0,
                "loss_boot": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
            }

        self.model.train() if train else self.model.eval()

        total_loss = total_exist = total_attr = total_contrast = 0.0
        total_seed = total_boot = 0.0

        total_prec = total_rec = total_f1 = total_acc = 0.0
        n_batches = 0

        all_prob: List[torch.Tensor] = []
        all_y: List[torch.Tensor] = []

        from tqdm import tqdm
        pbar = tqdm(iterable=loader, desc=f"[{phase}] Epoch {epoch}/{self.epochs}", ncols=200)

        for b_idx, batch in enumerate(pbar, start=1):
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            # 默认：ref_batch 就是 batch（eval 也用它）
            pu_batch = batch
            w = None
            # --- ALWAYS init losses to avoid UnboundLocalError ---
            loss_exist = torch.tensor(0.0, device=self.device)
            loss_attr = torch.tensor(0.0, device=self.device)
            loss_contrast = torch.tensor(0.0, device=self.device)
            loss_seed = torch.tensor(0.0, device=self.device)
            loss_boot = torch.tensor(0.0, device=self.device)
            did_pu_supervision = False  # 本 batch 是否真的用到了 PU 的 exist 监督

            # ========= TRAIN: PU sub-sample =========
            if train:
                # --- seed dropout (label-level) ---
                if self.seed_dropout:
                    sd_stat = _apply_seed_dropout_inplace(
                        batch,
                        pkeeps=self.seed_dropout_pkeeps,
                        probs=self.seed_dropout_probs,
                        per_graph=self.seed_dropout_per_graph,
                    )
                    if (b_idx % max(self.log_interval, 1) == 0) and sd_stat["E"] > 0:
                        logger.info(
                            "🧷 [SEED_DROPOUT_STAT] %s epoch=%d batch=%d pos_before=%d pos_after=%d pkeep_hist=%s",
                            phase, epoch, b_idx,
                            int(sd_stat["pos_before"]), int(sd_stat["pos_after"]),
                            str(sd_stat["pkeep_hist"]),
                        )

                # --- PU sampling on (possibly dropped) labels ---
                knn_ei = getattr(batch, "knn_edge_index", None)
                sel_idx, neg_idx = _sample_pu_indices(
                    edge_index=batch.edge_index,
                    y=batch.y,
                    knn_edge_index=knn_ei,
                    neg_pos_ratio=self.pu_neg_pos_ratio,
                    hard_u_ratio=self.pu_hard_u_ratio,
                    per_src_balance=self.pu_per_src_balance,
                )

                # --- seedless fallback: no pos after dropout => do NOT train exist ---
                if sel_idx.numel() == 0:
                    out = self.model(batch, compute_metrics=True)

                    loss_exist = torch.tensor(0.0, device=self.device)

                    loss_attr = out.get("loss_attr")
                    loss_contrast = out.get("loss_contrast")
                    if loss_attr is None:
                        loss_attr = torch.tensor(0.0, device=self.device)
                    if loss_contrast is None:
                        loss_contrast = torch.tensor(0.0, device=self.device)

                    pu_batch = batch
                    w = None
                    did_pu_supervision = False
                else:
                    pos_w = float(self.pu_neg_pos_ratio * self.pu_lambda_u)
                    pu_batch, w = _make_pu_batch(
                        batch=batch,
                        sel_idx=sel_idx,
                        lambda_u=self.pu_lambda_u,
                        pos_w=pos_w,
                    )

                    out = self.model(pu_batch, compute_metrics=True)
                    loss_exist = _bce_weighted_from_out(out=out, y=pu_batch.y, w=w)
                    did_pu_supervision = True

            else:
                with torch.no_grad():
                    out = self.model(batch, compute_metrics=True)
                loss_exist = out.get("loss_exist")
                if loss_exist is None:
                    raise ValueError("model 输出必须包含 loss_exist（eval 阶段）")

                # --- read optional heads safely ---
                if out is None:
                    raise RuntimeError("model forward returned None")

                if not isinstance(out, dict):
                    raise TypeError(f"model forward must return dict, got {type(out)}")

                tmp = out.get("loss_attr", None)
                if tmp is not None:
                    loss_attr = tmp

                tmp = out.get("loss_contrast", None)
                if tmp is not None:
                    loss_contrast = tmp

            # optional weak supervision terms
            loss_seed = torch.tensor(0.0, device=self.device)
            loss_boot = torch.tensor(0.0, device=self.device)

            if (self.lambda_seed > 0.0 or self.lambda_boot > 0.0) and did_pu_supervision:
                prob = out.get("prob", None)
                if prob is not None:
                    prob_flat = prob.view(-1)

                    ref_batch = pu_batch
                    y_true = ref_batch.y.float().view(-1) if (hasattr(ref_batch, "y") and ref_batch.y is not None) else None
                    y_seed = ref_batch.y_seed.float().view(-1) if (hasattr(ref_batch, "y_seed") and ref_batch.y_seed is not None) else None

                    if self.lambda_seed > 0.0 and (y_seed is not None):
                        loss_seed = self.bce_soft(prob_flat, y_seed)

                    if self.lambda_boot > 0.0 and (y_true is not None):
                        y_boot = ((1.0 - self.boot_mix) * y_true + self.boot_mix * y_seed) if (y_seed is not None) else y_true
                        loss_boot = self.bce_soft(prob_flat, y_boot)

            alpha = torch.exp(self.log_alpha) if self.learn_alpha else torch.tensor(getattr(self, "alpha", 1.0), device=self.device)
            beta = torch.exp(self.log_beta) if self.learn_beta else torch.tensor(getattr(self, "beta", 0.0), device=self.device)
            gamma = torch.exp(self.log_gamma) if self.learn_gamma else torch.tensor(getattr(self, "gamma", 0.0), device=self.device)

            loss_total = (
                alpha * loss_exist
                + beta * loss_attr
                + gamma * loss_contrast
                + self.lambda_seed * loss_seed
                + self.lambda_boot * loss_boot
            )

            if train:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                self._clamp_log_weights()

            metrics = out.get("metrics") or {}

            if not train:
                self._collect_probs_labels(
                    phase=phase,
                    epoch=epoch,
                    batch_idx=b_idx,
                    num_batches=len(loader),
                    batch=batch,
                    out=out,
                    all_prob=all_prob,
                    all_y=all_y,
                    log_per_batch=True,
                )

            total_loss += float(loss_total.item())
            total_exist += float(loss_exist.item())
            total_attr += float(loss_attr.item())
            total_contrast += float(loss_contrast.item())
            total_seed += float(loss_seed.item())
            total_boot += float(loss_boot.item())

            total_prec += float(metrics.get("precision", 0.0) or 0.0)
            total_rec += float(metrics.get("recall", 0.0) or 0.0)
            total_f1 += float(metrics.get("f1", 0.0) or 0.0)
            total_acc += float(metrics.get("accuracy", 0.0) or 0.0)

            n_batches += 1
            pbar.set_postfix(loss=total_loss / max(n_batches, 1), F1=total_f1 / max(n_batches, 1))

        if n_batches == 0:
            return {
                "loss_total": 0.0,
                "loss_exist": 0.0,
                "loss_attr": 0.0,
                "loss_contrast": 0.0,
                "loss_seed": 0.0,
                "loss_boot": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
            }

        if train:
            return {
                "loss_total": total_loss / n_batches,
                "loss_exist": total_exist / n_batches,
                "loss_attr": total_attr / n_batches,
                "loss_contrast": total_contrast / n_batches,
                "loss_seed": total_seed / n_batches,
                "loss_boot": total_boot / n_batches,
                "precision": total_prec / n_batches,
                "recall": total_rec / n_batches,
                "f1": total_f1 / n_batches,
                "accuracy": total_acc / n_batches,
            }

        from src.prediction.models.atg_gnn import compute_binary_metrics

        if len(all_prob) == 0 or len(all_y) == 0:
            micro = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        else:
            y_prob = torch.cat(all_prob, dim=0)
            y_true = torch.cat(all_y, dim=0)
            micro = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=0.5)

        batch_avg_f1 = float(total_f1 / max(n_batches, 1))
        micro_f1 = float(micro.get("f1", 0.0))
        logger.info(
            "🧩 [EVAL_METRIC_CHECK] phase=%s epoch=%d batch_avg_f1=%.6f micro_f1=%.6f",
            phase, epoch, batch_avg_f1, micro_f1
        )

        return {
            "loss_total": total_loss / n_batches,
            "loss_exist": total_exist / n_batches,
            "loss_attr": total_attr / n_batches,
            "loss_contrast": total_contrast / n_batches,
            "loss_seed": total_seed / n_batches,
            "loss_boot": total_boot / n_batches,
            "precision": float(micro.get("precision", 0.0)),
            "recall": float(micro.get("recall", 0.0)),
            "f1": float(micro.get("f1", 0.0)),
            "accuracy": float(micro.get("accuracy", 0.0)),
            "tp": int(micro.get("tp", 0)),
            "fp": int(micro.get("fp", 0)),
            "fn": int(micro.get("fn", 0)),
            "tn": int(micro.get("tn", 0)),
        }