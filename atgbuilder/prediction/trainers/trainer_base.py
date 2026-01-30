
import copy
import logging
import os
from typing import List, Dict, Optional, Any

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class BaseTrainer:


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
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.log_interval = int(log_interval)
        self.save_dir = save_dir

        self.learn_alpha = bool(learn_alpha)
        self.learn_beta = bool(learn_beta)
        self.learn_gamma = bool(learn_gamma)

        if self.learn_alpha:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(float(init_alpha), device=self.device, dtype=torch.float32))
            )
        else:
            self.alpha = float(init_alpha)

        if self.learn_beta:
            self.log_beta = nn.Parameter(
                torch.log(torch.tensor(float(init_beta), device=self.device, dtype=torch.float32))
            )
        else:
            self.beta = float(init_beta)

        if self.learn_gamma:
            self.log_gamma = nn.Parameter(
                torch.log(torch.tensor(float(init_gamma), device=self.device, dtype=torch.float32))
            )
        else:
            self.gamma = float(init_gamma)

        # ====== optional weak supervision terms ======
        self.lambda_seed = float(lambda_seed)
        self.lambda_boot = float(lambda_boot)
        self.boot_mix = float(boot_mix)
        self.bce_soft = nn.BCELoss()

        model_params = list(self.model.parameters())
        log_weight_params = []
        if self.learn_alpha:
            log_weight_params.append(self.log_alpha)
        if self.learn_beta:
            log_weight_params.append(self.log_beta)
        if self.learn_gamma:
            log_weight_params.append(self.log_gamma)

        param_groups = [
            {"params": model_params, "weight_decay": float(weight_decay), "lr": float(lr)},
        ]

        if len(log_weight_params) > 0:
            param_groups.append({
                "params": log_weight_params,
                "weight_decay": 0.0,
                "lr": float(lr) * 0.1,  # 你想要就保留；不想就改成 lr
            })

        self.optimizer = Adam(param_groups)

        if self.learn_gamma:
            logger.info("DBG init gamma: log_gamma=%.6f gamma=%.6f",
                        float(self.log_gamma.item()),
                        float(torch.exp(self.log_gamma).item()))

    def get_current_weights(self) -> Dict[str, float]:
        if self.learn_alpha:
            alpha = float(torch.exp(self.log_alpha.detach()).item())
        else:
            alpha = float(getattr(self, "alpha", 1.0))

        if self.learn_beta:
            beta = float(torch.exp(self.log_beta.detach()).item())
        else:
            beta = float(getattr(self, "beta", 0.0))

        if self.learn_gamma:
            gamma = float(torch.exp(self.log_gamma.detach()).item())
        else:
            gamma = float(getattr(self, "gamma", 0.0))

        return {"alpha": alpha, "beta": beta, "gamma": gamma}

    def _clamp_log_weights(self, min_log: float = -10.0, max_log: float = 3.0):
        with torch.no_grad():
            if self.learn_alpha:
                self.log_alpha.clamp_(min_log, max_log)
            if self.learn_beta:
                self.log_beta.clamp_(min_log, max_log)
            if self.learn_gamma:
                self.log_gamma.clamp_(min_log, max_log)

    @staticmethod
    def _make_loader(graphs: Optional[List[Data]], batch_size: int, shuffle: bool) -> Optional[DataLoader]:
        if graphs is None or len(graphs) == 0:
            return None
        return DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )


    def _collect_probs_labels(
            self,
            phase: str,
            epoch: int,
            batch_idx: int,
            num_batches: int,
            batch: Data,
            out: Dict[str, Any],
            all_prob: List[torch.Tensor],
            all_y: List[torch.Tensor],
            log_per_batch: bool = True,
    ) -> None:

        prob = out.get("prob", None)
        y_true_eval = getattr(batch, "y", None)

        if prob is None or y_true_eval is None:
            return

        prob_flat = prob.detach().view(-1).float()
        y_flat = y_true_eval.detach().view(-1).long()

        all_prob.append(prob_flat.cpu())
        all_y.append(y_flat.cpu())

        if log_per_batch:
            zeros = int((prob_flat == 0).sum().item())
            E = int(prob_flat.numel())
            zero_ratio_batch = float(zeros / max(E, 1))

        return

    # -------------------------
    # Main training loop
    # -------------------------

    def train(
            self,
            train_graphs: List[Data],
            val_graphs: Optional[List[Data]] = None,
            test_graphs: Optional[List[Data]] = None,
            run_name: str = "",
    ) -> Dict[str, Dict[str, float]]:

        train_loader = self._make_loader(train_graphs, batch_size=self.batch_size, shuffle=True)
        val_loader = self._make_loader(val_graphs, batch_size=self.batch_size, shuffle=False)
        test_loader = self._make_loader(test_graphs, batch_size=self.batch_size, shuffle=False)

        # legacy outputs (kept for summary compatibility)
        best_val_f1 = -1.0
        best_val_stats: Optional[Dict[str, float]] = None
        best_state = None
        best_weights: Optional[Dict[str, float]] = None
        best_epoch: int = -1

        # ====== B+ selection hyperparams (define once) ======
        P_MIN = 0.60
        FPR_MAX = 0.03

        # ====== B+ best trackers (define once) ======
        best_val_f1_ok = -1.0
        best_val_stats_ok = None
        best_state_ok = None
        best_weights_ok = None
        best_logw_ok = None
        best_epoch_ok = -1

        best_val_f1_any = -1.0
        best_val_stats_any = None
        best_state_any = None
        best_weights_any = None
        best_logw_any = None
        best_epoch_any = -1

        for epoch in range(1, self.epochs + 1):
            # 1) Train
            _ = self._run_one_epoch(
                loader=train_loader,
                train=True,
                epoch=epoch,
                phase=f"{run_name}-train" if run_name else "train",
            )

            # 2) Val
            if val_loader is not None:
                val_stats = self._run_one_epoch(
                    loader=val_loader,
                    train=False,
                    epoch=epoch,
                    phase=f"{run_name}-val" if run_name else "val",
                )

                p = float(val_stats.get("precision", 0.0))
                f1 = float(val_stats.get("f1", 0.0))
                fp = int(val_stats.get("fp", 0))
                tn = int(val_stats.get("tn", 0))
                fpr = fp / max(fp + tn, 1)

                ok = (p >= P_MIN) and (fpr <= FPR_MAX)

                # --- always track ANY best (fallback) ---
                if f1 > best_val_f1_any:
                    best_val_f1_any = f1
                    best_val_stats_any = val_stats
                    best_state_any = copy.deepcopy(self.model.state_dict())
                    best_weights_any = self.get_current_weights()
                    snap = {}
                    if self.learn_alpha: snap["log_alpha"] = self.log_alpha.detach().clone()
                    if self.learn_beta:  snap["log_beta"] = self.log_beta.detach().clone()
                    if self.learn_gamma: snap["log_gamma"] = self.log_gamma.detach().clone()
                    best_logw_any = snap
                    best_epoch_any = epoch

                # --- track OK best (primary) ---
                updated_ok = False
                if ok and (f1 > best_val_f1_ok):
                    best_val_f1_ok = f1
                    best_val_stats_ok = val_stats
                    best_state_ok = copy.deepcopy(self.model.state_dict())
                    best_weights_ok = self.get_current_weights()
                    snap = {}
                    if self.learn_alpha: snap["log_alpha"] = self.log_alpha.detach().clone()
                    if self.learn_beta:  snap["log_beta"] = self.log_beta.detach().clone()
                    if self.learn_gamma: snap["log_gamma"] = self.log_gamma.detach().clone()
                    best_logw_ok = snap
                    best_epoch_ok = epoch
                    updated_ok = True

                logger.info(
                    "🧪 [BPLUS_SELECT] %s epoch=%d | VAL: P=%.4f F1=%.4f FP=%d TN=%d FPR=%.6f ok=%s "
                    "| best_ok(F1=%.4f@%d) best_any(F1=%.4f@%d)%s",
                    (run_name if run_name else "run"),
                    epoch,
                    p, f1, fp, tn, fpr, str(ok),
                    float(best_val_f1_ok), int(best_epoch_ok),
                    float(best_val_f1_any), int(best_epoch_any),
                    " ✅update_ok" if updated_ok else ""
                )

        best_logw = None  # ✅ NEW

        if best_state_ok is not None:
            best_state = best_state_ok
            best_val_stats = best_val_stats_ok
            best_weights = best_weights_ok
            best_logw = best_logw_ok  # ✅ NEW
            best_epoch = best_epoch_ok
            best_val_f1 = float(best_val_f1_ok)
            logger.info(
                "✅ [BPLUS_FINAL] Using OK-best: epoch=%d F1=%.4f (P>=%.2f & FPR<=%.3f satisfied)",
                int(best_epoch), float(best_val_f1), float(P_MIN), float(FPR_MAX)
            )
        elif best_state_any is not None:
            best_state = best_state_any
            best_val_stats = best_val_stats_any
            best_weights = best_weights_any
            best_logw = best_logw_any  # ✅ NEW
            best_epoch = best_epoch_any
            best_val_f1 = float(best_val_f1_any)
            logger.info(
                "⚠️ [BPLUS_FINAL] No OK candidate found; fallback to ANY-best: epoch=%d F1=%.4f",
                int(best_epoch), float(best_val_f1)
            )
        else:
            logger.info("⚠️ [BPLUS_FINAL] No best model captured (unexpected).")

        # 3) Load best
        if best_state is not None:
            self.model.load_state_dict(best_state)

        if best_logw is not None:
            with torch.no_grad():
                if self.learn_alpha and "log_alpha" in best_logw:
                    self.log_alpha.copy_(best_logw["log_alpha"].to(self.device))
                if self.learn_beta and "log_beta" in best_logw:
                    self.log_beta.copy_(best_logw["log_beta"].to(self.device))
                if self.learn_gamma and "log_gamma" in best_logw:
                    self.log_gamma.copy_(best_logw["log_gamma"].to(self.device))

            logger.info(
                "✅ [BEST_LOGW_RESTORE] alpha=%.6f beta=%.6f gamma=%.6f",
                float(torch.exp(self.log_alpha).item()) if self.learn_alpha else float(getattr(self, "alpha", 1.0)),
                float(torch.exp(self.log_beta).item()) if self.learn_beta else float(getattr(self, "beta", 0.0)),
                float(torch.exp(self.log_gamma).item()) if self.learn_gamma else float(getattr(self, "gamma", 0.0)),
            )

        # 4) Final test (best model)
        test_stats = None
        if test_loader is not None:
            test_stats = self._run_one_epoch(
                loader=test_loader,
                train=False,
                epoch=self.epochs,
                phase=f"{run_name}-test" if run_name else "test",
            )

        weights_at_best = best_weights if best_weights is not None else self.get_current_weights()

        # Save (save best-loaded model)
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

            model_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(self.model.state_dict(), model_path)

            ckpt_path = os.path.join(self.save_dir, "best_model_ckpt.pt")
            ckpt = {
                "model_state": self.model.state_dict(),
                "best_epoch": int(best_epoch),
                "best_val_f1": float(best_val_f1),
                "weights_at_best": weights_at_best,
            }
            if self.learn_alpha: ckpt["log_alpha"] = self.log_alpha.detach().cpu()
            if self.learn_beta:  ckpt["log_beta"] = self.log_beta.detach().cpu()
            if self.learn_gamma: ckpt["log_gamma"] = self.log_gamma.detach().cpu()

            torch.save(ckpt, ckpt_path)

            logger.info("💾 Saved best_model state_dict to %s", model_path)
            logger.info("💾 Saved best_model checkpoint  to %s", ckpt_path)

        weights_at_best = best_weights if best_weights is not None else self.get_current_weights()
        weights_final = self.get_current_weights()

        summary: Dict[str, Any] = {
            "best_epoch": int(best_epoch),
            "best_val": best_val_stats if best_val_stats is not None else {"f1": best_val_f1},
            "test": test_stats,
            "weights_at_best_val": weights_at_best,
            "weights_final": weights_final,
            "best_val_F1": float((best_val_stats or {}).get("f1", best_val_f1)),
            "best_alpha": float(weights_at_best.get("alpha", 1.0)),
            "best_beta": float(weights_at_best.get("beta", 0.0)),
            "best_gamma": float(weights_at_best.get("gamma", 0.0)),
        }

        if test_stats is not None:
            summary["test_F1_thr0.5"] = {
                "F1": float(test_stats.get("f1", 0.0)),
                "P": float(test_stats.get("precision", 0.0)),
                "R": float(test_stats.get("recall", 0.0)),
            }

        return summary

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

        worst_graphs: List[Dict[str, Any]] = []

        from tqdm import tqdm
        pbar = tqdm(iterable=loader, desc=f"[{phase}] Epoch {epoch}/{self.epochs}", ncols=200)

        for b_idx, batch in enumerate(pbar, start=1):
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            if train:
                out = self.model(batch, compute_metrics=True)
            else:
                with torch.no_grad():
                    out = self.model(batch, compute_metrics=True)

            loss_exist = out.get("loss_exist")
            loss_attr = out.get("loss_attr")
            loss_contrast = out.get("loss_contrast")

            if loss_exist is None:
                raise ValueError("model 输出必须包含 loss_exist")

            if loss_attr is None:
                loss_attr = torch.tensor(0.0, device=self.device)
            if loss_contrast is None:
                loss_contrast = torch.tensor(0.0, device=self.device)

            loss_seed = torch.tensor(0.0, device=self.device)
            loss_boot = torch.tensor(0.0, device=self.device)

            if self.lambda_seed > 0.0 or self.lambda_boot > 0.0:
                prob = out.get("prob", None)
                if prob is not None:
                    prob_flat = prob.view(-1)

                    y_true = batch.y.float().view(-1) if hasattr(batch, "y") and batch.y is not None else None
                    y_seed = batch.y_seed.float().view(-1) if hasattr(batch,
                                                                      "y_seed") and batch.y_seed is not None else None

                    if self.lambda_seed > 0.0 and (y_seed is not None):
                        loss_seed = self.bce_soft(prob_flat, y_seed)

                    if self.lambda_boot > 0.0 and (y_true is not None):
                        y_boot = ((
                                              1.0 - self.boot_mix) * y_true + self.boot_mix * y_seed) if y_seed is not None else y_true
                        loss_boot = self.bce_soft(prob_flat, y_boot)

            alpha = torch.exp(self.log_alpha) if self.learn_alpha else torch.tensor(getattr(self, "alpha", 1.0),
                                                                                    device=self.device)
            beta = torch.exp(self.log_beta) if self.learn_beta else torch.tensor(getattr(self, "beta", 0.0),
                                                                                 device=self.device)
            gamma = torch.exp(self.log_gamma) if self.learn_gamma else torch.tensor(getattr(self, "gamma", 0.0),
                                                                                    device=self.device)

            loss_total = (
                    alpha * loss_exist
                    + beta * loss_attr
                    + gamma * loss_contrast
                    + self.lambda_seed * loss_seed
                    + self.lambda_boot * loss_boot
            )

            if train:
                self.optimizer.zero_grad(set_to_none=True)

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                if self.learn_gamma:
                    g = self.log_gamma.grad


                self.optimizer.step()
                self._clamp_log_weights()

                moco = getattr(self.model, "moco", None)
                if moco is not None:
                    keys = out.get("moco_keys", None)
                    moco.update_queue(keys)

            metrics = out.get("metrics") or {}

            # eval：收集 prob/y + zero_ratio debug（batch/graph）
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

        # train：batch-avg
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
