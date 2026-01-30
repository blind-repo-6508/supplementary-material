# src/prediction/training/atg_trainer_dynamic_learn_weights.py
# -*- coding: utf-8 -*-
import logging
import os
import copy
from typing import List, Dict, Optional, Any

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.prediction.utils.balanced_sampling import BalancedSampler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True


class ATGTrainerDynamicLearnWeights:

    logger.info("[TRAINER_FILE] %s", os.path.abspath(__file__))

    def __init__(
            self,
            model: nn.Module,
            device: Optional[torch.device] = None,
            lr: float = 0.01,
            weight_decay: float = 1e-5,
            epochs: int = 100,
            batch_size: int = 128,
            balance_edges: bool = True,
            balance_rate: float = 1.0,
            log_interval: int = 10,
            save_dir: Optional[str] = None,
            random_seed: int = 42,
            init_alpha: float = 1.0,
            init_beta: float = 0.5,
            init_gamma: float = 0.1,
            learn_alpha: bool = False,
            learn_beta: bool = True,
            learn_gamma: bool = True,
            balance_num_workers: int = 0,
            lambda_seed: float = 0.0,
            lambda_boot: float = 0.0,
            boot_mix: float = 0.5,
            balance_eval_edges: bool = False,
    ):
        self.balance_num_workers = balance_num_workers
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_dir = save_dir

        self.balance_edges = balance_edges
        self.balance_rate = balance_rate

        self.learn_alpha = learn_alpha
        self.learn_beta = learn_beta
        self.learn_gamma = learn_gamma

        if learn_alpha:
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(init_alpha), dtype=torch.float32)))
        else:
            self.alpha = float(init_alpha)

        if learn_beta:
            self.log_beta = nn.Parameter(torch.log(torch.tensor(float(init_beta), dtype=torch.float32)))
        else:
            self.beta = float(init_beta)

        if learn_gamma:
            self.log_gamma = nn.Parameter(torch.log(torch.tensor(float(init_gamma), dtype=torch.float32)))
        else:
            self.gamma = float(init_gamma)

        self.lambda_seed = float(lambda_seed)
        self.lambda_boot = float(lambda_boot)
        self.boot_mix = float(boot_mix)
        self.bce_soft = nn.BCELoss()

        self.balancer = BalancedSampler(random_seed=random_seed) if balance_edges else None

        params = list(self.model.parameters())
        if learn_alpha:
            params.append(self.log_alpha)
        if learn_beta:
            params.append(self.log_beta)
        if learn_gamma:
            params.append(self.log_gamma)

        self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
        self.balance_eval_edges = balance_eval_edges

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

    def _clamp_log_weights(self, min_log: float = -3.0, max_log: float = 3.0):
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
        return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

    def _balance_once_for_eval(self, graphs: Optional[List[Data]], phase: str) -> Optional[List[Data]]:
        if graphs is None:
            return None
        if (not self.balance_eval_edges) or (not self.balance_edges) or (self.balancer is None):
            return graphs

        return self.balancer.balance_graphs_list(graphs, rate=self.balance_rate, num_workers=self.balance_num_workers)

    def train(
            self,
            train_graphs: List[Data],
            val_graphs: Optional[List[Data]] = None,
            test_graphs: Optional[List[Data]] = None,
            run_name: str = "",
    ) -> Dict[str, Dict[str, float]]:

        orig_train_graphs = train_graphs

        balanced_val_graphs = self._balance_once_for_eval(val_graphs, phase="val")
        balanced_test_graphs = self._balance_once_for_eval(test_graphs, phase="test")

        val_loader = self._make_loader(balanced_val_graphs, batch_size=self.batch_size, shuffle=False)
        test_loader = self._make_loader(balanced_test_graphs, batch_size=self.batch_size, shuffle=False)

        best_val_f1 = -1.0
        best_val_stats: Optional[Dict[str, float]] = None
        best_state = None
        best_weights: Optional[Dict[str, float]] = None
        best_epoch: int = -1

        for epoch in range(1, self.epochs + 1):
            if self.balance_edges and self.balancer is not None:
                balanced_train_graphs = self.balancer.balance_graphs_list(
                    orig_train_graphs,
                    rate=self.balance_rate,
                    num_workers=self.balance_num_workers,
                )
            else:
                balanced_train_graphs = orig_train_graphs

            train_loader = self._make_loader(balanced_train_graphs, batch_size=self.batch_size, shuffle=True)

            train_stats = self._run_one_epoch(
                loader=train_loader,
                train=True,
                epoch=epoch,
                phase=f"{run_name}-train" if run_name else "train",
            )

            if val_loader is not None:
                val_stats = self._run_one_epoch(
                    loader=val_loader,
                    train=False,
                    epoch=epoch,
                    phase=f"{run_name}-val" if run_name else "val",
                )

                if val_stats["f1"] > best_val_f1:
                    best_val_f1 = val_stats["f1"]
                    best_val_stats = val_stats
                    # ✅ 关键修复：必须 deepcopy，否则后续训练会把 best_state 覆盖掉
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_weights = self.get_current_weights()
                    best_epoch = epoch

        if best_state is not None:
            self.model.load_state_dict(best_state)

        test_stats = None
        if test_loader is not None:
            test_stats = self._run_one_epoch(
                loader=test_loader,
                train=False,
                epoch=self.epochs,
                phase=f"{run_name}-test" if run_name else "test",
            )

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            model_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(self.model.state_dict(), model_path)

        weights_at_best = best_weights if best_weights is not None else self.get_current_weights()
        weights_final = self.get_current_weights()

        summary: Dict[str, Any] = {
            "best_epoch": int(best_epoch),
            "best_val": best_val_stats if best_val_stats is not None else {"f1": best_val_f1},
            "test": test_stats,
            "weights_at_best_val": weights_at_best,
            "weights_final": weights_final,
            "best_val_F1": float((best_val_stats or {}).get("f1", best_val_f1)),
        }

        if test_stats is not None:
            summary["test_F1_thr0.5"] = {
                "F1": float(test_stats.get("f1", 0.0)),
                "P": float(test_stats.get("precision", 0.0)),
                "R": float(test_stats.get("recall", 0.0)),
            }

        summary["best_alpha"] = float(weights_at_best.get("alpha", 1.0))
        summary["best_beta"] = float(weights_at_best.get("beta", 0.0))
        summary["best_gamma"] = float(weights_at_best.get("gamma", 0.0))

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

        all_prob = []
        all_y = []

        from tqdm import tqdm
        pbar = tqdm(iterable=loader, desc=f"[{phase}] Epoch {epoch}/{self.epochs}", ncols=200)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            out = self.model(batch, compute_metrics=True)

            loss_exist = out.get("loss_exist")
            loss_attr = out.get("loss_attr")
            loss_contrast = out.get("loss_contrast")

            if loss_exist is None:
                raise ValueError("model must include loss_exist")

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
                    y_seed = batch.y_seed.float().view(-1) if hasattr(batch, "y_seed") and batch.y_seed is not None else None

                    if self.lambda_seed > 0.0 and (y_seed is not None):
                        loss_seed = self.bce_soft(prob_flat, y_seed)

                    if self.lambda_boot > 0.0 and (y_true is not None):
                        y_boot = ((1.0 - self.boot_mix) * y_true + self.boot_mix * y_seed) if y_seed is not None else y_true
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

            if (not train):
                prob = out.get("prob", None)
                y_true_eval = getattr(batch, "y", None)
                if prob is not None and y_true_eval is not None:
                    all_prob.append(prob.detach().view(-1).float().cpu())
                    all_y.append(y_true_eval.detach().view(-1).long().cpu())

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