# -*- coding: utf-8 -*-
"""
RQ1 Common Runner Base

Provides:
- split_seed only controls dataset split (train/val/test)
- per-graph 1:1 sampling for train/val/test (deterministic by split_seed + sample_rep + phase)
- post-hoc threshold t* selected on VAL, then evaluate TEST at t*
- also evaluate TEST at 0.5 (fixed threshold)
- full artifact dumping per run + run-level CSV + split-level summary + final summary
- ✅ best_model checkpoint dumped into ONE shared directory per RQ:
    experiments_results/<rq_tag>/best_models/

Notes:
- We intentionally DO NOT set global random seeds for training.
- Sampling is deterministic and variant-independent to ensure fair ablations.
"""

from __future__ import annotations

import csv
import json
import os
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.embedding.config.base_config import BaseConfig
from src.embedding.components.seed_atg_builder_packer import DataPacker, GraphConstructionUtils

from src.prediction.models.atg_gnn import ATGGNN, compute_binary_metrics
from src.prediction.models.atg_gnn_gin_consistency import ATGGNNGinConsistency
from src.prediction.trainers.trainer_base import BaseTrainer
from src.prediction.trainers.training_logger_utils import TrainingLoggerUtils
from src.prediction.utils.training_utils import TrainingUtils
from src.prediction.utils.mp_graph_utils import ensure_mp_for_graphs


# =========================================================
# Small utils
# =========================================================

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dump_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    arr = np.array(xs, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(xs) >= 2 else 0.0
    return mean, std


def _fmt_metrics(m: Dict[str, Any]) -> str:
    p = _safe_float(m.get("precision", 0.0))
    r = _safe_float(m.get("recall", 0.0))
    f1 = _safe_float(m.get("f1", 0.0))
    acc = _safe_float(m.get("accuracy", 0.0))
    return f"P={p:.4f}, R={r:.4f}, F1={f1:.4f}, Acc={acc:.4f}"


def _stable_int_hash(s: str) -> int:
    """Stable hash -> 32-bit positive int, good for generator.manual_seed."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def _safe_name(s: str) -> str:
    """
    Sanitize a string to be filesystem-friendly.
    """
    if s is None:
        return "NA"
    s = str(s)
    for ch in [" ", "/", "\\", ":", ";", "|", "\n", "\t", "=", ",", "{", "}", "[", "]", "(", ")", "<", ">", "\"", "'"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def graphs_signature_quick(graphs: List[Data], take_k: int = 200) -> int:
    sig = []
    for g in graphs[:take_k]:
        n = int(getattr(g, "num_nodes", 0))
        e = int(g.edge_index.size(1)) if getattr(g, "edge_index", None) is not None else 0
        sig.append((n, e))
    return hash(tuple(sig))


def graph_fingerprint(g: Data) -> str:
    """Lightweight fingerprint per graph: (num_nodes, num_edges, md5 of first 512 edges)."""
    if getattr(g, "edge_index", None) is None:
        return f"{int(getattr(g, 'num_nodes', 0))}_0_md5=none"
    ei = g.edge_index.detach().cpu().numpy()
    take = min(ei.shape[1], 512)
    blob = ei[:, :take].tobytes()
    h = hashlib.md5(blob).hexdigest()
    return f"{int(getattr(g, 'num_nodes', 0))}_{int(ei.shape[1])}_{h}"


def split_signature(graphs: List[Data]) -> str:
    fps = [graph_fingerprint(g) for g in graphs]
    fps.sort()
    return hashlib.md5(("|".join(fps)).encode("utf-8")).hexdigest()


def _count_pos_neg(graphs: List[Data]) -> Dict[str, Any]:
    tot_pos, tot_neg, tot_e = 0, 0, 0
    n_graphs = 0
    n_pos0 = 0
    for g in graphs:
        y = getattr(g, "y", None)
        ei = getattr(g, "edge_index", None)
        if y is None or ei is None:
            continue
        yv = y.view(-1)
        E = int(yv.numel())
        pos = int((yv > 0).sum().item())
        neg = int(E - pos)
        tot_pos += pos
        tot_neg += neg
        tot_e += E
        n_graphs += 1
        if pos == 0:
            n_pos0 += 1
    return {
        "graphs": n_graphs,
        "edges": tot_e,
        "pos": tot_pos,
        "neg": tot_neg,
        "pos_rate": float(tot_pos / max(tot_e, 1)),
        "pos0_graphs": n_pos0,
    }


def _slice_edge_tensor(x: Any, idx: torch.Tensor, E: int) -> Any:
    """Slice tensor-like edge attributes by idx if shape matches [E, ...]."""
    if not torch.is_tensor(x):
        return x
    if x.dim() == 0:
        return x
    if x.size(0) == E:
        return x[idx]
    return x


# =========================================================
# Sampling (deterministic per split_seed + sample_rep + phase)
# =========================================================

def sample_graph_edges_posneg_1to1(
    g: Data,
    *,
    gen: torch.Generator,
    max_neg_when_pos0: int = 0,
    shuffle_selected: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Per-graph 1:1 sampling:
    - Keep all positive edges
    - Sample the same number of negative edges
    - If pos==0:
        - max_neg_when_pos0==0 -> empty edge set
        - else keep up to max_neg_when_pos0 negative edges

    Returns:
        new_g: sampled graph
        rec: sampling record (counts + a hash for selected indices)
    """
    if g is None:
        return g, {"skip": True}
    if getattr(g, "edge_index", None) is None or getattr(g, "y", None) is None:
        return g, {"skip": True}

    y = g.y.view(-1)
    E = int(y.numel())
    if E == 0:
        return g, {"E_raw": 0, "pos_raw": 0, "neg_raw": 0, "sel_total": 0}

    pos_idx = (y > 0).nonzero(as_tuple=False).view(-1)
    neg_idx = (y <= 0).nonzero(as_tuple=False).view(-1)

    pos_n = int(pos_idx.numel())
    neg_n = int(neg_idx.numel())

    if pos_n == 0:
        if max_neg_when_pos0 <= 0:
            sel = pos_idx  # empty
        else:
            k = min(neg_n, int(max_neg_when_pos0))
            if k <= 0:
                sel = pos_idx
            else:
                perm = torch.randperm(neg_n, generator=gen)[:k]
                sel = neg_idx[perm]
    else:
        k = min(pos_n, neg_n)
        if k <= 0:
            sel = pos_idx
        else:
            perm = torch.randperm(neg_n, generator=gen)[:k]
            sel = torch.cat([pos_idx, neg_idx[perm]], dim=0)

    if int(sel.numel()) > 0 and shuffle_selected:
        sel = sel[torch.randperm(int(sel.numel()), generator=gen)]

    new_g = g.clone()

    if int(sel.numel()) == 0:
        new_g.edge_index = new_g.edge_index[:, :0]
    else:
        new_g.edge_index = new_g.edge_index[:, sel]

    # slice edge-level attributes only
    EDGE_KEYS_EXACT = {"y", "y_seed", "y_boot", "edge_attr", "edge_weight", "edge_type"}
    for kname in list(new_g.keys()):
        if kname in ("edge_index", "mp_edge_index"):
            continue
        v = getattr(new_g, kname)
        is_edge_like = (kname in EDGE_KEYS_EXACT) or kname.startswith("edge_") or kname.startswith("y_")
        if not is_edge_like:
            continue
        setattr(new_g, kname, _slice_edge_tensor(v, sel, E))

    # mp_edge_index must be rebuilt based on sampled edge_index
    if hasattr(new_g, "mp_edge_index"):
        new_g.mp_edge_index = None

    neg_sel_n = max(0, int(sel.numel()) - pos_n) if pos_n > 0 else int(sel.numel())
    rec = {
        "E_raw": E,
        "pos_raw": pos_n,
        "neg_raw": neg_n,
        "sel_total": int(sel.numel()),
        "pos_kept": int(min(pos_n, int(sel.numel()))),
        "neg_sampled": int(neg_sel_n),
        "sel_md5": hashlib.md5(sel.detach().cpu().numpy().tobytes()).hexdigest() if int(sel.numel()) > 0 else "empty",
    }
    return new_g, rec


def sample_graphs_1to1(
    graphs: List[Data],
    *,
    seed_tag: str,
    max_neg_when_pos0: int = 0,
) -> Tuple[List[Data], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sample a list of graphs with deterministic generator based on seed_tag.

    Returns:
        sampled_graphs
        summary_meta (counts before/after, signature)
        per_graph_records (for audit)
    """
    gen = torch.Generator()
    gen.manual_seed(_stable_int_hash(seed_tag))

    before = _count_pos_neg(graphs)
    sampled: List[Data] = []
    records: List[Dict[str, Any]] = []

    for g in graphs:
        gid = graph_fingerprint(g)
        new_g, rec = sample_graph_edges_posneg_1to1(
            g,
            gen=gen,
            max_neg_when_pos0=max_neg_when_pos0,
            shuffle_selected=True,
        )
        rec["graph_id"] = gid
        sampled.append(new_g)
        records.append(rec)

    sampled = ensure_mp_for_graphs(sampled, add_reverse=True, add_self_loop=False, only_if_missing=True)
    after = _count_pos_neg(sampled)

    sig = hashlib.md5(
        ("|".join([f"{r.get('graph_id','')}:{r.get('sel_md5','')}" for r in records])).encode("utf-8")
    ).hexdigest()

    meta = {
        "seed_tag": seed_tag,
        "before": before,
        "after": after,
        "sample_signature": sig,
        "timestamp": _now(),
    }
    return sampled, meta, records


def sample_graph_edges_posneg_hard_same_source_1to1(
    g: Data,
    *,
    gen: torch.Generator,
    max_neg_when_pos0: int = 0,
    shuffle_selected: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
    """
    Hard 1:1 negatives:
    - Keep all positive edges
    - Negatives: preferentially sample unlabeled edges whose SOURCE node appears in positive sources
      (harder than uniform random, still cheap + deterministic).
    """
    if g is None:
        return g, {"skip": True}
    if getattr(g, "edge_index", None) is None or getattr(g, "y", None) is None:
        return g, {"skip": True}

    y = g.y.view(-1)
    E = int(y.numel())
    if E == 0:
        return g, {"E_raw": 0, "pos_raw": 0, "neg_raw": 0, "sel_total": 0}

    pos_idx = (y > 0).nonzero(as_tuple=False).view(-1)
    neg_idx = (y <= 0).nonzero(as_tuple=False).view(-1)

    pos_n = int(pos_idx.numel())
    neg_n = int(neg_idx.numel())

    if pos_n == 0:
        if max_neg_when_pos0 <= 0:
            sel = pos_idx
        else:
            k = min(neg_n, int(max_neg_when_pos0))
            if k <= 0:
                sel = pos_idx
            else:
                perm = torch.randperm(neg_n, generator=gen)[:k]
                sel = neg_idx[perm]
    else:
        k = min(pos_n, neg_n)
        if k <= 0:
            sel = pos_idx
        else:
            ei = g.edge_index
            pos_src = ei[0, pos_idx]
            pos_src_set = set(pos_src.detach().cpu().numpy().tolist())

            neg_src = ei[0, neg_idx].detach().cpu().numpy().tolist()
            hard_mask = torch.tensor([s in pos_src_set for s in neg_src], device=neg_idx.device, dtype=torch.bool)
            hard_pool = neg_idx[hard_mask]

            # sample from hard pool first, then fill with random
            if int(hard_pool.numel()) > 0:
                k_hard = min(k, int(hard_pool.numel()))
                perm_h = torch.randperm(int(hard_pool.numel()), generator=gen)[:k_hard]
                neg_sel = hard_pool[perm_h]

                if k_hard < k:
                    # fill remaining by random from all neg (best-effort without removing duplicates)
                    perm_r = torch.randperm(neg_n, generator=gen)[: (k - k_hard)]
                    neg_fill = neg_idx[perm_r]
                    neg_sel = torch.cat([neg_sel, neg_fill], dim=0)
            else:
                perm = torch.randperm(neg_n, generator=gen)[:k]
                neg_sel = neg_idx[perm]

            sel = torch.cat([pos_idx, neg_sel], dim=0)

    if int(sel.numel()) > 0 and shuffle_selected:
        sel = sel[torch.randperm(int(sel.numel()), generator=gen)]

    new_g = g.clone()
    if int(sel.numel()) == 0:
        new_g.edge_index = new_g.edge_index[:, :0]
    else:
        new_g.edge_index = new_g.edge_index[:, sel]

    EDGE_KEYS_EXACT = {"y", "y_seed", "y_boot", "edge_attr", "edge_weight", "edge_type"}
    for kname in list(new_g.keys()):
        if kname in ("edge_index", "mp_edge_index"):
            continue
        v = getattr(new_g, kname)
        is_edge_like = (kname in EDGE_KEYS_EXACT) or kname.startswith("edge_") or kname.startswith("y_")
        if not is_edge_like:
            continue
        setattr(new_g, kname, _slice_edge_tensor(v, sel, E))

    if hasattr(new_g, "mp_edge_index"):
        new_g.mp_edge_index = None

    neg_sel_n = max(0, int(sel.numel()) - pos_n) if pos_n > 0 else int(sel.numel())
    rec = {
        "mode": "hard_same_source_1to1",
        "E_raw": E,
        "pos_raw": pos_n,
        "neg_raw": neg_n,
        "sel_total": int(sel.numel()),
        "pos_kept": int(min(pos_n, int(sel.numel()))),
        "neg_sampled": int(neg_sel_n),
        "sel_md5": hashlib.md5(sel.detach().cpu().numpy().tobytes()).hexdigest() if int(sel.numel()) > 0 else "empty",
    }
    return new_g, rec

def sample_graphs_hard_same_source_1to1(
    graphs: List[Data],
    *,
    seed_tag: str,
    max_neg_when_pos0: int = 0,
) -> Tuple[List[Data], Dict[str, Any], List[Dict[str, Any]]]:
    gen = torch.Generator()
    gen.manual_seed(_stable_int_hash(seed_tag))

    before = _count_pos_neg(graphs)
    sampled: List[Data] = []
    records: List[Dict[str, Any]] = []

    for g in graphs:
        gid = graph_fingerprint(g)
        new_g, rec = sample_graph_edges_posneg_hard_same_source_1to1(
            g,
            gen=gen,
            max_neg_when_pos0=max_neg_when_pos0,
            shuffle_selected=True,
        )
        rec["graph_id"] = gid
        sampled.append(new_g)
        records.append(rec)

    sampled = ensure_mp_for_graphs(sampled, add_reverse=True, add_self_loop=False, only_if_missing=True)
    after = _count_pos_neg(sampled)

    sig = hashlib.md5(
        ("|".join([f"{r.get('graph_id','')}:{r.get('sel_md5','')}" for r in records])).encode("utf-8")
    ).hexdigest()

    meta = {
        "seed_tag": seed_tag,
        "before": before,
        "after": after,
        "sample_signature": sig,
        "timestamp": _now(),
        "mode": "hard_same_source_1to1",
    }
    return sampled, meta, records

# =========================================================
# Prob/threshold utilities
# =========================================================

def _summarize_probs_labels(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    if probs.size == 0:
        return {
            "n": 0, "pos_rate": 0.0,
            "zero_ratio": 0.0,
            "min": None, "max": None,
            "q01": None, "q25": None, "q50": None, "q75": None, "q99": None,
        }
    pos_rate = float(labels.mean()) if labels.size > 0 else 0.0
    zero_ratio = float((probs == 0).mean())
    qs = np.quantile(probs, [0.01, 0.25, 0.50, 0.75, 0.99]).astype(np.float64)
    return {
        "n": int(probs.size),
        "pos_rate": pos_rate,
        "zero_ratio": zero_ratio,
        "min": float(np.min(probs)),
        "max": float(np.max(probs)),
        "q01": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q99": float(qs[4]),
    }


@torch.no_grad()
def collect_probs_labels(
    model: torch.nn.Module,
    graphs: List[Data],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch, compute_metrics=False)
        prob = out.get("prob", None)
        y = getattr(batch, "y", None)
        if prob is None or y is None:
            continue
        probs_all.append(prob.detach().view(-1).float().cpu().numpy())
        labels_all.append(y.detach().view(-1).long().cpu().numpy())

    if not probs_all:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

    return np.concatenate(probs_all, axis=0), np.concatenate(labels_all, axis=0)


def metrics_from_probs_labels(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    if probs.size == 0 or labels.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    y_prob = torch.from_numpy(probs).float()
    y_true = torch.from_numpy(labels).long()
    return compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(threshold))


def find_best_f1_threshold_on_val(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    num_grid: int = 199,
) -> Tuple[float, Dict[str, Any]]:
    if val_probs.size == 0 or val_labels.size == 0:
        m = metrics_from_probs_labels(val_probs, val_labels, 0.5)
        return 0.5, m

    thrs = np.linspace(0.001, 0.999, num_grid, dtype=np.float64)
    best_thr = 0.5
    best_f1 = -1.0
    best_m: Dict[str, Any] = {}

    for t in thrs:
        m = metrics_from_probs_labels(val_probs, val_labels, float(t))
        f1 = float(m.get("f1", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
            best_m = m

    return best_thr, best_m


# =========================================================
# Experiment definitions
# =========================================================

@dataclass
class TrainerParams:
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 128
    log_interval: int = 10

    # loss weights learning
    learn_alpha: bool = False
    learn_beta: bool = True
    learn_gamma: bool = True
    init_alpha: float = 1.0
    init_beta: float = 0.5
    init_gamma: float = 0.1

    # seed/boot (default off)
    lambda_seed: float = 0.0
    lambda_boot: float = 0.0
    boot_mix: float = 0.5


@dataclass
class Variant:
    id: str
    encoder: str = "gin"              # "gcn" or "gin"
    regularizer: str = "fixed_moco"   # "none" / "moco" / "fixed_moco"
    sampling: str = "random_1to1"   # ✅ "none" / "random_1to1" / "hard_same_source_1to1"
    trainer: Optional[TrainerParams] = None


@dataclass
class RQ1RunConfig:
    rq_tag: str
    tasks: List[str]
    split_seeds: List[int]
    sample_reps: List[int]
    variants: List[Variant]
    pack_dir_name: str = "train_packed_data"
    method_suffix: str = "LINEAR"
    split_ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2)
    max_neg_when_pos0: int = 0
    threshold_grid: int = 199


# =========================================================
# Runner
# =========================================================

class RQ1Runner:
    """
    High-level runner that:
    - loads graphs per task
    - loops split_seed and sample_rep
    - samples train/val/test deterministically (variant-independent)
    - trains each variant
    - dumps artifacts and CSVs
    - ✅ saves best_model checkpoint into ONE shared directory (per RQ):
        experiments_results/<rq_tag>/best_models/
    """

    def __init__(self, config: BaseConfig, run_cfg: RQ1RunConfig):
        self.config = config
        self.run_cfg = run_cfg
        self.root_dir = config.ROOT_DIR

        # output roots
        self.result_dir = os.path.join(self.root_dir, "results", run_cfg.rq_tag)
        self.exp_root = os.path.join(self.root_dir, "experiments_results", run_cfg.rq_tag)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.exp_root, exist_ok=True)

        # ✅ shared best-model directory (ALL best models live here)
        self.best_models_dir = os.path.join(self.exp_root, "best_models")
        os.makedirs(self.best_models_dir, exist_ok=True)

        # global logger (high-level)
        self.logger = TrainingLoggerUtils.setup_logger(
            log_root=os.path.join(self.root_dir, "logs"),
            log_tag=run_cfg.rq_tag,
            log_class=f"RQ1_{run_cfg.rq_tag}",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("===== RQ1 Runner: %s =====", run_cfg.rq_tag)
        self.logger.info("Device: %s", self.device)
        self.logger.info("ROOT_DIR: %s", self.root_dir)
        self.logger.info("BEST_MODELS_DIR: %s", self.best_models_dir)

        # CSV paths
        self.csv_run_path = os.path.join(self.result_dir, f"{run_cfg.rq_tag}_runlevel.csv")
        self.csv_split_path = os.path.join(self.result_dir, f"{run_cfg.rq_tag}_splitlevel_summary.csv")
        self.csv_final_path = os.path.join(self.result_dir, f"{run_cfg.rq_tag}_final_summary.csv")

    # -------------------------
    # Model factory
    # -------------------------
    def _build_model(self, encoder: str, regularizer: str, activity_dim: int, widget_dim: int) -> torch.nn.Module:
        use_moco = regularizer in ("moco", "fixed_moco")
        if regularizer == "moco":
            moco_mode = "queue"
        elif regularizer == "fixed_moco":
            moco_mode = "fixed"
        else:
            moco_mode = "queue"

        if encoder == "gin":
            return ATGGNNGinConsistency(
                activity_dim=activity_dim,
                widget_dim=widget_dim,
                hidden_dim=128,
                embed_dim=128,
                dropout=0.0,
                alpha=1.0, beta=1.0, gamma=1.0,
                use_attr_head=True,
                use_moco=use_moco,
                moco_mode=moco_mode,
            ).to(self.device)

        if encoder == "gcn":
            return ATGGNN(
                activity_dim=activity_dim,
                widget_dim=widget_dim,
                hidden_dim=128,
                embed_dim=128,
                dropout=0.0,
                alpha=1.0, beta=1.0, gamma=1.0,
                use_attr_head=True,
                use_moco=use_moco,
                moco_mode=moco_mode,
            ).to(self.device)

        raise ValueError(f"Unknown encoder: {encoder}")

    # -------------------------
    # Per-run logger writing to run_dir/train.log
    # -------------------------
    def _setup_run_logger(self, run_dir: str) -> logging.Logger:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")

        logger = logging.getLogger(f"RQ1RunLogger::{run_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # clear handlers if reused
        if logger.handlers:
            for h in list(logger.handlers):
                logger.removeHandler(h)

        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    # -------------------------
    # Artifact dump helpers
    # -------------------------
    def _dump_threshold(self, path: str, threshold: float, strategy: str, extra: Optional[Dict[str, Any]] = None) -> None:
        obj = {
            "threshold": float(threshold),
            "strategy": strategy,
            "timestamp": _now(),
        }
        if extra:
            obj.update(extra)
        _dump_json(path, obj)

    def _write_history_jsonl(self, path: str, stats: Dict[str, Any]) -> None:
        """
        Best-effort: if trainer returns epoch history, dump it as jsonl.
        If not, still dump a minimal record for traceability.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hist = stats.get("history", None)

        with open(path, "w", encoding="utf-8") as f:
            if isinstance(hist, list) and hist:
                for row in hist:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                minimal = {
                    "timestamp": _now(),
                    "best_epoch": _safe_int(stats.get("best_epoch", -1), -1),
                    "best_val": stats.get("best_val", {}),
                    "test": stats.get("test", {}),
                    "weights_at_best_val": stats.get("weights_at_best_val", {}),
                    "note": "No epoch-level history returned by trainer; this is a minimal fallback record.",
                }
                f.write(json.dumps(minimal, ensure_ascii=False) + "\n")

    # -------------------------
    # ✅ Save best model to ONE shared directory
    # -------------------------
    def _best_model_basename(
        self,
        *,
        rq_tag: str,
        task_name: str,
        variant_id: str,
        encoder: str,
        regularizer: str,
        split_seed: int,
        sample_rep: int,
    ) -> str:
        """
        Construct a unique filename base (no extension) for best model checkpoint/meta.
        """
        parts = [
            f"rq={_safe_name(rq_tag)}",
            f"task={_safe_name(task_name)}",
            f"var={_safe_name(variant_id)}",
            f"enc={_safe_name(encoder)}",
            f"reg={_safe_name(regularizer)}",
            f"split={int(split_seed)}",
            f"rep={int(sample_rep)}",
        ]
        return "bestmodel__" + "__".join(parts)

    def save_best_model_to_shared_dir(
        self,
        *,
        model: torch.nn.Module,
        rq_tag: str,
        task_name: str,
        variant_id: str,
        encoder: str,
        regularizer: str,
        activity_dim: int,
        widget_dim: int,
        split_seed: int,
        sample_rep: int,
        best_epoch: int,
        weights_at_best: Dict[str, Any],
        run_dir: str,
    ) -> Dict[str, str]:
        """
        Save best model checkpoint into self.best_models_dir with a unique name.
        Also save meta JSON next to it.
        Additionally, write a small reference file inside run_dir to point to shared checkpoint.
        """
        os.makedirs(self.best_models_dir, exist_ok=True)

        base = self._best_model_basename(
            rq_tag=rq_tag,
            task_name=task_name,
            variant_id=variant_id,
            encoder=encoder,
            regularizer=regularizer,
            split_seed=split_seed,
            sample_rep=sample_rep,
        )

        ckpt_path = os.path.join(self.best_models_dir, base + ".pt")
        meta_path = os.path.join(self.best_models_dir, base + ".meta.json")

        # 1) checkpoint: state_dict only (portable & safe)
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        # 2) meta: reconstruction + auditing
        meta = {
            "timestamp": _now(),
            "rq_tag": rq_tag,
            "task": task_name,
            "variant_id": variant_id,
            "encoder": encoder,
            "regularizer": regularizer,
            "dims": {"activity_dim": activity_dim, "widget_dim": widget_dim},
            "split_seed": int(split_seed),
            "sample_rep": int(sample_rep),
            "best_epoch": int(best_epoch),
            "weights_at_best_val": weights_at_best,
            "note": "Load by rebuilding the same model (encoder/regularizer/dims), then load_state_dict from ckpt.",
        }
        _dump_json(meta_path, meta)

        # 3) reference file in run_dir (so each run can find its shared model quickly)
        ref = {
            "timestamp": _now(),
            "best_model_ckpt": ckpt_path,
            "best_model_meta": meta_path,
        }
        _dump_json(os.path.join(run_dir, "best_model_ref.json"), ref)

        return {"ckpt": ckpt_path, "meta": meta_path}

    # -------------------------
    # Main loop
    # -------------------------
    def run(self) -> None:
        cfg = self.run_cfg

        # CSV writers
        run_csv = open(self.csv_run_path, "w", newline="", encoding="utf-8")
        run_w = csv.writer(run_csv)
        run_w.writerow([
            "task", "variant_id", "encoder", "regularizer",
            "split_seed", "sample_rep",
            "best_epoch",
            "t_star",
            "val_f1_tstar",
            "test_p_tstar", "test_r_tstar", "test_f1_tstar",
            "test_p_05", "test_r_05", "test_f1_05",
            "alpha_at_best", "beta_at_best", "gamma_at_best",
        ])

        split_csv = open(self.csv_split_path, "w", newline="", encoding="utf-8")
        split_w = csv.writer(split_csv)
        split_w.writerow([
            "task", "variant_id", "encoder", "regularizer",
            "split_seed",
            "t_star_mean", "t_star_std",
            "test_p_tstar_mean", "test_p_tstar_std",
            "test_r_tstar_mean", "test_r_tstar_std",
            "test_f1_tstar_mean", "test_f1_tstar_std",
            "test_p_05_mean", "test_p_05_std",
            "test_r_05_mean", "test_r_05_std",
            "test_f1_05_mean", "test_f1_05_std",
        ])

        final_csv = open(self.csv_final_path, "w", newline="", encoding="utf-8")
        final_w = csv.writer(final_csv)
        final_w.writerow([
            "task", "variant_id", "encoder", "regularizer",
            "split_seeds", "sample_reps",
            "t_star_mean", "t_star_std",
            "test_f1_tstar_mean", "test_f1_tstar_std",
            "test_f1_05_mean", "test_f1_05_std",
        ])

        for task_name in cfg.tasks:
            self.logger.info("")
            self.logger.info("=======================================")
            self.logger.info("🔬 Task: %s", task_name)
            self.logger.info("=======================================")

            packer = DataPacker(self.config, pack_dir_name=cfg.pack_dir_name)
            pack_path = os.path.join(packer.pack_dir, f"{task_name}_{cfg.method_suffix}")
            self.logger.info("📦 Loading graphs from: %s", pack_path)

            all_graphs: List[Data] = packer.load_packed_graphs(pack_path)
            self.logger.info("✅ Loaded graphs=%d | quick_sig=%s", len(all_graphs), str(graphs_signature_quick(all_graphs)))
            if not all_graphs:
                self.logger.warning("Task=%s has no graphs; skip.", task_name)
                continue

            all_graphs = ensure_mp_for_graphs(all_graphs, add_reverse=True, add_self_loop=False, only_if_missing=True)

            activity_dim, widget_dim = GraphConstructionUtils.get_dimensions(all_graphs)
            self.logger.info("📐 Activity dim=%d, Widget dim=%d", activity_dim, widget_dim)

            for var in cfg.variants:
                var_tp = var.trainer or TrainerParams()
                self.logger.info("")
                self.logger.info("========== Variant %s (enc=%s reg=%s) ==========", var.id, var.encoder, var.regularizer)

                per_split_repmean = {"t_star": [], "test_f1_tstar": [], "test_f1_05": []}

                for split_seed in cfg.split_seeds:
                    self.logger.info("")
                    self.logger.info("🧩 split_seed=%d (only affects split)", split_seed)

                    train_full, val_raw, test_raw = TrainingUtils.split_graphs(
                        all_graphs,
                        train_ratio=cfg.split_ratios[0],
                        val_ratio=cfg.split_ratios[1],
                        test_ratio=cfg.split_ratios[2],
                        seed=split_seed,
                    )

                    split_sig = {
                        "train": split_signature(train_full),
                        "val": split_signature(val_raw),
                        "test": split_signature(test_raw),
                        "timestamp": _now(),
                    }

                    rep_collect = {
                        "t_star": [],
                        "test_p_tstar": [], "test_r_tstar": [], "test_f1_tstar": [],
                        "test_p_05": [], "test_r_05": [], "test_f1_05": [],
                    }

                    for rep in cfg.sample_reps:
                        # deterministic tags: include sampling mode so S0/S1/S2 可复现且彼此不同
                        tag_train = f"{task_name}|split={split_seed}|rep={rep}|phase=train|sampling={var.sampling}"
                        tag_val = f"{task_name}|split={split_seed}|rep={rep}|phase=val|sampling={var.sampling}"
                        tag_test = f"{task_name}|split={split_seed}|rep={rep}|phase=test|sampling={var.sampling}"

                        if var.sampling == "none":
                            train_s, val_s, test_s = train_full, val_raw, test_raw
                            train_meta = {"mode": "none", "before": _count_pos_neg(train_full),
                                          "after": _count_pos_neg(train_full), "seed_tag": tag_train,
                                          "timestamp": _now()}
                            val_meta = {"mode": "none", "before": _count_pos_neg(val_raw),
                                        "after": _count_pos_neg(val_raw), "seed_tag": tag_val, "timestamp": _now()}
                            test_meta = {"mode": "none", "before": _count_pos_neg(test_raw),
                                         "after": _count_pos_neg(test_raw), "seed_tag": tag_test, "timestamp": _now()}
                            train_s = ensure_mp_for_graphs(train_s, add_reverse=True, add_self_loop=False,
                                                           only_if_missing=True)
                            val_s = ensure_mp_for_graphs(val_s, add_reverse=True, add_self_loop=False,
                                                         only_if_missing=True)
                            test_s = ensure_mp_for_graphs(test_s, add_reverse=True, add_self_loop=False,
                                                          only_if_missing=True)

                        elif var.sampling == "random_1to1":
                            train_s, train_meta, _ = sample_graphs_1to1(train_full, seed_tag=tag_train,
                                                                        max_neg_when_pos0=cfg.max_neg_when_pos0)
                            val_s, val_meta, _ = sample_graphs_1to1(val_raw, seed_tag=tag_val,
                                                                    max_neg_when_pos0=cfg.max_neg_when_pos0)
                            test_s, test_meta, _ = sample_graphs_1to1(test_raw, seed_tag=tag_test,
                                                                      max_neg_when_pos0=cfg.max_neg_when_pos0)

                        elif var.sampling == "hard_same_source_1to1":
                            train_s, train_meta, _ = sample_graphs_hard_same_source_1to1(train_full, seed_tag=tag_train,
                                                                                         max_neg_when_pos0=cfg.max_neg_when_pos0)
                            val_s, val_meta, _ = sample_graphs_hard_same_source_1to1(val_raw, seed_tag=tag_val,
                                                                                     max_neg_when_pos0=cfg.max_neg_when_pos0)
                            test_s, test_meta, _ = sample_graphs_hard_same_source_1to1(test_raw, seed_tag=tag_test,
                                                                                       max_neg_when_pos0=cfg.max_neg_when_pos0)

                        else:
                            raise ValueError(f"Unknown sampling mode: {var.sampling}")

                        run_dir = os.path.join(
                            self.exp_root,
                            task_name,
                            var.id,
                            var.encoder,
                            var.regularizer,
                            f"splitseed_{split_seed}",
                            f"rep_{rep}",
                        )
                        run_logger = self._setup_run_logger(run_dir)

                        run_logger.info("===== RQ1 Run =====")
                        run_logger.info("task=%s variant=%s encoder=%s regularizer=%s", task_name, var.id, var.encoder, var.regularizer)
                        run_logger.info("split_seed=%d sample_rep=%d", split_seed, rep)
                        run_logger.info("split_signature: %s", json.dumps(split_sig, ensure_ascii=False))
                        run_logger.info("RAW_COUNT train=%s", str(_count_pos_neg(train_full)))
                        run_logger.info("RAW_COUNT val  =%s", str(_count_pos_neg(val_raw)))
                        run_logger.info("RAW_COUNT test =%s", str(_count_pos_neg(test_raw)))
                        run_logger.info("SAMPLED_COUNT train=%s", str(train_meta["after"]))
                        run_logger.info("SAMPLED_COUNT val  =%s", str(val_meta["after"]))
                        run_logger.info("SAMPLED_COUNT test =%s", str(test_meta["after"]))

                        _dump_json(os.path.join(run_dir, "split_signature.json"), split_sig)
                        _dump_json(os.path.join(run_dir, "sample_signature.json"), {
                            "train": train_meta,
                            "val": val_meta,
                            "test": test_meta,
                            "note": "Sampling is deterministic by seed_tag and variant-independent for fair comparison.",
                        })

                        cfg_snapshot = {
                            "timestamp": _now(),
                            "rq_tag": cfg.rq_tag,
                            "task": task_name,
                            "variant": {"id": var.id, "encoder": var.encoder, "regularizer": var.regularizer},
                            "split_seed": split_seed,
                            "sample_rep": rep,
                            "split_ratios": cfg.split_ratios,
                            "sampling": {
                                "mode": "per-graph 1:1 pos:neg",
                                "max_neg_when_pos0": cfg.max_neg_when_pos0,
                                "seed_tags": {"train": tag_train, "val": tag_val, "test": tag_test},
                            },
                            "trainer_params": asdict(var_tp),
                            "dims": {"activity_dim": activity_dim, "widget_dim": widget_dim},
                        }
                        _dump_json(os.path.join(run_dir, "config_snapshot.json"), cfg_snapshot)

                        model = self._build_model(var.encoder, var.regularizer, activity_dim, widget_dim)

                        trainer = BaseTrainer(
                            model=model,
                            device=self.device,
                            lr=var_tp.lr,
                            weight_decay=var_tp.weight_decay,
                            epochs=var_tp.epochs,
                            batch_size=var_tp.batch_size,
                            log_interval=var_tp.log_interval,
                            learn_alpha=var_tp.learn_alpha,
                            init_alpha=var_tp.init_alpha,
                            init_beta=var_tp.init_beta,
                            init_gamma=var_tp.init_gamma,
                            learn_beta=var_tp.learn_beta,
                            learn_gamma=var_tp.learn_gamma,
                            lambda_seed=var_tp.lambda_seed,
                            lambda_boot=var_tp.lambda_boot,
                            boot_mix=var_tp.boot_mix,
                            save_dir=run_dir,
                        )

                        stats = trainer.train(
                            train_graphs=train_s,
                            val_graphs=val_s,
                            test_graphs=test_s,
                            run_name=f"{task_name}_{cfg.rq_tag}_{var.id}_splitseed_{split_seed}_rep_{rep}",
                        )

                        best_epoch = _safe_int(stats.get("best_epoch", -1), -1)
                        best_w = stats.get("weights_at_best_val", {}) or {}
                        best_model = trainer.model

                        run_logger.info("📌 best_epoch=%d", best_epoch)
                        run_logger.info("📌 weights_at_best: α=%.4f β=%.4f γ=%.4f",
                                        _safe_float(best_w.get("alpha", 1.0), 1.0),
                                        _safe_float(best_w.get("beta", 0.0), 0.0),
                                        _safe_float(best_w.get("gamma", 0.0), 0.0))

                        # ✅ dump best_model to ONE shared directory
                        ref = self.save_best_model_to_shared_dir(
                            model=best_model,
                            rq_tag=cfg.rq_tag,
                            task_name=task_name,
                            variant_id=var.id,
                            encoder=var.encoder,
                            regularizer=var.regularizer,
                            activity_dim=activity_dim,
                            widget_dim=widget_dim,
                            split_seed=split_seed,
                            sample_rep=rep,
                            best_epoch=best_epoch,
                            weights_at_best=best_w,
                            run_dir=run_dir,
                        )
                        run_logger.info("✅ best_model saved: %s", ref["ckpt"])

                        # collect probs/labels
                        val_probs, val_labels = collect_probs_labels(best_model, val_s, device=self.device, batch_size=var_tp.batch_size)
                        test_probs, test_labels = collect_probs_labels(best_model, test_s, device=self.device, batch_size=var_tp.batch_size)

                        # fixed threshold 0.5
                        val_m_05 = metrics_from_probs_labels(val_probs, val_labels, threshold=0.5)
                        test_m_05 = metrics_from_probs_labels(test_probs, test_labels, threshold=0.5)

                        # t* selected on VAL
                        t_star, val_m_t = find_best_f1_threshold_on_val(val_probs, val_labels, num_grid=cfg.threshold_grid)
                        test_m_t = metrics_from_probs_labels(test_probs, test_labels, threshold=t_star)

                        run_logger.info("✅ VAL@0.5 : %s", _fmt_metrics(val_m_05))
                        run_logger.info("✅ TEST@0.5: %s", _fmt_metrics(test_m_05))
                        run_logger.info("✅ t*=%.4f | VAL@t*: %s | TEST@t*: %s", t_star, _fmt_metrics(val_m_t), _fmt_metrics(test_m_t))

                        prob_sum = {
                            "val": _summarize_probs_labels(val_probs, val_labels),
                            "test": _summarize_probs_labels(test_probs, test_labels),
                        }
                        _dump_json(os.path.join(run_dir, "prob_summary.json"), prob_sum)

                        _dump_json(os.path.join(run_dir, "metrics_05.json"), {
                            "timestamp": _now(),
                            "threshold": 0.5,
                            "val": val_m_05,
                            "test": test_m_05,
                        })
                        self._dump_threshold(
                            os.path.join(run_dir, "threshold.json"),
                            threshold=t_star,
                            strategy="best_f1_on_val(1to1)",
                            extra={"threshold_grid": cfg.threshold_grid},
                        )
                        _dump_json(os.path.join(run_dir, "metrics_tstar.json"), {
                            "timestamp": _now(),
                            "t_star": t_star,
                            "val_at_t_star": val_m_t,
                            "test_at_t_star": test_m_t,
                            "weights_at_best_val": best_w,
                            "best_model_ref": ref,
                        })

                        self._write_history_jsonl(os.path.join(run_dir, "history.jsonl"), stats)

                        run_w.writerow([
                            task_name, var.id, var.encoder, var.regularizer,
                            split_seed, rep,
                            best_epoch,
                            t_star,
                            _safe_float(val_m_t.get("f1", 0.0)),
                            _safe_float(test_m_t.get("precision", 0.0)),
                            _safe_float(test_m_t.get("recall", 0.0)),
                            _safe_float(test_m_t.get("f1", 0.0)),
                            _safe_float(test_m_05.get("precision", 0.0)),
                            _safe_float(test_m_05.get("recall", 0.0)),
                            _safe_float(test_m_05.get("f1", 0.0)),
                            _safe_float(best_w.get("alpha", 1.0), 1.0),
                            _safe_float(best_w.get("beta", 0.0), 0.0),
                            _safe_float(best_w.get("gamma", 0.0), 0.0),
                        ])

                        rep_collect["t_star"].append(float(t_star))
                        rep_collect["test_p_tstar"].append(_safe_float(test_m_t.get("precision", 0.0)))
                        rep_collect["test_r_tstar"].append(_safe_float(test_m_t.get("recall", 0.0)))
                        rep_collect["test_f1_tstar"].append(_safe_float(test_m_t.get("f1", 0.0)))
                        rep_collect["test_p_05"].append(_safe_float(test_m_05.get("precision", 0.0)))
                        rep_collect["test_r_05"].append(_safe_float(test_m_05.get("recall", 0.0)))
                        rep_collect["test_f1_05"].append(_safe_float(test_m_05.get("f1", 0.0)))

                    # split-level summary (mean±std over reps)
                    t_m, t_s = _mean_std(rep_collect["t_star"])
                    p_t_m, p_t_s = _mean_std(rep_collect["test_p_tstar"])
                    r_t_m, r_t_s = _mean_std(rep_collect["test_r_tstar"])
                    f_t_m, f_t_s = _mean_std(rep_collect["test_f1_tstar"])
                    p_05_m, p_05_s = _mean_std(rep_collect["test_p_05"])
                    r_05_m, r_05_s = _mean_std(rep_collect["test_r_05"])
                    f_05_m, f_05_s = _mean_std(rep_collect["test_f1_05"])

                    split_w.writerow([
                        task_name, var.id, var.encoder, var.regularizer,
                        split_seed,
                        t_m, t_s,
                        p_t_m, p_t_s,
                        r_t_m, r_t_s,
                        f_t_m, f_t_s,
                        p_05_m, p_05_s,
                        r_05_m, r_05_s,
                        f_05_m, f_05_s,
                    ])

                    per_split_repmean["t_star"].append(float(t_m))
                    per_split_repmean["test_f1_tstar"].append(float(f_t_m))
                    per_split_repmean["test_f1_05"].append(float(f_05_m))

                # final summary across split_seeds (rep-mean per split -> mean±std over splits)
                t_all_m, t_all_s = _mean_std(per_split_repmean["t_star"])
                f_t_all_m, f_t_all_s = _mean_std(per_split_repmean["test_f1_tstar"])
                f_05_all_m, f_05_all_s = _mean_std(per_split_repmean["test_f1_05"])

                final_w.writerow([
                    task_name, var.id, var.encoder, var.regularizer,
                    json.dumps(cfg.split_seeds), json.dumps(cfg.sample_reps),
                    t_all_m, t_all_s,
                    f_t_all_m, f_t_all_s,
                    f_05_all_m, f_05_all_s,
                ])

                self.logger.info(
                    "📊 FINAL [%s][%s] TEST F1@t* %.4f±%.4f | TEST F1@0.5 %.4f±%.4f | t* %.4f±%.4f",
                    task_name, var.id, f_t_all_m, f_t_all_s, f_05_all_m, f_05_all_s, t_all_m, t_all_s
                )

        run_csv.close()
        split_csv.close()
        final_csv.close()

        self.logger.info("✅ Run-level CSV: %s", self.csv_run_path)
        self.logger.info("✅ Split-level summary CSV: %s", self.csv_split_path)
        self.logger.info("✅ Final summary CSV: %s", self.csv_final_path)
        self.logger.info("✅ Best models saved to: %s", self.best_models_dir)