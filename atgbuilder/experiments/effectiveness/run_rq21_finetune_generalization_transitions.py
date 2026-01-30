# -*- coding: utf-8 -*-
"""
RQ2: Generalization-set (98 apps) with two-branch outputs:
  - NF: no-finetune (evaluate pretrained checkpoint directly)
  - FT: finetune on a small target-domain train split

Protocol (fixed):
  - Split by app (1 graph per app): train/val/test = 0.2/0.1/0.7
  - split_seeds = [42, 123, 2025]
  - sample_reps = [17, 88, 241]  (random negative sampling; deterministic by seed_tag)
  - sampling mode for training/eval edges: random_1to1
  - t*: select on VAL by sweeping thresholds, evaluate TEST at t* and at 0.5

Outputs:
  - experiments_results/<rq2_tag>/
      splits/splitseed_<seed>/{train_apps,val_apps,test_apps}.txt
      splits/splitseed_<seed>/baseline_metrics_on_testapps.json
      splits/splitseed_<seed>/baseline_<tool>_testapps_transitions.txt
      <task>/<branch>/<splitseed_..>/rep_../
          run.log
          sample_signature.json
          threshold.json
          metrics_05.json
          metrics_tstar.json
          predicted_transitions_test_tstar.txt
          predicted_transitions_test_05.txt
          baseline_tool_transition_counts_on_testapps.json
  - results/<rq2_tag>/
      <rq2_tag>_runlevel.csv
      <rq2_tag>_splitlevel_summary.csv
      <rq2_tag>_final_summary.csv
      <rq2_tag>_baseline_splitlevel.csv
      <rq2_tag>_baseline_final.csv

Notes:
  - Graphs in verify_ground_truth_packed_data may NOT include activity names currently.
    This script will output transitions as v<idx> -> v<idx> by default.
  - Baseline tools parsing from all_raw_transitions.txt is included and evaluated vs GT
    using /root/autodl-tmp/atg/gt_transitions_bigger1.txt (Start->* filtered out).
"""

from __future__ import annotations

# ---- bootstrap: allow running from arbitrary cwd (fix: No module named 'src') ----
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# expected layout: <root>/src/experiments/this_file.py
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import csv
import glob
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.embedding.components.seed_atg_builder_packer import DataPacker, GraphConstructionUtils
from src.embedding.config.base_config import BaseConfig
from src.prediction.models.atg_gnn import ATGGNN, compute_binary_metrics
from src.prediction.models.atg_gnn_gin_consistency import ATGGNNGinConsistency
from src.prediction.trainers.trainer_base import BaseTrainer
from src.prediction.trainers.training_logger_utils import TrainingLoggerUtils
from src.prediction.utils.mp_graph_utils import ensure_mp_for_graphs
from src.prediction.utils.training_utils import TrainingUtils


# =========================================================
# Small utils
# =========================================================

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def _dump_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _stable_int_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def graphs_signature_quick(graphs: List[Data], take_k: int = 200) -> int:
    sig = []
    for g in graphs[:take_k]:
        n = int(getattr(g, "num_nodes", 0))
        e = int(g.edge_index.size(1)) if getattr(g, "edge_index", None) is not None else 0
        sig.append((n, e))
    return hash(tuple(sig))


def graph_fingerprint(g: Data) -> str:
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
    if not torch.is_tensor(x):
        return x
    if x.dim() == 0:
        return x
    if x.size(0) == E:
        return x[idx]
    return x


# =========================================================
# Activity name mapping (optional)
# =========================================================

def _get_node_names_from_graph(g: Data) -> Optional[List[str]]:
    """
    Try best-effort extraction of node(activity) names from graph fields.
    If not found, return None.
    """
    candidates = [
        "node_names",
        "activity_names",
        "activities",
        "id2activity",
        "idx2activity",
        "activity_id_to_name",
        "activity_id_map",
        "nodes",
    ]
    for k in candidates:
        if hasattr(g, k):
            v = getattr(g, k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                return v
            if isinstance(v, dict):
                try:
                    items = sorted(v.items(), key=lambda it: int(it[0]))
                    names = [str(name) for _, name in items]
                    return names
                except Exception:
                    pass
    return None


def _node_to_name(node_idx: int, node_names: Optional[List[str]]) -> str:
    if node_names is None:
        return f"v{node_idx}"
    if 0 <= node_idx < len(node_names):
        return str(node_names[node_idx])
    return f"v{node_idx}"


# =========================================================
# NEW: map node embeddings -> activity names via per-app activity npz
# =========================================================

# cache per (task, reduction, app_name)
_APP_ACTIVITY_NPZ_CACHE: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}
_APP_NODE2NAME_CACHE: Dict[Tuple[str, str, str], List[str]] = {}

def _activity_npz_base_dir(task_name: str, reduction_method: str) -> str:
    rm = str(reduction_method).upper()
    return (
        f"/root/autodl-tmp/atg/verify_ground_truth_embedding_data/embeddings/"
        f"fused_embeddings/dimension_reduced/{rm}/activity/{task_name}"
    )

def _resolve_activity_npz_path(task_name: str, reduction_method: str, app_name: str) -> Optional[str]:
    """
    Try to locate the per-app activity npz file.

    Typical:
      .../dimension_reduced/LINEAR/activity/<task>/<full_app_key>.npz

    But your graph app_name may be "24" while file is "*_24.npz".
    So:
      1) exact match <app_name>.npz
      2) suffix match "*_<app_name>.npz"
    """
    base_dir = _activity_npz_base_dir(task_name, reduction_method)
    p_exact = os.path.join(base_dir, f"{app_name}.npz")
    if os.path.exists(p_exact):
        return p_exact

    hits = glob.glob(os.path.join(base_dir, f"*_{app_name}.npz"))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"[WARN] multiple activity npz matched for app_name={app_name}: {hits[:5]} ... choose {hits[0]}")
        return hits[0]
    return None

def _load_app_activity_npz(task_name: str, reduction_method: str, app_name: str) -> Dict[str, np.ndarray]:
    key = (task_name, str(reduction_method).upper(), str(app_name))
    if key in _APP_ACTIVITY_NPZ_CACHE:
        return _APP_ACTIVITY_NPZ_CACHE[key]

    npz_path = _resolve_activity_npz_path(task_name, reduction_method, app_name)
    if not npz_path or (not os.path.exists(npz_path)):
        _APP_ACTIVITY_NPZ_CACHE[key] = {}
        return {}

    z = np.load(npz_path, allow_pickle=True)
    mp = {k: z[k].astype(np.float32) for k in z.files}  # activity_name -> (128,)
    _APP_ACTIVITY_NPZ_CACHE[key] = mp
    return mp

def _extract_node_activity_matrix(g: Data, activity_dim: int) -> Optional[torch.Tensor]:
    """
    Expect g.x is [num_nodes, activity_dim] (often 128).
    If x has more dims, take first activity_dim (best-effort).
    """
    x = getattr(g, "x", None)
    if x is None or (not torch.is_tensor(x)) or x.dim() != 2:
        return None
    if x.size(1) == activity_dim:
        return x
    if x.size(1) > activity_dim:
        return x[:, :activity_dim]
    return None

def match_nodes_to_activity_names(
    g: Data,
    *,
    task_name: str,
    reduction_method: str,
    activity_dim: int,
    sim_warn_thr: float = 0.90,
) -> Optional[List[str]]:
    """
    Map node i -> best matching activity name by cosine nearest in the app's npz dict.
    """
    app = str(getattr(g, "app_name", "NA"))
    key = (task_name, str(reduction_method).upper(), app)
    if key in _APP_NODE2NAME_CACHE:
        return _APP_NODE2NAME_CACHE[key]

    mp = _load_app_activity_npz(task_name, reduction_method, app)
    if not mp:
        return None

    X = _extract_node_activity_matrix(g, activity_dim)
    if X is None:
        print(f"[WARN] cannot extract node activity matrix: app={app}, x={getattr(g,'x',None)}, activity_dim={activity_dim}")
        return None

    names = list(mp.keys())
    A = np.stack([mp[n] for n in names], axis=0)  # [M,128]
    A = torch.from_numpy(A).float()
    X = X.detach().cpu().float()

    # cosine similarity
    A = torch.nn.functional.normalize(A, dim=1)
    X = torch.nn.functional.normalize(X, dim=1)
    sim = X @ A.t()  # [N,M]
    best_idx = sim.argmax(dim=1)          # [N]
    best_sim = sim.max(dim=1).values      # [N]

    out = [names[int(j)] for j in best_idx.tolist()]

    # warn on low similarity (useful to detect mismatch)
    for i, s in enumerate(best_sim.tolist()):
        if s < sim_warn_thr:
            print(f"[WARN] low cos-sim app={app} node={i} sim={s:.4f} (maybe x not pure-activity or embedding mismatch)")

    _APP_NODE2NAME_CACHE[key] = out
    return out


# =========================================================
# Sampling (random 1:1, deterministic per seed_tag)
# =========================================================

def sample_graph_edges_posneg_1to1(
    g: Data,
    *,
    gen: torch.Generator,
    max_neg_when_pos0: int = 0,
    shuffle_selected: bool = True,
) -> Tuple[Data, Dict[str, Any]]:
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
) -> Tuple[List[Data], Dict[str, Any]]:
    gen = torch.Generator()
    gen.manual_seed(_stable_int_hash(seed_tag))

    before = _count_pos_neg(graphs)
    sampled: List[Data] = []

    for g in graphs:
        new_g, _ = sample_graph_edges_posneg_1to1(
            g,
            gen=gen,
            max_neg_when_pos0=max_neg_when_pos0,
            shuffle_selected=True,
        )
        sampled.append(new_g)

    sampled = ensure_mp_for_graphs(sampled, add_reverse=True, add_self_loop=False, only_if_missing=True)
    after = _count_pos_neg(sampled)

    meta = {
        "seed_tag": seed_tag,
        "before": before,
        "after": after,
        "timestamp": _now(),
        "mode": "random_1to1",
    }
    return sampled, meta


# =========================================================
# Prob/threshold utilities
# =========================================================

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
# Baseline tools transitions parsing + GT parsing + evaluation
# =========================================================

TOOLS_7 = ["humanoid", "monkey", "qtest", "stoat", "ape", "fastbot", "scenedroid"]


def parse_all_raw_transitions(path: str) -> Dict[str, Dict[str, Set[Tuple[str, str]]]]:
    data: Dict[str, Dict[str, Set[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(set))
    if not path or (not os.path.exists(path)):
        return data

    cur_app: Optional[str] = None
    cur_tool: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("APK:"):
                apk_path = s.split("APK:", 1)[1].strip()
                base = os.path.basename(apk_path)
                if base.endswith(".apk"):
                    base = base[:-4]
                cur_app = base
                cur_tool = None
                continue
            if s.startswith("Tool:"):
                cur_tool = s.split("Tool:", 1)[1].strip().lower()
                continue
            if "->" in s and cur_app and cur_tool:
                parts = [p.strip() for p in s.split("->", 1)]
                if len(parts) == 2:
                    data[cur_app][cur_tool].add((parts[0], parts[1]))
    return data


def parse_gt_transitions_bigger1(path: str) -> Dict[str, Set[Tuple[str, str]]]:
    gt: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    if not path or (not os.path.exists(path)):
        return gt

    cur_app: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("APK:"):
                apk_path = s.split("APK:", 1)[1].strip()
                base = os.path.basename(apk_path)
                if base.endswith(".apk"):
                    base = base[:-4]
                cur_app = base
                continue

            if s.lower() in ("all transitions", "all results"):
                continue

            if "->" in s and cur_app:
                parts = [p.strip() for p in s.split("->", 1)]
                if len(parts) != 2:
                    continue
                src, dst = parts[0], parts[1]
                if src.strip().lower() == "start":
                    continue
                gt[cur_app].add((src, dst))

    return gt


def _prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_baseline_tools_on_apps(
    *,
    test_apps: List[str],
    gt_edges: Dict[str, Set[Tuple[str, str]]],
    tool_edges: Dict[str, Dict[str, Set[Tuple[str, str]]]],
    tools: List[str],
) -> Dict[str, Dict[str, Any]]:
    res: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        tp = fp = fn = 0
        n_apps = 0
        missing_gt = 0
        missing_tool = 0

        for app in test_apps:
            gt_set = gt_edges.get(app, set())
            pred_set = tool_edges.get(app, {}).get(tool, set())

            if app not in gt_edges:
                missing_gt += 1
            if app not in tool_edges or tool not in tool_edges.get(app, {}):
                missing_tool += 1

            inter = gt_set.intersection(pred_set)
            tp += len(inter)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
            n_apps += 1

        m = _prf_from_counts(tp, fp, fn)
        res[tool] = {
            "apps": n_apps,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "missing_gt_apps": missing_gt,
            "missing_tool_apps": missing_tool,
        }
    return res


def dump_tool_transitions_for_apps(
    *,
    out_path: str,
    test_apps: List[str],
    tool_edges: Dict[str, Dict[str, Set[Tuple[str, str]]]],
    tool: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for app in test_apps:
            f.write(f"APK: {app}\n")
            edges = tool_edges.get(app, {}).get(tool, set())
            if not edges:
                f.write("(empty)\n\n")
                continue
            for (src, dst) in sorted(edges):
                if src.strip().lower() == "start":
                    continue
                f.write(f"{src} -> {dst}\n")
            f.write("\n")


# =========================================================
# Experiment config
# =========================================================

@dataclass
class TrainerParams:
    lr: float = 0.00005
    weight_decay: float = 1e-5
    epochs: int = 100
    batch_size: int = 128
    log_interval: int = 10

    learn_alpha: bool = False
    learn_beta: bool = True
    learn_gamma: bool = False
    init_alpha: float = 1.0
    init_beta: float = 0.5
    init_gamma: float = 0.0

    lambda_seed: float = 0.1
    lambda_boot: float = 0.0
    boot_mix: float = 0.5


@dataclass
class RQ2RunConfig:
    rq_tag: str
    task_name: str
    split_seeds: List[int]
    sample_reps: List[int]
    split_ratios: Tuple[float, float, float] = (0.1, 0.7, 0.2)  # train/val/test
    sampling_mode: str = "random_1to1"
    max_neg_when_pos0: int = 0
    threshold_grid: int = 199

    encoder: str = "gin"
    regularizer: str = "none"  # "none" / "moco" / "fixed_moco"

    pretrained_ckpt: str = ""        # required
    strict_load: bool = True

    finetune_tp: TrainerParams = TrainerParams()

    all_raw_transitions_path: str = "/root/autodl-tmp/atg/all_raw_transitions.txt"
    gt_transitions_path: str = "/root/autodl-tmp/atg/gt_transitions_bigger1.txt"


# =========================================================
# Runner
# =========================================================

class RQ2Runner:

    def __init__(self, config: BaseConfig, cfg: RQ2RunConfig):
        self.config = config
        self.cfg = cfg
        self.root_dir = config.ROOT_DIR

        self.result_dir = os.path.join(self.root_dir, "results", cfg.rq_tag)
        self.exp_root = os.path.join(self.root_dir, "experiments_results", cfg.rq_tag)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.exp_root, exist_ok=True)

        self.logger = TrainingLoggerUtils.setup_logger(
            log_root=os.path.join(self.root_dir, "logs"),
            log_tag=cfg.rq_tag,
            log_class=f"RQ2_{cfg.rq_tag}",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("===== RQ2 Runner: %s =====", cfg.rq_tag)
        self.logger.info("Device: %s", self.device)
        self.logger.info("ROOT_DIR: %s", self.root_dir)
        self.logger.info("task_name: %s", cfg.task_name)
        self.logger.info("split_ratios: %s", str(cfg.split_ratios))
        self.logger.info("split_seeds: %s", str(cfg.split_seeds))
        self.logger.info("sample_reps: %s", str(cfg.sample_reps))
        self.logger.info("pretrained_ckpt: %s", cfg.pretrained_ckpt)

        # will be set in run()
        self.activity_dim: int = 0
        self.widget_dim: int = 0

        # CSV paths
        self.csv_run_path = os.path.join(self.result_dir, f"{cfg.rq_tag}_runlevel.csv")
        self.csv_split_path = os.path.join(self.result_dir, f"{cfg.rq_tag}_splitlevel_summary.csv")
        self.csv_final_path = os.path.join(self.result_dir, f"{cfg.rq_tag}_final_summary.csv")

        self.csv_baseline_split_path = os.path.join(self.result_dir, f"{cfg.rq_tag}_baseline_splitlevel.csv")
        self.csv_baseline_final_path = os.path.join(self.result_dir, f"{cfg.rq_tag}_baseline_final.csv")

        # baseline + gt
        self.baseline = parse_all_raw_transitions(cfg.all_raw_transitions_path)
        self.logger.info("baseline apps parsed = %d (from %s)", len(self.baseline), cfg.all_raw_transitions_path)

        self.gt_edges = parse_gt_transitions_bigger1(cfg.gt_transitions_path)
        self.logger.info("GT apps parsed = %d (from %s)", len(self.gt_edges), cfg.gt_transitions_path)

    def _setup_run_logger(self, run_dir: str) -> logging.Logger:
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")

        logger = logging.getLogger(f"RQ2RunLogger::{run_dir}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

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

    def _build_model(self, activity_dim: int, widget_dim: int) -> torch.nn.Module:
        use_moco = self.cfg.regularizer in ("moco", "fixed_moco")
        if self.cfg.regularizer == "moco":
            moco_mode = "queue"
        elif self.cfg.regularizer == "fixed_moco":
            moco_mode = "fixed"
        else:
            moco_mode = "queue"

        if self.cfg.encoder == "gin":
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

        if self.cfg.encoder == "gcn":
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

        raise ValueError(f"Unknown encoder: {self.cfg.encoder}")

    def _load_pretrained(self, model: torch.nn.Module, ckpt_path: str, strict: bool) -> None:
        if not ckpt_path or (not os.path.exists(ckpt_path)):
            raise FileNotFoundError(f"pretrained_ckpt not found: {ckpt_path}")
        obj = torch.load(ckpt_path, map_location="cpu")
        sd = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        model.load_state_dict(sd, strict=strict)

    @torch.no_grad()
    def _dump_predicted_transitions(
        self,
        *,
        model: torch.nn.Module,
        graphs: List[Data],
        threshold: float,
        out_path: str,
    ) -> None:
        """
        Dump predicted positive transitions on TEST graphs into a txt file.
        Format:
          APK: <app_name>
          src -> tgt
          ...
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        model.eval()

        with open(out_path, "w", encoding="utf-8") as f:
            for g in graphs:
                app = str(getattr(g, "app_name", "NA"))

                # 1) try direct names in graph
                node_names = _get_node_names_from_graph(g)

                # 2) fallback: infer by matching g.x to activity npz vectors
                if node_names is None:
                    reduction = str(getattr(g, "reduction_method", "LINEAR"))
                    node_names = match_nodes_to_activity_names(
                        g,
                        task_name=self.cfg.task_name,
                        reduction_method=reduction,
                        activity_dim=int(self.activity_dim) if int(self.activity_dim) > 0 else 128,
                        sim_warn_thr=0.90,
                    )

                gg = g.to(self.device)
                out = model(gg, compute_metrics=False)
                prob = out.get("prob", None)
                if prob is None:
                    f.write(f"APK: {app}\n")
                    f.write("(no prob)\n\n")
                    continue

                prob = prob.detach().view(-1).float().cpu()
                ei = gg.edge_index.detach().cpu()
                if ei.size(1) != prob.numel():
                    f.write(f"APK: {app}\n")
                    f.write(f"(shape mismatch: E={ei.size(1)} prob={prob.numel()})\n\n")
                    continue

                keep = (prob >= float(threshold)).nonzero(as_tuple=False).view(-1)
                f.write(f"APK: {app}\n")
                if keep.numel() == 0:
                    f.write("(empty)\n\n")
                    continue

                # -------- NEW: dedup by (src,tgt), keep max prob --------
                best: Dict[Tuple[int, int], float] = {}
                for idx in keep.tolist():
                    s = int(ei[0, idx].item())
                    t = int(ei[1, idx].item())
                    p = float(prob[idx].item())
                    key = (s, t)
                    if (key not in best) or (p > best[key]):
                        best[key] = p

                # sort by prob desc (方便你在线挖掘按置信度调度；不是 topK)
                items = sorted(best.items(), key=lambda kv: kv[1], reverse=True)

                for (s, t), p in items:
                    f.write(
                        f"{_node_to_name(s, node_names)} -> {_node_to_name(t, node_names)} | p={p:.6f}\n"
                    )
                f.write("\n")

    def _dump_app_lists(self, split_seed: int, train: List[Data], val: List[Data], test: List[Data]) -> str:
        out_dir = os.path.join(self.exp_root, "splits", f"splitseed_{split_seed}")
        os.makedirs(out_dir, exist_ok=True)

        def _write(name: str, gs: List[Data]):
            p = os.path.join(out_dir, name)
            with open(p, "w", encoding="utf-8") as f:
                for g in gs:
                    f.write(str(getattr(g, "app_name", "NA")) + "\n")

        _write("train_apps.txt", train)
        _write("val_apps.txt", val)
        _write("test_apps.txt", test)
        return out_dir

    def run(self) -> None:
        cfg = self.cfg

        # Load 98 graphs (verify set)
        packer = DataPacker(self.config, pack_dir_name="verify_ground_truth_packed_data")
        pack_path = os.path.join(packer.pack_dir, f"{cfg.task_name}_LINEAR")
        self.logger.info("📦 Loading verify graphs from: %s", pack_path)

        graphs_all: List[Data] = []
        graphs_all = packer.load_packed_graphs(pack_path)
        self.logger.info("✅ loaded graphs=%d | quick_sig=%s", len(graphs_all), str(graphs_signature_quick(graphs_all)))
        if not graphs_all:
            self.logger.error("No graphs loaded. Abort.")
            return

        graphs_all = ensure_mp_for_graphs(graphs_all, add_reverse=True, add_self_loop=False, only_if_missing=True)

        activity_dim, widget_dim = GraphConstructionUtils.get_dimensions(graphs_all)
        self.activity_dim = int(activity_dim)
        self.widget_dim = int(widget_dim)
        self.logger.info("📐 activity_dim=%d widget_dim=%d", activity_dim, widget_dim)

        # CSV writers (model)
        run_csv = open(self.csv_run_path, "w", newline="", encoding="utf-8")
        run_w = csv.writer(run_csv)
        run_w.writerow([
            "task", "branch", "split_seed", "sample_rep",
            "best_epoch",
            "t_star",
            "val_f1_tstar",
            "test_p_tstar", "test_r_tstar", "test_f1_tstar",
            "test_p_05", "test_r_05", "test_f1_05",
        ])

        split_csv = open(self.csv_split_path, "w", newline="", encoding="utf-8")
        split_w = csv.writer(split_csv)
        split_w.writerow([
            "task", "branch", "split_seed",
            "t_star_mean", "t_star_std",
            "test_f1_tstar_mean", "test_f1_tstar_std",
            "test_f1_05_mean", "test_f1_05_std",
        ])

        final_csv = open(self.csv_final_path, "w", newline="", encoding="utf-8")
        final_w = csv.writer(final_csv)
        final_w.writerow([
            "task", "branch",
            "split_seeds", "sample_reps",
            "t_star_mean", "t_star_std",
            "test_f1_tstar_mean", "test_f1_tstar_std",
            "test_f1_05_mean", "test_f1_05_std",
        ])

        # CSV writers (baseline tools)
        baseline_split_csv = open(self.csv_baseline_split_path, "w", newline="", encoding="utf-8")
        baseline_split_w = csv.writer(baseline_split_csv)
        baseline_split_w.writerow([
            "task", "split_seed", "tool",
            "apps", "tp", "fp", "fn",
            "precision", "recall", "f1",
            "missing_gt_apps", "missing_tool_apps",
        ])

        baseline_final_csv = open(self.csv_baseline_final_path, "w", newline="", encoding="utf-8")
        baseline_final_w = csv.writer(baseline_final_csv)
        baseline_final_w.writerow([
            "task", "tool",
            "split_seeds",
            "precision_mean", "precision_std",
            "recall_mean", "recall_std",
            "f1_mean", "f1_std",
        ])

        baseline_collect: Dict[str, Dict[str, List[float]]] = {
            t: {"precision": [], "recall": [], "f1": []} for t in TOOLS_7
        }

        branches = ["NF", "FT"]  # no-finetune, finetune

        # collect across splits for final summary (rep-mean per split)
        final_collect: Dict[str, Dict[str, List[float]]] = {
            b: {"t_star": [], "test_f1_tstar": [], "test_f1_05": []} for b in branches
        }

        for split_seed in cfg.split_seeds:
            self.logger.info("")
            self.logger.info("=======================================")
            self.logger.info("🧩 split_seed=%d  (only affects split)", split_seed)
            self.logger.info("=======================================")

            train_raw, val_raw, test_raw = TrainingUtils.split_graphs(
                graphs_all,
                train_ratio=cfg.split_ratios[0],
                val_ratio=cfg.split_ratios[1],
                test_ratio=cfg.split_ratios[2],
                seed=split_seed,
            )

            train_raw = ensure_mp_for_graphs(train_raw, add_reverse=True, add_self_loop=False, only_if_missing=True)
            val_raw = ensure_mp_for_graphs(val_raw, add_reverse=True, add_self_loop=False, only_if_missing=True)
            test_raw = ensure_mp_for_graphs(test_raw, add_reverse=True, add_self_loop=False, only_if_missing=True)

            split_out_dir = self._dump_app_lists(split_seed, train_raw, val_raw, test_raw)

            self.logger.info("split sizes: train=%d val=%d test=%d", len(train_raw), len(val_raw), len(test_raw))
            self.logger.info("SIG train=%s val=%s test=%s",
                             split_signature(train_raw), split_signature(val_raw), split_signature(test_raw))

            # ===== baseline tools vs GT on this split's TEST apps (micro) =====
            test_apps = [str(getattr(g, "app_name", "NA")) for g in test_raw]
            base_metrics = evaluate_baseline_tools_on_apps(
                test_apps=test_apps,
                gt_edges=self.gt_edges,
                tool_edges=self.baseline,
                tools=TOOLS_7,
            )

            _dump_json(os.path.join(split_out_dir, "baseline_metrics_on_testapps.json"), {
                "timestamp": _now(),
                "task": cfg.task_name,
                "split_seed": split_seed,
                "n_test_apps": len(test_apps),
                "tools": base_metrics,
                "note": "Micro-averaged over test apps; Start->* filtered in GT and tool outputs.",
            })

            for tool in TOOLS_7:
                dump_tool_transitions_for_apps(
                    out_path=os.path.join(split_out_dir, f"baseline_{tool}_testapps_transitions.txt"),
                    test_apps=test_apps,
                    tool_edges=self.baseline,
                    tool=tool,
                )

            for tool, m in base_metrics.items():
                baseline_split_w.writerow([
                    cfg.task_name, split_seed, tool,
                    int(m.get("apps", 0)),
                    int(m.get("tp", 0)), int(m.get("fp", 0)), int(m.get("fn", 0)),
                    float(m.get("precision", 0.0)),
                    float(m.get("recall", 0.0)),
                    float(m.get("f1", 0.0)),
                    int(m.get("missing_gt_apps", 0)),
                    int(m.get("missing_tool_apps", 0)),
                ])
                baseline_collect[tool]["precision"].append(float(m.get("precision", 0.0)))
                baseline_collect[tool]["recall"].append(float(m.get("recall", 0.0)))
                baseline_collect[tool]["f1"].append(float(m.get("f1", 0.0)))

            # per split summaries (mean±std over reps) per branch
            per_branch_rep: Dict[str, Dict[str, List[float]]] = {
                b: {"t_star": [], "test_f1_tstar": [], "test_f1_05": []} for b in branches
            }

            for rep in cfg.sample_reps:
                # deterministic sampling tags
                tag_train = f"{cfg.task_name}|split={split_seed}|rep={rep}|phase=train|sampling={cfg.sampling_mode}"
                tag_val = f"{cfg.task_name}|split={split_seed}|rep={rep}|phase=val|sampling={cfg.sampling_mode}"
                tag_test = f"{cfg.task_name}|split={split_seed}|rep={rep}|phase=test|sampling={cfg.sampling_mode}"

                if cfg.sampling_mode == "random_1to1":
                    train_s, train_meta = sample_graphs_1to1(train_raw, seed_tag=tag_train, max_neg_when_pos0=cfg.max_neg_when_pos0)
                    val_s, val_meta = sample_graphs_1to1(val_raw, seed_tag=tag_val, max_neg_when_pos0=cfg.max_neg_when_pos0)
                    test_s, test_meta = sample_graphs_1to1(test_raw, seed_tag=tag_test, max_neg_when_pos0=cfg.max_neg_when_pos0)
                else:
                    raise ValueError(f"Unknown sampling_mode: {cfg.sampling_mode}")

                for branch in branches:

                    run_dir = os.path.join(
                        self.exp_root,
                        cfg.task_name,
                        branch,
                        f"splitseed_{split_seed}",
                        f"rep_{rep}",
                    )
                    run_logger = self._setup_run_logger(run_dir)

                    run_logger.info("===== RQ2 RUN =====")
                    run_logger.info("task=%s branch=%s split_seed=%d rep=%d", cfg.task_name, branch, split_seed, rep)
                    run_logger.info("RAW_COUNT train=%s", str(_count_pos_neg(train_raw)))
                    run_logger.info("RAW_COUNT val  =%s", str(_count_pos_neg(val_raw)))
                    run_logger.info("RAW_COUNT test =%s", str(_count_pos_neg(test_raw)))
                    run_logger.info("SAMPLED_COUNT train=%s", str(train_meta["after"]))
                    run_logger.info("SAMPLED_COUNT val  =%s", str(val_meta["after"]))
                    run_logger.info("SAMPLED_COUNT test =%s", str(test_meta["after"]))

                    _dump_json(os.path.join(run_dir, "sample_signature.json"), {
                        "train": train_meta, "val": val_meta, "test": test_meta,
                        "timestamp": _now(),
                        "note": "Sampling is deterministic by seed_tag.",
                    })

                    # ===== Build model + load pretrained =====
                    model = self._build_model(activity_dim, widget_dim)
                    self._load_pretrained(model, cfg.pretrained_ckpt, strict=cfg.strict_load)
                    run_logger.info("✅ loaded pretrained ckpt: %s (strict=%s)", cfg.pretrained_ckpt, str(cfg.strict_load))

                    best_epoch = -1

                    # ===== Branch FT: finetune =====
                    if branch == "FT":
                        tp = cfg.finetune_tp
                        trainer = BaseTrainer(
                            model=model,
                            device=self.device,
                            lr=tp.lr,
                            weight_decay=tp.weight_decay,
                            epochs=tp.epochs,
                            batch_size=tp.batch_size,
                            log_interval=tp.log_interval,
                            learn_alpha=tp.learn_alpha,
                            init_alpha=tp.init_alpha,
                            init_beta=tp.init_beta,
                            init_gamma=tp.init_gamma,
                            learn_beta=tp.learn_beta,
                            learn_gamma=tp.learn_gamma,
                            lambda_seed=tp.lambda_seed,
                            lambda_boot=tp.lambda_boot,
                            boot_mix=tp.boot_mix,
                            save_dir=run_dir,
                        )

                        stats = trainer.train(
                            train_graphs=train_s,
                            val_graphs=val_s,
                            test_graphs=test_s,
                            run_name=f"RQ2_{cfg.rq_tag}_{cfg.task_name}_{branch}_split{split_seed}_rep{rep}",
                        )
                        best_epoch = _safe_int(stats.get("best_epoch", -1), -1)
                        model = trainer.model

                        _dump_json(os.path.join(run_dir, "finetune_stats.json"), {
                            "timestamp": _now(),
                            "best_epoch": best_epoch,
                            "weights_at_best_val": stats.get("weights_at_best_val", {}),
                            "note": "Trainer metrics are at threshold=0.5; post-hoc t* is computed below.",
                        })

                    # ===== Evaluate (NF / FT both) =====
                    val_probs, val_labels = collect_probs_labels(model, val_s, device=self.device, batch_size=128)
                    test_probs, test_labels = collect_probs_labels(model, test_s, device=self.device, batch_size=128)

                    test_m_05 = metrics_from_probs_labels(test_probs, test_labels, threshold=0.5)

                    t_star, _ = find_best_f1_threshold_on_val(val_probs, val_labels, num_grid=cfg.threshold_grid)
                    val_m_t = metrics_from_probs_labels(val_probs, val_labels, threshold=t_star)
                    test_m_t = metrics_from_probs_labels(test_probs, test_labels, threshold=t_star)

                    _dump_json(os.path.join(run_dir, "threshold.json"), {
                        "timestamp": _now(),
                        "t_star": float(t_star),
                        "strategy": "best_f1_on_val",
                        "threshold_grid": int(cfg.threshold_grid),
                    })
                    _dump_json(os.path.join(run_dir, "metrics_05.json"), {
                        "timestamp": _now(),
                        "threshold": 0.5,
                        "test": test_m_05,
                    })
                    _dump_json(os.path.join(run_dir, "metrics_tstar.json"), {
                        "timestamp": _now(),
                        "t_star": float(t_star),
                        "val_at_t_star": val_m_t,
                        "test_at_t_star": test_m_t,
                    })

                    # dump predicted transitions on TEST (NOW with activity names if possible)
                    self._dump_predicted_transitions(
                        model=model,
                        graphs=test_s,
                        threshold=float(t_star),
                        out_path=os.path.join(run_dir, "predicted_transitions_test_tstar.txt"),
                    )
                    self._dump_predicted_transitions(
                        model=model,
                        graphs=test_s,
                        threshold=0.5,
                        out_path=os.path.join(run_dir, "predicted_transitions_test_05.txt"),
                    )

                    # baseline coverage counts dump (always)
                    baseline_summary = {}
                    for g in test_raw:
                        app = str(getattr(g, "app_name", "NA"))
                        baseline_summary[app] = {tool: len(self.baseline.get(app, {}).get(tool, set())) for tool in TOOLS_7}
                    _dump_json(os.path.join(run_dir, "baseline_tool_transition_counts_on_testapps.json"), {
                        "timestamp": _now(),
                        "note": "Counts only (per test app). Metrics vs GT are produced at split level in results/*.csv.",
                        "counts": baseline_summary,
                    })

                    # log + CSV row
                    run_logger.info(
                        "RESULT %s split=%d rep=%d | t*=%.6f | VAL_F1@t*=%.6f | TEST_F1@t*=%.6f | TEST_F1@0.5=%.6f",
                        branch, split_seed, rep, float(t_star),
                        _safe_float(val_m_t.get("f1", 0.0)),
                        _safe_float(test_m_t.get("f1", 0.0)),
                        _safe_float(test_m_05.get("f1", 0.0))
                    )

                    run_w.writerow([
                        cfg.task_name, branch, split_seed, rep,
                        best_epoch,
                        float(t_star),
                        _safe_float(val_m_t.get("f1", 0.0)),
                        _safe_float(test_m_t.get("precision", 0.0)),
                        _safe_float(test_m_t.get("recall", 0.0)),
                        _safe_float(test_m_t.get("f1", 0.0)),
                        _safe_float(test_m_05.get("precision", 0.0)),
                        _safe_float(test_m_05.get("recall", 0.0)),
                        _safe_float(test_m_05.get("f1", 0.0)),
                    ])

                    per_branch_rep[branch]["t_star"].append(float(t_star))
                    per_branch_rep[branch]["test_f1_tstar"].append(_safe_float(test_m_t.get("f1", 0.0)))
                    per_branch_rep[branch]["test_f1_05"].append(_safe_float(test_m_05.get("f1", 0.0)))

            # split-level summary per branch (mean±std over reps)
            for branch in branches:
                t_m, t_s = _mean_std(per_branch_rep[branch]["t_star"])
                f_t_m, f_t_s = _mean_std(per_branch_rep[branch]["test_f1_tstar"])
                f_05_m, f_05_s = _mean_std(per_branch_rep[branch]["test_f1_05"])

                split_w.writerow([
                    cfg.task_name, branch, split_seed,
                    t_m, t_s,
                    f_t_m, f_t_s,
                    f_05_m, f_05_s,
                ])

                final_collect[branch]["t_star"].append(float(t_m))
                final_collect[branch]["test_f1_tstar"].append(float(f_t_m))
                final_collect[branch]["test_f1_05"].append(float(f_05_m))

        # final summary across split seeds (rep-mean per split -> mean±std over splits)
        for branch in branches:
            t_all_m, t_all_s = _mean_std(final_collect[branch]["t_star"])
            f_t_all_m, f_t_all_s = _mean_std(final_collect[branch]["test_f1_tstar"])
            f_05_all_m, f_05_all_s = _mean_std(final_collect[branch]["test_f1_05"])

            final_w.writerow([
                cfg.task_name, branch,
                json.dumps(cfg.split_seeds), json.dumps(cfg.sample_reps),
                t_all_m, t_all_s,
                f_t_all_m, f_t_all_s,
                f_05_all_m, f_05_all_s,
            ])

            self.logger.info(
                "📊 FINAL [%s][%s] TEST F1@t* %.6f±%.6f | TEST F1@0.5 %.6f±%.6f | t* %.6f±%.6f",
                cfg.task_name, branch, f_t_all_m, f_t_all_s, f_05_all_m, f_05_all_s, t_all_m, t_all_s
            )

        # baseline final summary across split seeds
        for tool in TOOLS_7:
            p_m, p_s = _mean_std(baseline_collect[tool]["precision"])
            r_m, r_s = _mean_std(baseline_collect[tool]["recall"])
            f_m, f_s = _mean_std(baseline_collect[tool]["f1"])
            baseline_final_w.writerow([
                cfg.task_name, tool,
                json.dumps(cfg.split_seeds),
                p_m, p_s, r_m, r_s, f_m, f_s,
            ])

        run_csv.close()
        split_csv.close()
        final_csv.close()
        baseline_split_csv.close()
        baseline_final_csv.close()

        self.logger.info("✅ Run-level CSV: %s", self.csv_run_path)
        self.logger.info("✅ Split-level summary CSV: %s", self.csv_split_path)
        self.logger.info("✅ Final summary CSV: %s", self.csv_final_path)
        self.logger.info("✅ Baseline split CSV: %s", self.csv_baseline_split_path)
        self.logger.info("✅ Baseline final CSV: %s", self.csv_baseline_final_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="ab_complete", help="task name, e.g., ab_complete")
    p.add_argument("--rq2_tag", type=str, default=None, help="tag for outputs; default auto timestamp")
    p.add_argument("--pretrained_ckpt", type=str, required=True, help="pretrained checkpoint (.pt).")
    p.add_argument("--encoder", type=str, default="gin", choices=["gin", "gcn"])
    p.add_argument("--regularizer", type=str, default="fixed_moco", choices=["none", "moco", "fixed_moco"])
    p.add_argument("--strict_load", action="store_true", help="strict load_state_dict (default True).")
    p.add_argument("--non_strict_load", action="store_true", help="if set, use strict=False")
    return p.parse_args()


def main():
    args = parse_args()
    config = BaseConfig.from_yaml()

    rq2_tag = args.rq2_tag or f"rq2_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    strict = True
    if args.non_strict_load:
        strict = False
    if args.strict_load:
        strict = True

    run_cfg = RQ2RunConfig(
        rq_tag=rq2_tag,
        task_name=args.task,
        split_seeds=[42, 123, 2025],
        sample_reps=[17, 88, 241],
        split_ratios=(0.7, 0.1, 0.2),
        sampling_mode="random_1to1",
        max_neg_when_pos0=0,
        threshold_grid=199,
        encoder=args.encoder,
        regularizer=args.regularizer,
        pretrained_ckpt=args.pretrained_ckpt,
        strict_load=strict,
        finetune_tp=TrainerParams(
            lr=0.00005,
            weight_decay=1e-5,
            epochs=100,
            batch_size=128,
            log_interval=10,
            learn_alpha=False,
            learn_beta=True,
            learn_gamma=False,
            init_alpha=1.0,
            init_beta=0.5,
            init_gamma=0.0,
            lambda_seed=0.1,
            lambda_boot=0.0,
            boot_mix=0.5,
        ),
        all_raw_transitions_path="/root/autodl-tmp/atg/all_raw_transitions.txt",
        gt_transitions_path="/root/autodl-tmp/atg/gt_transitions_bigger1.txt",
    )

    runner = RQ2Runner(config, run_cfg)
    runner.run()


if __name__ == "__main__":
    main()