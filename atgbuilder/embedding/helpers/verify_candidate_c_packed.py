# -*- coding: utf-8 -*-
"""
Verify packed PyG graphs for Candidate-set C design.
+ Optional: evaluate GT coverage of Candidate edges by loading external GT file (no pkl modification).

Usage:
  python -m src.embedding.helpers.verify_candidate_c_packed \
    --pack-dir /path/to/packed_dir \
    --expected-task ab_complete \
    --expected-method LINEAR \
    --check-meta --require-meta \
    --cand-recall-thr 0.98 \
    --gt-file /root/autodl-tmp/atg/gt_transitions_bigger1.txt \
    --gt-recall-thr 0.80 \
    --dump-csv /tmp/c_verify.csv
"""

import argparse
import os
import pickle
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch


Transition = Tuple[str, str]


# ----------------------------
# GT utils (no pkl change)
# ----------------------------
def _norm_act(s: str) -> str:
    return (s or "").strip()


def _app_keys_from_apk_path(apk_path: str) -> Set[str]:
    """
    From '.../com.xxx.yyy_123.apk' generate multiple keys to align with Data.app_name variations.
    """
    base = os.path.splitext(os.path.basename(apk_path.strip()))[0]  # com.xxx_123
    keys = {base}

    m = re.match(r"^(.*)_(\d+)$", base)
    if m:
        keys.add(m.group(1))

    keys.add(apk_path.strip())
    keys.add(os.path.basename(apk_path.strip()))
    return keys


def load_gt_transitions_bigger1(gt_path: str) -> Dict[str, Set[Transition]]:
    """
    Parse gt_transitions_bigger1.txt:
      APK: <path>
      All results
      A -> B
      Start -> C
    Return: key -> set((srcAct, dstAct))
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)

    gt: Dict[str, Set[Transition]] = {}
    cur_keys: Optional[Set[str]] = None

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("APK:"):
                apk_path = line[len("APK:"):].strip()
                cur_keys = _app_keys_from_apk_path(apk_path)
                for k in cur_keys:
                    gt.setdefault(k, set())
                continue

            if line in ("All Transitions", "All results"):
                continue
            if line.startswith("..."):
                continue

            if "->" in line and cur_keys:
                parts = [p.strip() for p in line.split("->", 1)]
                if len(parts) != 2:
                    continue
                src, dst = _norm_act(parts[0]), _norm_act(parts[1])
                if src.lower() == "start":
                    continue
                if not src or not dst:
                    continue
                for k in cur_keys:
                    gt[k].add((src, dst))

    gt = {k: v for k, v in gt.items() if len(v) > 0}
    return gt


def _extract_node_names(g: Any) -> Optional[List[str]]:
    """
    Best-effort: try to get node_id -> activity_name list.
    """
    for attr in ("node_names", "activity_names", "activity_set", "id2activity", "id2name"):
        v = getattr(g, attr, None)
        if v is None:
            continue

        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]

        if isinstance(v, dict):
            # id -> name dict
            if all(isinstance(k, int) for k in v.keys()):
                max_k = max(v.keys()) if v else -1
                out = [""] * (max_k + 1)
                for k, name in v.items():
                    kk = int(k)
                    if 0 <= kk < len(out):
                        out[kk] = str(name)
                return out

    return None


def _graph_app_keys(g: Any) -> Set[str]:
    """
    Try to infer multiple app keys from g.app_name / g.apk_path / g.apk_name.
    """
    keys: Set[str] = set()
    for attr in ("app_name", "apk_name", "apk_path"):
        v = getattr(g, attr, None)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        keys.add(s)
        base = os.path.splitext(os.path.basename(s))[0]
        if base:
            keys.add(base)
            m = re.match(r"^(.*)_(\d+)$", base)
            if m:
                keys.add(m.group(1))
    if not keys:
        keys.add("unknown_app")
    return keys


def _gt_stats_for_graph(g: Any, gt_map: Dict[str, Set[Transition]]) -> Dict[str, Any]:
    """
    Compute GT coverage of Candidate edges (edge_index) for this graph:
      gt_recall_in_C = |GT ∩ C| / |GT|
    Requires node_names mapping to translate node ids to activity names.

    Returns dict with:
      gt_hit_key, gt_edges, gt_in_C, gt_recall_in_C, ok(bool), reason(str)
    """
    keys = _graph_app_keys(g)
    gt_edges: Optional[Set[Transition]] = None
    hit_key: Optional[str] = None
    for k in keys:
        if k in gt_map:
            gt_edges = gt_map[k]
            hit_key = k
            break

    if not gt_edges:
        return {"ok": False, "reason": "no_gt_key", "gt_hit_key": None, "gt_edges": None, "gt_in_C": None, "gt_recall_in_C": None}

    node_names = _extract_node_names(g)
    if node_names is None or len(node_names) == 0:
        return {"ok": False, "reason": "no_node_names", "gt_hit_key": hit_key, "gt_edges": len(gt_edges), "gt_in_C": None, "gt_recall_in_C": None}

    ei = getattr(g, "edge_index", None)
    if ei is None or (not isinstance(ei, torch.Tensor)) or ei.dim() != 2 or ei.size(0) != 2:
        return {"ok": False, "reason": "no_edge_index", "gt_hit_key": hit_key, "gt_edges": len(gt_edges), "gt_in_C": None, "gt_recall_in_C": None}

    id2name = [str(x) for x in node_names]
    E = int(ei.size(1))
    if E == 0:
        return {"ok": True, "reason": "empty_C", "gt_hit_key": hit_key, "gt_edges": len(gt_edges), "gt_in_C": 0, "gt_recall_in_C": 0.0}

    # iterate edges on CPU for speed/compat
    src_ids = ei[0].detach().cpu().numpy().astype(np.int64)
    dst_ids = ei[1].detach().cpu().numpy().astype(np.int64)

    gt_in_c = 0
    for i in range(E):
        si = int(src_ids[i]); di = int(dst_ids[i])
        if si < 0 or di < 0 or si >= len(id2name) or di >= len(id2name):
            continue
        sname = _norm_act(id2name[si])
        dname = _norm_act(id2name[di])
        if (sname, dname) in gt_edges:
            gt_in_c += 1

    gt_cnt = len(gt_edges)
    rec = float(gt_in_c / (gt_cnt + 1e-12)) if gt_cnt > 0 else 0.0
    return {"ok": True, "reason": "ok", "gt_hit_key": hit_key, "gt_edges": gt_cnt, "gt_in_C": gt_in_c, "gt_recall_in_C": rec}


# ----------------------------
# Existing utils
# ----------------------------
def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(a, q))


def _to_edge_tuples(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
    return list(zip(src.tolist(), dst.tolist()))


def _edge_set(edge_index: torch.Tensor) -> set:
    return set(_to_edge_tuples(edge_index))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _is_valid_edge_index(ei: Any) -> bool:
    return isinstance(ei, torch.Tensor) and ei.dim() == 2 and ei.size(0) == 2


def _hash_edge_index(edge_index: Optional[torch.Tensor]) -> str:
    if edge_index is None or (not isinstance(edge_index, torch.Tensor)):
        return "NONE"
    ei_cpu = edge_index.detach().cpu()
    if ei_cpu.numel() == 0:
        return "EMPTY"
    ei_np = ei_cpu.numpy().astype(np.int64)
    order = np.lexsort((ei_np[1], ei_np[0]))
    ei_np = ei_np[:, order]
    return str(hash(ei_np.tobytes()))


def _get_pos_edges_from_graph(g: Any) -> Tuple[Optional[torch.Tensor], str]:
    if hasattr(g, "seed_edge_index") and getattr(g, "seed_edge_index") is not None:
        ei = getattr(g, "seed_edge_index")
        if _is_valid_edge_index(ei):
            return ei, "seed_edge_index"

    if hasattr(g, "pos_edge_index") and getattr(g, "pos_edge_index") is not None:
        ei = getattr(g, "pos_edge_index")
        if _is_valid_edge_index(ei):
            return ei, "pos_edge_index"

    if hasattr(g, "gt_edge_index") and getattr(g, "gt_edge_index") is not None:
        gt_ei = getattr(g, "gt_edge_index")
        if _is_valid_edge_index(gt_ei):
            if hasattr(g, "gt_y") and getattr(g, "gt_y") is not None:
                gt_y = getattr(g, "gt_y")
                if isinstance(gt_y, torch.Tensor) and gt_y.numel() == gt_ei.size(1):
                    pos_mask = (gt_y == 1)
                    return gt_ei[:, pos_mask], "gt_edge_index+gt_y"
            return gt_ei, "gt_edge_index"

    if (
        hasattr(g, "edge_index")
        and getattr(g, "edge_index") is not None
        and hasattr(g, "y")
        and getattr(g, "y") is not None
    ):
        ei = getattr(g, "edge_index")
        y = getattr(g, "y")
        if _is_valid_edge_index(ei) and isinstance(y, torch.Tensor) and y.numel() == ei.size(1):
            pos_mask = (y == 1)
            return ei[:, pos_mask], "edge_index(y==1)"

    return None, "none"


def _compute_recall(pos_ei: torch.Tensor, cand_ei: torch.Tensor) -> Tuple[Optional[float], int]:
    if (pos_ei is None) or (not _is_valid_edge_index(pos_ei)) or pos_ei.numel() == 0:
        return None, 0
    P = _edge_set(pos_ei)
    n_pos = len(P)
    if n_pos == 0:
        return None, 0
    if (cand_ei is None) or (not _is_valid_edge_index(cand_ei)) or cand_ei.numel() == 0:
        return 0.0, n_pos
    C = _edge_set(cand_ei)
    return (len(P.intersection(C)) / float(n_pos)), n_pos


def _collect_app_edge_hash(
    pack_dir: str,
    attr_name: str = "edge_index",
    max_graphs: Optional[int] = None,
) -> Dict[str, str]:
    batch_files = sorted([f for f in os.listdir(pack_dir) if f.endswith(".pkl")])
    out: Dict[str, str] = {}
    total = 0
    for bf in batch_files:
        path = os.path.join(pack_dir, bf)
        with open(path, "rb") as f:
            graphs = pickle.load(f)
        if not isinstance(graphs, list):
            continue
        for gi, g in enumerate(graphs):
            total += 1
            if max_graphs is not None and total > max_graphs:
                return out
            app_name = getattr(g, "app_name", f"unknown_app@{bf}:{gi}")
            ei = getattr(g, attr_name, None)
            out.setdefault(app_name, _hash_edge_index(ei))
    return out


# ----------------------------
# Core verification
# ----------------------------
def verify_pack_dir(
    pack_dir: str,
    expected_task: Optional[str],
    expected_method: Optional[str],
    check_meta: bool,
    require_meta: bool,
    cand_recall_thr: float,
    dump_csv: Optional[str],
    compare_pack_dirs: Optional[List[str]] = None,
    compare_strict: bool = False,
    max_graphs: Optional[int] = None,
    # ===== GT options =====
    gt_file: Optional[str] = None,
    gt_recall_thr: float = 0.0,
) -> int:
    if not os.path.isdir(pack_dir):
        print(f"[FATAL] pack_dir not found: {pack_dir}")
        return 2

    batch_files = sorted([f for f in os.listdir(pack_dir) if f.endswith(".pkl")])
    if not batch_files:
        print(f"[FATAL] no .pkl files under: {pack_dir}")
        return 2

    # Load GT once (optional)
    gt_map: Optional[Dict[str, Set[Transition]]] = None
    if gt_file:
        try:
            gt_map = load_gt_transitions_bigger1(gt_file)
            print(f"[GT] loaded GT transitions: keys={len(gt_map)} from {gt_file}")
        except Exception as e:
            print(f"[GT] failed to load gt_file={gt_file}: {e}")
            gt_map = None

    # Stats collectors
    rec_list: List[float] = []
    knn_only_rec_list: List[float] = []

    e_pred_list: List[int] = []
    e_mp_list: List[int] = []
    knn_e_pred_list: List[int] = []

    out_pred_list: List[float] = []
    out_mp_list: List[float] = []
    pos_rate_list: List[float] = []
    density_list: List[float] = []
    self_loop_rate_pred_list: List[float] = []
    dup_rate_pred_list: List[float] = []

    # GT stats
    gt_rec_list: List[float] = []
    gt_edges_sum = 0
    gt_in_c_sum = 0
    gt_labeled_graphs = 0
    gt_with_key = 0
    bad_gt_recall_apps: List[str] = []
    gt_skip_reason_cnt: Dict[str, int] = {}

    errors: List[str] = []
    bad_recall_apps: List[str] = []

    rows_for_csv: List[Dict[str, Any]] = []

    app_edge_hash: Dict[str, str] = {}
    app_mp_edge_hash: Dict[str, str] = {}

    total_graphs = 0

    for bf in batch_files:
        path = os.path.join(pack_dir, bf)
        with open(path, "rb") as f:
            try:
                graphs = pickle.load(f)
            except Exception as e:
                errors.append(f"[LOAD_FAIL] {bf}: {e}")
                continue

        if not isinstance(graphs, list):
            errors.append(f"[FORMAT_FAIL] {bf}: expected list[Data], got {type(graphs)}")
            continue

        for gi, g in enumerate(graphs):
            total_graphs += 1
            if max_graphs is not None and total_graphs > max_graphs:
                break

            app_name = getattr(g, "app_name", f"unknown_app@{bf}:{gi}")
            task_name = getattr(g, "task_name", None)
            reduction_method = getattr(g, "reduction_method", None)

            # ---- meta check ----
            if check_meta:
                if require_meta and (task_name is None or reduction_method is None):
                    errors.append(
                        f"[META_MISSING] app={app_name} file={bf}:{gi} task={task_name} method={reduction_method}"
                    )
                if expected_task is not None and task_name is not None and task_name != expected_task:
                    errors.append(f"[META_TASK_MISMATCH] app={app_name} got={task_name} expected={expected_task}")
                if expected_method is not None and reduction_method is not None and str(reduction_method).upper() != str(expected_method).upper():
                    errors.append(
                        f"[META_METHOD_MISMATCH] app={app_name} got={reduction_method} expected={expected_method}"
                    )

            # ---- required tensors ----
            if not hasattr(g, "x") or g.x is None:
                errors.append(f"[MISSING_X] app={app_name}")
                continue
            if not hasattr(g, "edge_index") or g.edge_index is None:
                errors.append(f"[MISSING_EDGE_INDEX] app={app_name}")
                continue

            x = g.x
            edge_index = g.edge_index
            mp_edge_index = getattr(g, "mp_edge_index", None)

            ei_hash = _hash_edge_index(edge_index)
            mp_hash = _hash_edge_index(mp_edge_index)

            prev = app_edge_hash.get(app_name)
            if prev is None:
                app_edge_hash[app_name] = ei_hash
            elif prev != ei_hash:
                errors.append(
                    f"[EDGE_INDEX_INCONSISTENT_WITHIN_DIR] app={app_name} prev={prev} now={ei_hash} file={bf}:{gi}"
                )

            prev_mp = app_mp_edge_hash.get(app_name)
            if prev_mp is None:
                app_mp_edge_hash[app_name] = mp_hash
            elif prev_mp != mp_hash:
                errors.append(
                    f"[MP_EDGE_INDEX_INCONSISTENT_WITHIN_DIR] app={app_name} prev={prev_mp} now={mp_hash} file={bf}:{gi}"
                )

            if not isinstance(x, torch.Tensor) or x.dim() != 2:
                errors.append(f"[BAD_X_SHAPE] app={app_name} x.shape={getattr(x, 'shape', None)}")
                continue
            if not _is_valid_edge_index(edge_index):
                errors.append(f"[BAD_EDGE_INDEX_SHAPE] app={app_name} edge_index.shape={getattr(edge_index, 'shape', None)}")
                continue

            N = int(x.size(0))
            E_pred = int(edge_index.size(1))

            if E_pred > 0:
                min_idx = int(edge_index.min().item())
                max_idx = int(edge_index.max().item())
                if min_idx < 0 or max_idx >= N:
                    errors.append(f"[EDGE_INDEX_OOB] app={app_name} N={N} min={min_idx} max={max_idx}")
                    continue

            y = getattr(g, "y", None)
            if y is not None:
                if not isinstance(y, torch.Tensor) or y.dim() != 1 or int(y.numel()) != E_pred:
                    errors.append(f"[BAD_Y_SHAPE] app={app_name} y.shape={getattr(y, 'shape', None)} E_pred={E_pred}")

            edge_attr = getattr(g, "edge_attr", None)
            if edge_attr is not None:
                if not isinstance(edge_attr, torch.Tensor) or edge_attr.dim() != 2 or int(edge_attr.size(0)) != E_pred:
                    errors.append(
                        f"[BAD_EDGE_ATTR_SHAPE] app={app_name} edge_attr.shape={getattr(edge_attr, 'shape', None)} E_pred={E_pred}"
                    )

            # self-loop / dup (candidate edges)
            if E_pred > 0:
                src = edge_index[0]
                dst = edge_index[1]
                self_loop_cnt = int((src == dst).sum().item())
                self_loop_rate = float(self_loop_cnt / max(E_pred, 1))
                unique_edges = torch.unique(edge_index, dim=1)
                E_unique = int(unique_edges.size(1))
                dup_rate = float(1.0 - (E_unique / max(E_pred, 1)))
            else:
                self_loop_rate = 0.0
                dup_rate = 0.0

            self_loop_rate_pred_list.append(self_loop_rate)
            dup_rate_pred_list.append(dup_rate)

            # mp_edge_index stats
            E_mp = 0
            if mp_edge_index is not None:
                if not _is_valid_edge_index(mp_edge_index):
                    errors.append(
                        f"[BAD_MP_EDGE_INDEX_SHAPE] app={app_name} mp_edge_index.shape={getattr(mp_edge_index, 'shape', None)}"
                    )
                else:
                    E_mp = int(mp_edge_index.size(1))
                    if E_mp > 0:
                        mp_min = int(mp_edge_index.min().item())
                        mp_max = int(mp_edge_index.max().item())
                        if mp_min < 0 or mp_max >= N:
                            errors.append(f"[MP_EDGE_INDEX_OOB] app={app_name} N={N} min={mp_min} max={mp_max}")

            # basic metrics
            avg_outdeg_pred = (E_pred / N) if N > 0 else float("nan")
            avg_outdeg_mp = (E_mp / N) if N > 0 else float("nan")
            density = (E_pred / (N * (N - 1))) if N > 1 else float("nan")

            pos_rate = None
            if y is not None and isinstance(y, torch.Tensor) and y.numel() == E_pred and E_pred > 0:
                pos_rate = float(y.float().mean().item())

            # cand_recall (seed coverage)
            pos_ei, pos_src = _get_pos_edges_from_graph(g)
            cand_recall = None
            n_pos = None
            if pos_ei is not None and _is_valid_edge_index(pos_ei):
                cand_recall, n_pos_val = _compute_recall(pos_ei, edge_index)
                n_pos = n_pos_val if n_pos_val > 0 else None
                if cand_recall is not None and cand_recall < cand_recall_thr:
                    bad_recall_apps.append(
                        f"app={app_name} cand_recall={cand_recall:.6f} n_pos={n_pos_val} pos_src={pos_src} "
                        f"N={N} E_pred={E_pred} E_mp={E_mp}"
                    )

            # knn-only recall
            knn_only_recall = None
            knn_e_pred = None
            knn_src = "none"

            knn_edge_index = getattr(g, "knn_edge_index", None)
            if knn_edge_index is not None and _is_valid_edge_index(knn_edge_index):
                knn_src = "knn_edge_index"
                knn_e_pred = int(knn_edge_index.size(1))
                if pos_ei is not None and _is_valid_edge_index(pos_ei):
                    knn_only_recall, _ = _compute_recall(pos_ei, knn_edge_index)

            if knn_only_recall is None:
                if hasattr(g, "knn_only_recall"):
                    knn_only_recall = _safe_float(getattr(g, "knn_only_recall"))
                    if knn_only_recall is not None:
                        knn_src = "knn_only_recall(scalar)"

            # ===== GT coverage of C (optional) =====
            gt_ok = None
            gt_reason = None
            gt_hit_key = None
            gt_edges = None
            gt_in_c = None
            gt_recall_in_c = None

            if gt_map is not None:
                st = _gt_stats_for_graph(g, gt_map)
                gt_ok = bool(st.get("ok", False))
                gt_reason = st.get("reason", None)
                gt_hit_key = st.get("gt_hit_key", None)
                gt_edges = st.get("gt_edges", None)
                gt_in_c = st.get("gt_in_C", None)
                gt_recall_in_c = st.get("gt_recall_in_C", None)

                if gt_reason == "no_gt_key":
                    gt_skip_reason_cnt["no_gt_key"] = gt_skip_reason_cnt.get("no_gt_key", 0) + 1
                elif gt_reason == "no_node_names":
                    gt_skip_reason_cnt["no_node_names"] = gt_skip_reason_cnt.get("no_node_names", 0) + 1
                elif gt_reason == "no_edge_index":
                    gt_skip_reason_cnt["no_edge_index"] = gt_skip_reason_cnt.get("no_edge_index", 0) + 1
                else:
                    # ok / empty_C
                    pass

                if gt_hit_key is not None:
                    gt_with_key += 1
                if gt_ok and (gt_edges is not None) and (gt_in_c is not None):
                    gt_labeled_graphs += 1
                    gt_edges_sum += int(gt_edges)
                    gt_in_c_sum += int(gt_in_c)
                    if gt_recall_in_c is not None:
                        gt_rec_list.append(float(gt_recall_in_c))
                        if gt_recall_thr > 0.0 and float(gt_recall_in_c) < float(gt_recall_thr):
                            bad_gt_recall_apps.append(
                                f"app={app_name} gt_recall_in_C={float(gt_recall_in_c):.6f} gt_edges={gt_edges} gt_in_C={gt_in_c} hit_key={gt_hit_key}"
                            )

            # collect stats
            e_pred_list.append(E_pred)
            e_mp_list.append(E_mp)
            out_pred_list.append(avg_outdeg_pred)
            out_mp_list.append(avg_outdeg_mp)
            density_list.append(density)

            if pos_rate is not None:
                pos_rate_list.append(pos_rate)
            if cand_recall is not None:
                rec_list.append(cand_recall)
            if knn_only_recall is not None:
                knn_only_rec_list.append(float(knn_only_recall))
            if knn_e_pred is not None:
                knn_e_pred_list.append(int(knn_e_pred))

            rows_for_csv.append(
                {
                    "app_name": app_name,
                    "task_name": task_name,
                    "reduction_method": reduction_method,
                    "N": N,
                    "E_pred": E_pred,
                    "E_mp": E_mp,
                    "avg_outdeg_pred": avg_outdeg_pred,
                    "avg_outdeg_mp": avg_outdeg_mp,
                    "density_pred": density,
                    "pos_rate_seed_on_candidates": pos_rate,
                    "cand_recall": cand_recall,
                    "pos_edge_source": pos_src,
                    "n_pos": n_pos,
                    "batch_file": bf,
                    "batch_i": gi,
                    "self_loop_rate_pred": float(self_loop_rate),
                    "dup_rate_pred": float(dup_rate),
                    "knn_only_recall": knn_only_recall,
                    "knn_E_pred": knn_e_pred,
                    "knn_pos_edge_source": knn_src,

                    # ---- GT fields ----
                    "gt_ok": gt_ok,
                    "gt_reason": gt_reason,
                    "gt_hit_key": gt_hit_key,
                    "gt_edges": gt_edges,
                    "gt_in_C": gt_in_c,
                    "gt_recall_in_C": gt_recall_in_c,
                }
            )

        if max_graphs is not None and total_graphs >= max_graphs:
            break

    # ----------------------------
    # Report
    # ----------------------------
    print("=" * 90)
    print("✅ Candidate-C Packed Verification Report")
    print("=" * 90)
    print(f"pack_dir          : {pack_dir}")
    print(f"total_batches     : {len(batch_files)}")
    print(f"total_graphs      : {total_graphs}")
    if expected_task:
        print(f"expected_task     : {expected_task}")
    if expected_method:
        print(f"expected_method   : {expected_method}")
    print(f"check_meta        : {check_meta} (require_meta={require_meta})")
    print(f"cand_recall_thr   : {cand_recall_thr}")
    if gt_file:
        print(f"gt_file           : {gt_file}")
        print(f"gt_recall_thr     : {gt_recall_thr}")
    print("-" * 90)

    def _summ_int(xs: List[int], name: str):
        if not xs:
            print(f"{name:18s}: (none)")
            return
        a = np.asarray(xs, dtype=np.float64)
        print(f"{name:18s}: mean={a.mean():.2f} median={np.median(a):.0f} p95={np.percentile(a, 95):.0f} max={a.max():.0f}")

    def _summ_float(xs: List[float], name: str):
        if not xs:
            print(f"{name:18s}: (none)")
            return
        a = np.asarray(xs, dtype=np.float64)
        print(
            f"{name:18s}: mean={a.mean():.6f} median={np.median(a):.6f} p10={np.percentile(a, 10):.6f} "
            f"p1={np.percentile(a, 1):.6f} min={a.min():.6f}"
        )

    _summ_int(e_pred_list, "E_pred")
    _summ_int(e_mp_list, "E_mp")
    _summ_int(knn_e_pred_list, "E_knn_pred")
    _summ_float(out_pred_list, "avg_outdeg_pred")
    _summ_float(out_mp_list, "avg_outdeg_mp")
    _summ_float(density_list, "density_pred")
    _summ_float(pos_rate_list, "pos_rate(y on C)")
    _summ_float(rec_list, "cand_recall")
    _summ_float(knn_only_rec_list, "knn_only_recall")
    _summ_float(self_loop_rate_pred_list, "self_loop_rate_pred")
    _summ_float(dup_rate_pred_list, "dup_rate_pred")

    if gt_map is not None:
        micro = float(gt_in_c_sum / (gt_edges_sum + 1e-12)) if gt_edges_sum > 0 else 0.0
        print("-" * 90)
        print(f"🧪 GT coverage (no pkl write):")
        print(f"  gt_keys_loaded      : {len(gt_map)}")
        print(f"  graphs_with_gt_key  : {gt_with_key}/{total_graphs}")
        print(f"  graphs_labeled_gt   : {gt_labeled_graphs}/{total_graphs}  (requires node_names & edge_index)")
        print(f"  gt_edges_sum        : {gt_edges_sum}")
        print(f"  gt_in_C_sum         : {gt_in_c_sum}")
        print(f"  gt_recall_micro     : {micro:.6f}")
        _summ_float(gt_rec_list, "gt_recall_in_C")
        if gt_skip_reason_cnt:
            print(f"  gt_skip_reasons     : {gt_skip_reason_cnt}")

    print("-" * 90)
    print(f"❌ Error count     : {len(errors)}")
    if errors:
        for e in errors[:30]:
            print("  " + e)
        if len(errors) > 30:
            print(f"  ... ({len(errors) - 30} more)")

    print("-" * 90)
    print(f"⚠️  cand_recall < {cand_recall_thr} apps: {len(bad_recall_apps)}")
    if bad_recall_apps:
        for s in bad_recall_apps[:50]:
            print("  " + s)
        if len(bad_recall_apps) > 50:
            print(f"  ... ({len(bad_recall_apps) - 50} more)")

    if gt_map is not None and gt_recall_thr > 0.0:
        print("-" * 90)
        print(f"⚠️  gt_recall_in_C < {gt_recall_thr} apps: {len(bad_gt_recall_apps)}")
        if bad_gt_recall_apps:
            for s in bad_gt_recall_apps[:50]:
                print("  " + s)
            if len(bad_gt_recall_apps) > 50:
                print(f"  ... ({len(bad_gt_recall_apps) - 50} more)")

    # dump csv
    if dump_csv:
        try:
            import csv
            if rows_for_csv:
                os.makedirs(os.path.dirname(dump_csv), exist_ok=True) if os.path.dirname(dump_csv) else None
                fieldnames = list(rows_for_csv[0].keys())
                with open(dump_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    rows_sorted = sorted(
                        rows_for_csv,
                        key=lambda r: (str(r.get("app_name", "")), str(r.get("batch_file", "")), int(r.get("batch_i", 0))),
                    )
                    for r in rows_sorted:
                        writer.writerow(r)
                print("-" * 90)
                print(f"💾 CSV saved to: {dump_csv}")
            else:
                print("[WARN] no rows_for_csv to write.")
        except Exception as e:
            print(f"[WARN] failed to write csv: {e}")

    # Cross-pack-dir consistency (unchanged)
    compare_pack_dirs = compare_pack_dirs or []
    if compare_pack_dirs:
        base_edge = app_edge_hash
        base_mp = app_mp_edge_hash

        for other_dir in compare_pack_dirs:
            if not os.path.isdir(other_dir):
                errors.append(f"[COMPARE_DIR_NOT_FOUND] {other_dir}")
                continue

            other_edge = _collect_app_edge_hash(other_dir, attr_name="edge_index", max_graphs=max_graphs)
            other_mp = _collect_app_edge_hash(other_dir, attr_name="mp_edge_index", max_graphs=max_graphs)

            common = set(base_edge.keys()) & set(other_edge.keys())
            mismatch = [a for a in common if base_edge[a] != other_edge[a]]

            print("-" * 90)
            print(f"🔁 Compare edge_index hash with: {other_dir}")
            print(f"  common_apps   : {len(common)}")
            print(f"  mismatches    : {len(mismatch)}")

            for a in mismatch[:50]:
                errors.append(
                    f"[EDGE_INDEX_MISMATCH_ACROSS_DIRS] app={a} base={base_edge[a]} other={other_edge[a]} other_dir={other_dir}"
                )

            common_mp = set(base_mp.keys()) & set(other_mp.keys())
            mismatch_mp = [a for a in common_mp if base_mp[a] != other_mp[a]]

            print(f"🔁 Compare mp_edge_index hash with: {other_dir}")
            print(f"  common_apps_mp: {len(common_mp)}")
            print(f"  mp_mismatches : {len(mismatch_mp)}")

            for a in mismatch_mp[:50]:
                errors.append(
                    f"[MP_EDGE_INDEX_MISMATCH_ACROSS_DIRS] app={a} base={base_mp[a]} other={other_mp[a]} other_dir={other_dir}"
                )

            if compare_strict:
                missing_in_other = sorted(list(set(base_edge.keys()) - set(other_edge.keys())))
                missing_in_base = sorted(list(set(other_edge.keys()) - set(base_edge.keys())))
                for a in missing_in_other[:50]:
                    errors.append(f"[APP_MISSING_IN_OTHER_DIR] app={a} other_dir={other_dir}")
                for a in missing_in_base[:50]:
                    errors.append(f"[APP_MISSING_IN_BASE_DIR] app={a} other_dir={other_dir}")

                missing_in_other_mp = sorted(list(set(base_mp.keys()) - set(other_mp.keys())))
                missing_in_base_mp = sorted(list(set(other_mp.keys()) - set(base_mp.keys())))
                for a in missing_in_other_mp[:50]:
                    errors.append(f"[APP_MISSING_IN_OTHER_DIR_MP] app={a} other_dir={other_dir}")
                for a in missing_in_base_mp[:50]:
                    errors.append(f"[APP_MISSING_IN_BASE_DIR_MP] app={a} other_dir={other_dir}")

    ok = (len(errors) == 0) and (len(bad_recall_apps) == 0)
    if gt_map is not None and gt_recall_thr > 0.0:
        ok = ok and (len(bad_gt_recall_apps) == 0)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", required=True, help="Path to <task>_<method> packed directory containing batch_*.pkl")
    ap.add_argument(
        "--compare-pack-dir",
        action="append",
        default=[],
        help="Optional: add more packed dirs to compare edge_index/mp_edge_index consistency by app_name (can repeat).",
    )
    ap.add_argument("--compare-strict", action="store_true", help="If set, treat missing app in any compared dir as an error.")
    ap.add_argument("--expected-task", default=None)
    ap.add_argument("--expected-method", default=None)
    ap.add_argument("--check-meta", action="store_true", help="Check task_name/reduction_method fields if present")
    ap.add_argument("--require-meta", action="store_true", help="Require task_name/reduction_method fields exist")
    ap.add_argument("--cand-recall-thr", type=float, default=0.98)
    ap.add_argument("--dump-csv", default=None, help="Optional path to write per-app metrics as CSV")
    ap.add_argument("--max-graphs", type=int, default=None, help="Limit graphs for quick test")

    # ===== GT options =====
    ap.add_argument("--gt-file", default=None, help="Optional external GT transitions file (no pkl modification).")
    ap.add_argument("--gt-recall-thr", type=float, default=0.0, help="Warn/error list apps with gt_recall_in_C < this (0 disables).")

    args = ap.parse_args()

    code = verify_pack_dir(
        pack_dir=args.pack_dir,
        expected_task=args.expected_task,
        expected_method=args.expected_method,
        check_meta=args.check_meta,
        require_meta=args.require_meta,
        cand_recall_thr=args.cand_recall_thr,
        dump_csv=args.dump_csv,
        compare_pack_dirs=args.compare_pack_dir,
        compare_strict=args.compare_strict,
        max_graphs=args.max_graphs,
        gt_file=args.gt_file,
        gt_recall_thr=args.gt_recall_thr,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()