# -*- coding: utf-8 -*-
import os
import re
from typing import Dict, Set, Tuple, List, Optional, Any

import torch
from torch_geometric.data import Data


Transition = Tuple[str, str]


def _norm_act(s: str) -> str:
    return (s or "").strip()


def _app_keys_from_apk_path(apk_path: str) -> Set[str]:

    base = os.path.splitext(os.path.basename(apk_path.strip()))[0]
    keys = {base}

    m = re.match(r"^(.*)_(\d+)$", base)
    if m:
        keys.add(m.group(1))

    keys.add(apk_path.strip())
    keys.add(os.path.basename(apk_path.strip()))
    return keys


def load_gt_transitions_bigger1(gt_path: str) -> Dict[str, Set[Transition]]:

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


def _extract_node_names(g: Data) -> Optional[List[str]]:

    for attr in ("node_names", "activity_names", "activities", "id2activity", "id2name"):
        v = getattr(g, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        if isinstance(v, dict):
            # id->name
            if all(isinstance(k, (int,)) for k in v.keys()):
                max_k = max(v.keys()) if v else -1
                out = [""] * (max_k + 1)
                for k, name in v.items():
                    if 0 <= int(k) < len(out):
                        out[int(k)] = str(name)
                return out
    return None


def _graph_app_key(g: Data) -> str:

    for attr in ("app_name", "apk_name", "apk_path"):
        v = getattr(g, attr, None)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            base = os.path.splitext(os.path.basename(s))[0]
            return base if base else s
    return "unknown_app"


def attach_y_gt_on_candidate_edges(
    graphs: List[Data],
    gt_map: Dict[str, Set[Transition]],
    *,
    label_attr: str = "y_gt",
    stats_attr: str = "gt_stats",
) -> Dict[str, Any]:

    n_total = len(graphs)
    n_with_gt = 0
    n_labeled = 0

    sum_gt_edges = 0
    sum_gt_in_c = 0

    for g in graphs:
        key0 = _graph_app_key(g)

        keys = {key0}
        m = re.match(r"^(.*)_(\d+)$", key0)
        if m:
            keys.add(m.group(1))

        gt_edges: Optional[Set[Transition]] = None
        hit_key: Optional[str] = None
        for k in keys:
            if k in gt_map:
                gt_edges = gt_map[k]
                hit_key = k
                break

        if not gt_edges:
            continue
        n_with_gt += 1

        node_names = _extract_node_names(g)
        if node_names is None or len(node_names) == 0:
            continue

        id2name = [str(x) for x in node_names]

        ei = getattr(g, "edge_index", None)
        if ei is None or (not isinstance(ei, torch.Tensor)) or ei.dim() != 2 or ei.size(0) != 2:
            continue

        E = int(ei.size(1))
        y_gt = torch.zeros((E,), dtype=torch.long)

        gt_in_c = 0
        src_ids = ei[0].detach().cpu().tolist()
        dst_ids = ei[1].detach().cpu().tolist()

        for i in range(E):
            si = int(src_ids[i]); di = int(dst_ids[i])
            if si < 0 or di < 0 or si >= len(id2name) or di >= len(id2name):
                continue
            sname = _norm_act(id2name[si])
            dname = _norm_act(id2name[di])
            if (sname, dname) in gt_edges:
                y_gt[i] = 1
                gt_in_c += 1

        setattr(g, label_attr, y_gt.to(ei.device))

        gt_cnt = len(gt_edges)
        recall = (gt_in_c / (gt_cnt + 1e-12)) if gt_cnt > 0 else 0.0
        setattr(g, stats_attr, {
            "hit_key": hit_key,
            "gt_edges": int(gt_cnt),
            "gt_in_candidate": int(gt_in_c),
            "gt_recall_in_C": float(recall),
        })

        n_labeled += 1
        sum_gt_edges += gt_cnt
        sum_gt_in_c += gt_in_c

    return {
        "n_total": n_total,
        "n_with_gt_key": n_with_gt,
        "n_labeled_y_gt": n_labeled,
        "gt_edges_sum": int(sum_gt_edges),
        "gt_in_c_sum": int(sum_gt_in_c),
        "gt_recall_micro": float(sum_gt_in_c / (sum_gt_edges + 1e-12)) if sum_gt_edges > 0 else 0.0,
    }


def filter_graphs_with_y_gt(graphs: List[Data], label_attr: str = "y_gt") -> List[Data]:
    out = []
    for g in graphs:
        y = getattr(g, label_attr, None)
        if isinstance(y, torch.Tensor) and y.numel() > 0:
            out.append(g)
    return out