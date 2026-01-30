# src/embedding/tools/verify_packed_pkl.py
import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

import torch
from torch_geometric.data import Data


def _is_finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item() if t.numel() > 0 else True


def _shape_str(t: torch.Tensor) -> str:
    return f"dtype={t.dtype}, shape={tuple(t.shape)}, device={t.device}"


def _check_meta(
    g: Data,
    expected_task: Optional[str] = None,
    expected_method: Optional[str] = None,
    require_meta: bool = True,
) -> List[str]:
    errors = []

    # 你现在构图时写了这三个字段：app_name/task_name/reduction_method
    for k in ["app_name", "task_name", "reduction_method"]:
        if not hasattr(g, k):
            if require_meta:
                errors.append(f"missing meta field: {k}")
            continue

    if hasattr(g, "app_name"):
        v = getattr(g, "app_name")
        if not isinstance(v, str) or not v.strip():
            errors.append(f"invalid app_name: {v!r}")

    if hasattr(g, "task_name"):
        v = getattr(g, "task_name")
        if not isinstance(v, str) or not v.strip():
            errors.append(f"invalid task_name: {v!r}")
        if expected_task is not None and isinstance(v, str) and v != expected_task:
            errors.append(f"task_name mismatch: expected={expected_task}, got={v}")

    if hasattr(g, "reduction_method"):
        v = getattr(g, "reduction_method")
        if not isinstance(v, str) or not v.strip():
            errors.append(f"invalid reduction_method: {v!r}")
        if expected_method is not None and isinstance(v, str) and v != expected_method:
            errors.append(f"reduction_method mismatch: expected={expected_method}, got={v}")

    return errors


def _check_graph(
    g: Data,
    expected_task: Optional[str] = None,
    expected_method: Optional[str] = None,
    check_meta: bool = True,
    require_meta: bool = True,
) -> List[str]:
    errors = []

    # meta 校验
    if check_meta:
        errors.extend(_check_meta(
            g,
            expected_task=expected_task,
            expected_method=expected_method,
            require_meta=require_meta,
        ))

    # 必备字段
    for key in ["x", "edge_index", "y"]:
        if not hasattr(g, key):
            errors.append(f"missing field: {key}")
            return errors

    x = g.x
    edge_index = g.edge_index
    y = g.y
    edge_attr = getattr(g, "edge_attr", None)

    # x: [N, D]
    if not isinstance(x, torch.Tensor):
        errors.append(f"x is not torch.Tensor: {type(x)}")
        return errors
    if x.dim() != 2:
        errors.append(f"x dim != 2: {_shape_str(x)}")
    if not _is_finite_tensor(x):
        errors.append(f"x contains NaN/Inf: {_shape_str(x)}")
    num_nodes = x.shape[0] if x.dim() == 2 else None

    # edge_index: [2, E]
    if not isinstance(edge_index, torch.Tensor):
        errors.append(f"edge_index is not torch.Tensor: {type(edge_index)}")
        return errors
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        errors.append(f"edge_index shape invalid (expect [2, E]): {_shape_str(edge_index)}")
    if edge_index.dtype not in (torch.long, torch.int64):
        errors.append(f"edge_index dtype not int64: {_shape_str(edge_index)}")
    num_edges = edge_index.shape[1] if edge_index.dim() == 2 else None

    # y: [E]
    if not isinstance(y, torch.Tensor):
        errors.append(f"y is not torch.Tensor: {type(y)}")
        return errors
    if num_edges is not None and y.shape[0] != num_edges:
        errors.append(f"y length != num_edges: y={_shape_str(y)}, num_edges={num_edges}")
    if y.dtype not in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8):
        errors.append(f"y dtype not integer: {_shape_str(y)}")

    # edge_index 索引范围合法
    if num_nodes is not None and edge_index.numel() > 0 and edge_index.shape[0] == 2:
        min_idx = int(edge_index.min().item())
        max_idx = int(edge_index.max().item())
        if min_idx < 0:
            errors.append(f"edge_index has negative node id: min={min_idx}")
        if max_idx >= num_nodes:
            errors.append(f"edge_index out of range: max={max_idx} >= num_nodes={num_nodes}")

    # edge_attr：若存在，应为 [E, De]
    if edge_attr is not None:
        if not isinstance(edge_attr, torch.Tensor):
            errors.append(f"edge_attr is not torch.Tensor: {type(edge_attr)}")
        else:
            if edge_attr.dim() != 2:
                errors.append(f"edge_attr dim != 2: {_shape_str(edge_attr)}")
            if num_edges is not None and edge_attr.shape[0] != num_edges:
                errors.append(f"edge_attr rows != num_edges: edge_attr={_shape_str(edge_attr)}, num_edges={num_edges}")
            if not _is_finite_tensor(edge_attr):
                errors.append(f"edge_attr contains NaN/Inf: {_shape_str(edge_attr)}")

    # ab_widget_none：强约束 edge_attr 必须 None
    if expected_task == "ab_widget_none":
        if edge_attr is not None:
            errors.append("ab_widget_none should have edge_attr=None, but got non-None")

    # y 值只能是 0/1
    if y.numel() > 0:
        uniq = set(torch.unique(y).cpu().tolist())
        if not uniq.issubset({0, 1}):
            errors.append(f"y has values not in {{0,1}}: {sorted(list(uniq))[:10]}")

    return errors


def verify_pack_dir(
    pack_dir: str,
    expected_task: Optional[str] = None,
    expected_method: Optional[str] = None,
    check_meta: bool = True,
    require_meta: bool = True,
    max_errors: int = 50,
) -> Dict[str, Any]:
    if not os.path.isdir(pack_dir):
        raise FileNotFoundError(f"pack_dir not found: {pack_dir}")

    batch_files = sorted([f for f in os.listdir(pack_dir) if f.endswith(".pkl")])
    if not batch_files:
        raise FileNotFoundError(f"no .pkl found under: {pack_dir}")

    total_graphs = 0
    total_batches = 0
    ok_graphs = 0

    dim_stats = defaultdict(int)  # (node_dim, edge_dim)->count
    err_counter = defaultdict(int)
    err_samples: List[Tuple[str, int, List[str]]] = []

    for bf in batch_files:
        total_batches += 1
        p = os.path.join(pack_dir, bf)

        try:
            with open(p, "rb") as f:
                graphs = pickle.load(f)
        except Exception as e:
            err_counter[f"pickle_load_error:{type(e).__name__}"] += 1
            if len(err_samples) < max_errors:
                err_samples.append((bf, -1, [f"pickle load failed: {e}"]))
            continue

        if not isinstance(graphs, list):
            err_counter["batch_not_list"] += 1
            if len(err_samples) < max_errors:
                err_samples.append((bf, -1, [f"batch object is not list: {type(graphs)}"]))
            continue

        for gi, g in enumerate(graphs):
            total_graphs += 1
            if not isinstance(g, Data):
                err_counter["graph_not_Data"] += 1
                if len(err_samples) < max_errors:
                    err_samples.append((bf, gi, [f"graph is not Data: {type(g)}"]))
                continue

            errs = _check_graph(
                g,
                expected_task=expected_task,
                expected_method=expected_method,
                check_meta=check_meta,
                require_meta=require_meta,
            )
            if errs:
                for e in errs:
                    err_counter[e] += 1
                if len(err_samples) < max_errors:
                    err_samples.append((bf, gi, errs))
                continue

            ok_graphs += 1
            node_dim = int(g.x.shape[1])
            edge_dim = int(g.edge_attr.shape[1]) if getattr(g, "edge_attr", None) is not None else 0
            dim_stats[(node_dim, edge_dim)] += 1

    return {
        "pack_dir": pack_dir,
        "expected_task": expected_task,
        "expected_method": expected_method,
        "check_meta": check_meta,
        "require_meta": require_meta,
        "total_batches": total_batches,
        "total_graphs": total_graphs,
        "ok_graphs": ok_graphs,
        "ok_rate": (ok_graphs / total_graphs) if total_graphs > 0 else 0.0,
        "dim_stats": dict(sorted(dim_stats.items(), key=lambda x: -x[1])),
        "error_counts": dict(sorted(err_counter.items(), key=lambda x: -x[1])),
        "error_samples": err_samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=str, required=True, help=": .../ab_baseline_LINEAR")
    ap.add_argument("--expected-task", type=str, default=None, help=": ab_widget_none")
    ap.add_argument("--expected-method", type=str, default=None, help=": LINEAR")
    ap.add_argument("--check-meta", action="store_true")
    ap.add_argument("--require-meta", action="store_true")
    ap.add_argument("--max-errors", type=int, default=50)
    args = ap.parse_args()

    report = verify_pack_dir(
        args.pack_dir,
        expected_task=args.expected_task,
        expected_method=args.expected_method,
        check_meta=args.check_meta,
        require_meta=args.require_meta,
        max_errors=args.max_errors,
    )

    print("\n" + "=" * 80)
    print("✅ Packed PKL Verification Report")
    print("=" * 80)
    print(f"pack_dir       : {report['pack_dir']}")
    print(f"expected_task  : {report['expected_task']}")
    print(f"expected_method: {report['expected_method']}")
    print(f"check_meta     : {report['check_meta']}")
    print(f"require_meta   : {report['require_meta']}")
    print(f"total_batches  : {report['total_batches']}")
    print(f"total_graphs   : {report['total_graphs']}")
    print(f"ok_graphs      : {report['ok_graphs']}")
    print(f"ok_rate        : {report['ok_rate']:.4f}")

    print("\n📐 Dimension stats (node_dim, edge_dim) -> count:")
    if report["dim_stats"]:
        for k, v in report["dim_stats"].items():
            print(f"  {k} -> {v}")
    else:
        print("  (none)")

    print("\n❌ Error counts:")
    if report["error_counts"]:
        for k, v in report["error_counts"].items():
            print(f"  {v:6d}  {k}")
    else:
        print("  (none)")

    if report["error_samples"]:
        print("\n🔎 Error samples:")
        for bf, gi, errs in report["error_samples"]:
            where = f"{bf}" if gi < 0 else f"{bf} [graph#{gi}]"
            print(f"\n- {where}")
            for e in errs:
                print(f"    - {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()