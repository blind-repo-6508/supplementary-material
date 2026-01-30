# -*- coding: utf-8 -*-
"""
Plot RQ2.2 noise curves with BOTH thresholds in ONE figure:
- Curve A: metrics @ t* (selected on GT val)
- Curve B: metrics @ 0.5
- Seed-only baseline: horizontal dashed line (P/R/F1 computed by directly using tool edges vs GT labels)

Aggregation (recommended):
- For each (task,branch,tool,mode,eta):
    1) mean over reps within each split_seed  -> per-split value
    2) mean ± std over split_seeds           -> plotted curve with error band
"""

import os
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ---- Global plotting style (PDF embedding + ACM-like serif: Linux Libertine) ----
from matplotlib import font_manager as fm

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

_candidates = [
    "Linux Libertine O",  # most common name on Linux
    "Linux Libertine",
    "Libertinus Serif",  # close substitute if Libertine not installed
    "DejaVu Serif",
]
_available = {f.name for f in fm.fontManager.ttflist}
for _name in _candidates:
    if _name in _available:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [_name]
        break
else:
    plt.rcParams["font.family"] = "serif"

ROOT = "/root/autodl-tmp/atg"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.embedding.components.seed_atg_builder_packer import DataPacker, GraphConstructionUtils
from src.embedding.config.base_config import BaseConfig
from src.prediction.utils.training_utils import TrainingUtils

# Reuse from your rq22 runner (must exist in your repo)
from src.experiments.effectiveness.run_rq22_noise_seed import (
    parse_all_raw_transitions,
    get_node_names_best_effort,
    build_y_seed_from_tool,
)


def prf_from_binary(y_pred, y_true):
    y_pred = np.asarray(y_pred).astype(np.int64)
    y_true = np.asarray(y_true).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def load_runlevel_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _aggregate_one_threshold(rows, use_tstar: bool):
    """
    Return: curves[key][eta][split_seed][metric] = list(values over reps)
    key = (task, branch, tool, mode)
    """
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        task = r["task"]
        branch = r["branch"]
        tool = r["tool"]
        mode = r["noise_mode"]
        eta = float(r["eta"])
        split_seed = int(r["split_seed"])

        if use_tstar:
            p = float(r["test_p_tstar"])
            rr = float(r["test_r_tstar"])
            f1 = float(r["test_f1_tstar"])
        else:
            p = float(r["test_p_05"])
            rr = float(r["test_r_05"])
            f1 = float(r["test_f1_05"])

        key = (task, branch, tool, mode)
        store[key][eta][split_seed]["precision"].append(p)
        store[key][eta][split_seed]["recall"].append(rr)
        store[key][eta][split_seed]["f1"].append(f1)
    return store


def aggregate_dual_threshold(rows):
    """
    Build dual-threshold curves with mean±std over split-seeds (after rep-mean in each split).
    Output: curves[key] = {
        "eta": [...],
        "precision_mean_tstar": [...], "precision_std_tstar": [...],
        "precision_mean_05":    [...], "precision_std_05": [...],
        ... for recall/f1
    }
    """
    store_t = _aggregate_one_threshold(rows, use_tstar=True)
    store_05 = _aggregate_one_threshold(rows, use_tstar=False)

    keys = set(store_t.keys()) | set(store_05.keys())
    curves = {}

    for key in sorted(keys):
        etas = sorted(set(store_t[key].keys()) | set(store_05[key].keys()))
        out = {"eta": etas}

        for metric in ["precision", "recall", "f1"]:
            means_t, stds_t = [], []
            means_05, stds_05 = [], []

            for e in etas:
                # ---- t* aggregation ----
                per_split_vals = []
                if e in store_t[key]:
                    for _, mp in store_t[key][e].items():
                        vals = mp[metric]
                        if len(vals) > 0:
                            per_split_vals.append(float(np.mean(vals)))  # rep-mean within split
                if len(per_split_vals) == 0:
                    means_t.append(0.0)
                    stds_t.append(0.0)
                else:
                    means_t.append(float(np.mean(per_split_vals)))
                    stds_t.append(float(np.std(per_split_vals, ddof=1)) if len(per_split_vals) >= 2 else 0.0)

                # ---- 0.5 aggregation ----
                per_split_vals = []
                if e in store_05[key]:
                    for _, mp in store_05[key][e].items():
                        vals = mp[metric]
                        if len(vals) > 0:
                            per_split_vals.append(float(np.mean(vals)))  # rep-mean within split
                if len(per_split_vals) == 0:
                    means_05.append(0.0)
                    stds_05.append(0.0)
                else:
                    means_05.append(float(np.mean(per_split_vals)))
                    stds_05.append(float(np.std(per_split_vals, ddof=1)) if len(per_split_vals) >= 2 else 0.0)

            out[f"{metric}_mean_tstar"] = means_t
            out[f"{metric}_std_tstar"] = stds_t
            out[f"{metric}_mean_05"] = means_05
            out[f"{metric}_std_05"] = stds_05

        curves[key] = out

    return curves


def compute_seed_baseline(task_name, split_seeds, tool, all_raw_transitions_path):
    """
    Seed-only baseline (binary prediction = seed edges).
    Evaluate directly vs GT labels on test split.
    Return mean±std over split_seeds (no reps here).
    """
    cfg = BaseConfig.from_yaml()
    packer = DataPacker(cfg, pack_dir_name="verify_ground_truth_packed_data")
    graphs_all = packer.load_packed_graphs(os.path.join(packer.pack_dir, f"{task_name}_LINEAR"))
    activity_dim, _ = GraphConstructionUtils.get_dimensions(graphs_all)
    baseline = parse_all_raw_transitions(all_raw_transitions_path)

    Ps, Rs, F1s = [], [], []
    for split_seed in split_seeds:
        _, _, test_raw = TrainingUtils.split_graphs(
            graphs_all, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=int(split_seed)
        )
        ypred_all, ytrue_all = [], []
        for g in test_raw:
            app = str(getattr(g, "app_name", "NA"))
            tool_set = baseline.get(app, {}).get(tool, set())
            node_names = get_node_names_best_effort(g, task_name=task_name, activity_dim=int(activity_dim))
            y_seed = build_y_seed_from_tool(g, tool_edge_set=tool_set, node_names=node_names).detach().cpu().numpy()
            y_gt = g.y.detach().cpu().numpy().astype(np.int64)
            ypred_all.append(y_seed)
            ytrue_all.append(y_gt)

        if ypred_all:
            ypred = np.concatenate(ypred_all)
            ytrue = np.concatenate(ytrue_all)
            p, r, f1 = prf_from_binary(ypred, ytrue)
            Ps.append(p)
            Rs.append(r)
            F1s.append(f1)

    return {
        "precision": float(np.mean(Ps)) if Ps else 0.0,
        "recall": float(np.mean(Rs)) if Rs else 0.0,
        "f1": float(np.mean(F1s)) if F1s else 0.0,
        "precision_std": float(np.std(Ps, ddof=1)) if len(Ps) >= 2 else 0.0,
        "recall_std": float(np.std(Rs, ddof=1)) if len(Rs) >= 2 else 0.0,
        "f1_std": float(np.std(F1s, ddof=1)) if len(F1s) >= 2 else 0.0,
    }


def plot_metric_dual(curve, seed_base_val, seed_base_std, title, metric, out_pdf, show_std=True):
    etas = curve["eta"]

    y_t = np.asarray(curve[f"{metric}_mean_tstar"], dtype=np.float64)
    s_t = np.asarray(curve[f"{metric}_std_tstar"], dtype=np.float64)

    y_05 = np.asarray(curve[f"{metric}_mean_05"], dtype=np.float64)
    s_05 = np.asarray(curve[f"{metric}_std_05"], dtype=np.float64)

    # ---- 1:1 canvas ----
    fig, ax = plt.subplots(figsize=(4, 4))

    # @t*
    ax.plot(etas, y_t, marker="o", label=r"@$t^{*}$", zorder=3)
    if show_std:
        ax.fill_between(etas, y_t - s_t, y_t + s_t, alpha=0.2)

    # @0.5
    ax.plot(etas, y_05, marker="o", label=r"@0.5", zorder=3)
    if show_std:
        ax.fill_between(etas, y_05 - s_05, y_05 + s_05, alpha=0.2)

    for ln in ax.lines:
        ln.set_zorder(3)
        ln.set_clip_on(False)

    # ---- X axis: fixed [0,1] ticks every 0.2 ----
    step = 0.2
    ticks = np.arange(0.0, 1.0 + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xlim(0.0, 1.0)

    # ---- Y axis: fixed range + 2 decimals ----
    ax.set_ylim(0.4, 1.0)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # ---- seed-only baseline: ONE dashed line; label shows mean±std (NO band) ----
    if seed_base_val is not None:
        ax.axhline(seed_base_val, linestyle="--", label="seed-only baseline")

        y_off = 0.01
        if seed_base_std is not None and seed_base_std > 0:
            txt = f"{float(seed_base_val):.4f} ± {float(seed_base_std):.4f}"
        else:
            txt = f"{float(seed_base_val):.4f}"

        # place at right end (x in axes coords, y in data coords)
        ax.text(
            0.99, float(seed_base_val) + y_off, txt,
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom"
        )

    # ---- Labels (capitalized) ----
    ylabel_map = {"precision": "Precision", "recall": "Recall", "f1": "F1-score"}
    ax.set_xlabel(r"Noise Rate $\eta$")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title(title)

    # ---- Grid: dashed grey ----
    ax.grid(True, linestyle="--", color="gray", alpha=0.25)

    # ---- Legend: top, horizontal, framed ----
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=0.5,
        labelspacing=0.4,
        borderpad=0.6,
        fancybox=False,
        framealpha=1.0,
    )
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_facecolor("white")

    # Leave room for legend
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq_tag", required=True)
    ap.add_argument("--task", default="ab_complete")
    ap.add_argument("--all_raw_transitions", default="/root/autodl-tmp/atg/all_raw_transitions.txt")
    ap.add_argument("--split_seeds", nargs="+", type=int, default=[42, 123, 2025])
    ap.add_argument("--show_std", action="store_true")
    args = ap.parse_args()

    result_dir = os.path.join(ROOT, "results", args.rq_tag)
    runlevel_csv = os.path.join(result_dir, f"{args.rq_tag}_runlevel.csv")
    if not os.path.exists(runlevel_csv):
        raise FileNotFoundError(f"runlevel csv not found: {runlevel_csv}")

    out_dir = "/root/autodl-tmp/atg/figures"
    os.makedirs(out_dir, exist_ok=True)

    rows = load_runlevel_csv(runlevel_csv)
    curves = aggregate_dual_threshold(rows)

    seed_base_cache = {}

    for (task, branch, tool, mode), curve in curves.items():
        if tool not in seed_base_cache:
            seed_base_cache[tool] = compute_seed_baseline(
                task_name=args.task,
                split_seeds=args.split_seeds,
                tool=tool,
                all_raw_transitions_path=args.all_raw_transitions,
            )
        seed_base = seed_base_cache[tool]

        for metric in ["precision", "recall", "f1"]:
            out_pdf = os.path.join(out_dir, f"curve__{task}__{branch}__{tool}__{mode}__{metric}__dual.pdf")
            plot_metric_dual(
                curve=curve,
                seed_base_val=seed_base[metric],
                seed_base_std=seed_base.get(f"{metric}_std", 0.0),
                metric=metric,
                title="",
                out_pdf=out_pdf,
                show_std=bool(args.show_std),
            )

    print("saved to:", out_dir)


if __name__ == "__main__":
    main()
