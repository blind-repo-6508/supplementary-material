# src/prediction/utils/balanced_sampling.py
# -*- coding: utf-8 -*-
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class BalancedSampler:

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    def balance_single_graph(self, graph: Data, rate: float = 1.0) -> Data:
        if not hasattr(graph, "y") or graph.y is None:
            return graph

        positive_mask: torch.Tensor = (graph.y == 1)
        negative_mask: torch.Tensor = (graph.y == 0)

        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        negative_indices = negative_mask.nonzero(as_tuple=True)[0]

        n_positive = len(positive_indices)
        n_negative = len(negative_indices)

        if n_positive == 0 or n_negative == 0:
            return graph

        target_negative = int(n_positive * rate)

        if n_negative > target_negative:
            selected_negative_indices = self._random_choice(negative_indices, target_negative)
        else:
            selected_negative_indices = self._oversample(negative_indices, target_negative)

        balanced_indices = torch.cat([positive_indices, selected_negative_indices])
        balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

        balanced_graph_data = {
            "x": graph.x,

            "edge_index": graph.edge_index[:, balanced_indices],
            "y": graph.y[balanced_indices],
        }

        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            balanced_graph_data["edge_attr"] = graph.edge_attr[balanced_indices]

        if hasattr(graph, "mp_edge_index") and graph.mp_edge_index is not None:
            balanced_graph_data["mp_edge_index"] = graph.mp_edge_index

        if hasattr(graph, "node_cluster") and graph.node_cluster is not None:
            balanced_graph_data["node_cluster"] = graph.node_cluster

        if hasattr(graph, "y_seed") and graph.y_seed is not None:
            balanced_graph_data["y_seed"] = graph.y_seed[balanced_indices]

        balanced_graph = Data(**balanced_graph_data)
        return balanced_graph

    def balance_graphs_list(
            self,
            graphs: List[Data],
            rate: float = 1.0,
            num_workers: int = 0,
            show_stats: bool = False,
    ) -> List[Data]:

        stats = {
            "total_graphs": len(graphs),
            "balanced_graphs": 0,
            "skipped_graphs": 0,
            "before_positive": 0,
            "before_negative": 0,
            "after_positive": 0,
            "after_negative": 0,
        }

        if num_workers is None or num_workers <= 1:
            balanced_graphs: List[Data] = []
            for graph in tqdm(graphs, desc="Balancing graphs", unit="graph"):
                self._accumulate_and_balance_one(
                    graph, rate, stats, balanced_graphs
                )


            return balanced_graphs

        if num_workers <= 0:
            num_workers = 1

        balanced_graphs: List[Optional[Data]] = [None] * len(graphs)

        def _worker(idx_graph_rate: Tuple[int, Data, float]):
            idx, g, r = idx_graph_rate

            before_pos, before_neg = 0, 0
            if hasattr(g, "y") and g.y is not None:
                before_pos = int((g.y == 1).sum().item())
                before_neg = int((g.y == 0).sum().item())

            g_bal = self.balance_single_graph(g, r)

            after_pos, after_neg = 0, 0
            balanced_flag, skipped_flag = 0, 0
            if hasattr(g_bal, "y") and g_bal.y is not None:
                after_pos = int((g_bal.y == 1).sum().item())
                after_neg = int((g_bal.y == 0).sum().item())
                if after_pos > 0 and after_neg > 0:
                    balanced_flag = 1
                else:
                    skipped_flag = 1
            else:
                skipped_flag = 1

            return idx, g_bal, before_pos, before_neg, after_pos, after_neg, balanced_flag, skipped_flag

        tasks = [(i, g, rate) for i, g in enumerate(graphs)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_worker, t) for t in tasks]

            for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Balancing graphs (threads={num_workers})",
                    unit="graph",
            ):
                (
                    idx,
                    g_bal,
                    before_pos,
                    before_neg,
                    after_pos,
                    after_neg,
                    balanced_flag,
                    skipped_flag,
                ) = fut.result()

                balanced_graphs[idx] = g_bal

                stats["before_positive"] += before_pos
                stats["before_negative"] += before_neg
                stats["after_positive"] += after_pos
                stats["after_negative"] += after_neg
                stats["balanced_graphs"] += balanced_flag
                stats["skipped_graphs"] += skipped_flag


        return [g for g in balanced_graphs if g is not None]

    # -------------------------------------------------------

    def _accumulate_and_balance_one(
            self,
            graph: Data,
            rate: float,
            stats: dict,
            out_list: List[Data],
    ):
        if hasattr(graph, "y") and graph.y is not None:
            stats["before_positive"] += int((graph.y == 1).sum().item())
            stats["before_negative"] += int((graph.y == 0).sum().item())

        balanced_graph = self.balance_single_graph(graph, rate)

        if hasattr(balanced_graph, "y") and balanced_graph.y is not None:
            stats["after_positive"] += int((balanced_graph.y == 1).sum().item())
            stats["after_negative"] += int((balanced_graph.y == 0).sum().item())

            unique_positive = int((balanced_graph.y == 1).sum().item())
            unique_negative = int((balanced_graph.y == 0).sum().item())

            if unique_positive > 0 and unique_negative > 0:
                stats["balanced_graphs"] += 1
            else:
                stats["skipped_graphs"] += 1
        else:
            stats["skipped_graphs"] += 1

        out_list.append(balanced_graph)

    def _random_choice(self, indices: torch.Tensor, k: int) -> torch.Tensor:
        if len(indices) <= k:
            return indices
        perm = torch.randperm(len(indices), device=indices.device)
        return indices[perm[:k]]

    def _oversample(self, indices: torch.Tensor, k: int) -> torch.Tensor:
        if len(indices) == 0:
            return indices

        repeat_times = k // len(indices)
        remainder = k % len(indices)

        repeated = []
        for _ in range(repeat_times):
            repeated.append(indices)

        if remainder > 0:
            remainder_indices = self._random_choice(indices, remainder)
            repeated.append(remainder_indices)

        return torch.cat(repeated) if repeated else indices
