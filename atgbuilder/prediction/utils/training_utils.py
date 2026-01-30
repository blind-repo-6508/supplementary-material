
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def split_graphs(
    graphs: List[Data],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data]]:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(graphs)
    if n == 0:
        return [], [], []

    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    train_graphs = [graphs[i] for i in idx_train]
    val_graphs = [graphs[i] for i in idx_val]
    test_graphs = [graphs[i] for i in idx_test]

    return train_graphs, val_graphs, test_graphs


class TrainingUtils:


    @staticmethod
    def split_graphs(
        graphs: List[Data],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        seed: int = 42,
    ) -> Tuple[List[Data], List[Data], List[Data]]:

        return split_graphs(graphs, train_ratio, val_ratio, test_ratio, seed)

    @staticmethod
    def set_random_seed(seed: int):

        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def mean_std(values: List[float]) -> Tuple[float, float]:

        if not values:
            return 0.0, 0.0
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std())