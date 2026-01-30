
from typing import List, Optional
import torch
from torch_geometric.data import Data

def _add_reverse(edge_index: torch.Tensor) -> torch.Tensor:
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)

def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    idx = torch.arange(num_nodes, device=edge_index.device, dtype=torch.long)
    self_loop = torch.stack([idx, idx], dim=0)
    return torch.cat([edge_index, self_loop], dim=1)

def ensure_mp_for_graph(
    g: Data,
    add_reverse: bool = True,
    add_self_loop: bool = False,
    only_if_missing: bool = True,
) -> Data:
    if only_if_missing and hasattr(g, "mp_edge_index") and g.mp_edge_index is not None:
        return g

    if not hasattr(g, "y") or g.y is None:
        g.mp_edge_index = getattr(g, "mp_edge_index", None) or g.edge_index
        return g

    y = g.y
    pos_mask = (y == 1)
    if pos_mask.sum().item() == 0:
        g.mp_edge_index = g.edge_index[:, :0]
        return g

    mp = g.edge_index[:, pos_mask]

    if add_reverse:
        mp = _add_reverse(mp)
    if add_self_loop:
        mp = _add_self_loops(mp, num_nodes=g.num_nodes)

    g.mp_edge_index = mp
    return g

def ensure_mp_for_graphs(
    graphs: List[Data],
    add_reverse: bool = True,
    add_self_loop: bool = False,
    only_if_missing: bool = True,
) -> List[Data]:
    for g in graphs:
        ensure_mp_for_graph(g, add_reverse=add_reverse, add_self_loop=add_self_loop, only_if_missing=only_if_missing)
    return graphs