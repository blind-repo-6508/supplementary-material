# src/embedding/components/seed_atg_builder_packer.py
import gc
import logging
import os
import pickle
import random
from datetime import datetime
from typing import Dict, Optional, Union, List, Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.embedding.config.base_config import BaseConfig
from src.embedding.config.task_registry import TaskConfig, TaskUtils
from src.embedding.utils.embedding_utils import EmbeddingReader
from src.embedding.utils.file_utils import FileUtils


class GraphConstructionUtils:

    @staticmethod
    def _extract_pkg(activity_name: str) -> str:
        if not activity_name:
            return ""
        if "." not in activity_name:
            return activity_name
        return activity_name.rsplit(".", 1)[0]

    @staticmethod
    def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:

        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norm + eps)

    @staticmethod
    def _build_knn_edges(
            emb_mat: np.ndarray,
            pkgs: List[str],
            k: int,
            intra_first: bool = True,
            allow_cross_pkg_fill: bool = True,
    ) -> List[Tuple[int, int]]:
        n = emb_mat.shape[0]
        if n <= 1 or k <= 0:
            return []

        k = min(int(k), n - 1)

        z = GraphConstructionUtils._l2_normalize(emb_mat)
        sim = z @ z.T
        np.fill_diagonal(sim, -1e9)

        edges: List[Tuple[int, int]] = []

        for i in range(n):
            scores = sim[i]

            if intra_first:
                same_idx = [j for j in range(n) if j != i and pkgs[j] == pkgs[i]]
                if same_idx:
                    same_scores = scores[same_idx]
                    order = np.argsort(-same_scores)
                    picked = [same_idx[o] for o in order[:k]]
                else:
                    picked = []

                if allow_cross_pkg_fill and len(picked) < k:
                    need = k - len(picked)
                    order_all = np.argsort(-scores)
                    for j in order_all:
                        if j == i:
                            continue
                        if j in picked:
                            continue
                        picked.append(int(j))
                        if len(picked) >= k:
                            break
            else:
                order_all = np.argsort(-scores)
                picked = []
                for j in order_all:
                    if j == i:
                        continue
                    picked.append(int(j))
                    if len(picked) >= k:
                        break

            for j in picked:
                edges.append((i, j))

        return edges

    @staticmethod
    def _build_seed_maps(atg_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], str]]:

        label_map: Dict[Tuple[str, str], int] = {}
        widget_map: Dict[Tuple[str, str], str] = {}

        for row in atg_df.itertuples(index=False):
            s = getattr(row, "source")
            t = getattr(row, "target")
            w = getattr(row, "widget")
            y = int(getattr(row, "is_transition"))
            key = (s, t)

            if key not in label_map:
                label_map[key] = y
                widget_map[key] = w
            else:
                if y == 1 and label_map[key] == 0:
                    label_map[key] = 1
                    widget_map[key] = w

        return label_map, widget_map

    @staticmethod
    def _add_reverse_edges(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return edges + [(j, i) for (i, j) in edges]

    @staticmethod
    def _add_self_loops(n: int, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return edges + [(i, i) for i in range(n)]

    @staticmethod
    def _filter_self_loops(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return [(i, j) for (i, j) in edges if i != j]

    @staticmethod
    def create_pyg_data(
            activity_emb: Dict[str, np.ndarray],
            atg_df: pd.DataFrame,
            widget_emb: Optional[Dict[str, np.ndarray]] = None,
            device: torch.device = None,
            candidate_cfg: Optional[Dict[str, Union[str, int, bool]]] = None,
            candidate_activity_emb: Optional[Dict[str, np.ndarray]] = None,
    ) -> Data:
        if device is None:
            device = torch.device("cpu")

        candidate_cfg = candidate_cfg or {}
        candidate_mode = str(candidate_cfg.get("mode", "knn")).lower()
        k_pred = int(candidate_cfg.get("k_pred", 20))
        k_mp = int(candidate_cfg.get("k_mp", 60))
        intra_first = bool(candidate_cfg.get("intra_first", True))
        allow_cross_pkg_fill = bool(candidate_cfg.get("allow_cross_pkg_fill", True))
        add_reverse_pred = bool(candidate_cfg.get("add_reverse_pred", True))
        add_reverse_mp = bool(candidate_cfg.get("add_reverse_mp", True))
        add_self_loop_mp = bool(candidate_cfg.get("add_self_loop_mp", False))

        all_activities = sorted(set(atg_df["source"]).union(set(atg_df["target"])))
        activity_to_idx = {act: i for i, act in enumerate(all_activities)}
        n_nodes = len(all_activities)

        sample_emb = next(iter(activity_emb.values()))
        node_features_x: List[np.ndarray] = [
            activity_emb.get(act, np.zeros_like(sample_emb)) for act in all_activities
        ]
        x = torch.tensor(np.asarray(node_features_x), dtype=torch.float, device=device)


        if candidate_mode == "knn":
            small_n = int(candidate_cfg.get("small_n", 0))
            small_k_pred_ratio = float(candidate_cfg.get("small_k_pred_ratio", 0.0))
            small_k_mp_ratio = float(candidate_cfg.get("small_k_mp_ratio", 0.0))
            small_allow_cross_pkg_fill = bool(candidate_cfg.get("small_allow_cross_pkg_fill", allow_cross_pkg_fill))
            small_add_reverse_pred = bool(candidate_cfg.get("small_add_reverse_pred", add_reverse_pred))

            k_pred_eff = k_pred
            k_mp_eff = k_mp
            allow_cross_pkg_fill_eff = allow_cross_pkg_fill
            add_reverse_pred_eff = add_reverse_pred

            if small_n > 0 and n_nodes <= small_n:
                if n_nodes <= 1:
                    k_pred_eff, k_mp_eff = 0, 0
                else:
                    k_pred_eff = max(1,
                                     int(round((n_nodes - 1) * small_k_pred_ratio))) if small_k_pred_ratio > 0 else min(
                        k_pred, n_nodes - 1)
                    k_mp_eff = max(1, int(round((n_nodes - 1) * small_k_mp_ratio))) if small_k_mp_ratio > 0 else min(
                        k_mp, n_nodes - 1)
                allow_cross_pkg_fill_eff = small_allow_cross_pkg_fill
                add_reverse_pred_eff = small_add_reverse_pred

            if n_nodes > 1:
                k_pred_eff = min(int(k_pred_eff), n_nodes - 1)
                k_mp_eff = min(int(k_mp_eff), n_nodes - 1)
            else:
                k_pred_eff, k_mp_eff = 0, 0

            cand_emb = candidate_activity_emb if (candidate_activity_emb is not None) else activity_emb
            cand_sample = next(iter(cand_emb.values()))
            node_features_cand: List[np.ndarray] = [
                cand_emb.get(act, np.zeros_like(cand_sample)) for act in all_activities
            ]
            emb_mat = np.asarray(node_features_cand, dtype=np.float32)

            pkgs = [GraphConstructionUtils._extract_pkg(a) for a in all_activities]


            seed_label_map, seed_widget_map = GraphConstructionUtils._build_seed_maps(atg_df)

            seed_pos_pairs: List[Tuple[int, int]] = []
            for (s_name, t_name), yv in seed_label_map.items():
                if int(yv) == 1 and s_name in activity_to_idx and t_name in activity_to_idx:
                    si = activity_to_idx[s_name]
                    ti = activity_to_idx[t_name]
                    if si != ti:
                        seed_pos_pairs.append((si, ti))

            if len(seed_pos_pairs) > 0:
                seed_edge_index = torch.tensor(seed_pos_pairs, dtype=torch.long, device=device).t().contiguous()
            else:
                seed_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            seed_pos_set = set(seed_pos_pairs)


            knn_pred = GraphConstructionUtils._build_knn_edges(
                emb_mat=emb_mat, pkgs=pkgs, k=k_pred_eff, intra_first=intra_first,
                allow_cross_pkg_fill=allow_cross_pkg_fill_eff,
            )
            knn_mp = GraphConstructionUtils._build_knn_edges(
                emb_mat=emb_mat, pkgs=pkgs, k=k_mp_eff, intra_first=intra_first,
                allow_cross_pkg_fill=allow_cross_pkg_fill_eff,
            )
            knn_pred = GraphConstructionUtils._filter_self_loops(knn_pred)
            knn_mp = GraphConstructionUtils._filter_self_loops(knn_mp)

            knn_pred_eval_edges = list(knn_pred)
            if add_reverse_pred_eff:
                knn_pred_eval_edges = list(set(GraphConstructionUtils._add_reverse_edges(knn_pred_eval_edges)))
                knn_pred_eval_edges = GraphConstructionUtils._filter_self_loops(knn_pred_eval_edges)

            knn_pred_set_for_eval = set(knn_pred_eval_edges)
            if len(seed_pos_pairs) == 0:
                knn_only_recall = 1.0
            else:
                knn_only_recall = float(len(seed_pos_set & knn_pred_set_for_eval)) / float(len(seed_pos_set))

            if len(knn_pred_eval_edges) > 0:
                knn_edge_index = torch.tensor(sorted(list(set(knn_pred_eval_edges))), dtype=torch.long,
                                              device=device).t().contiguous()
            else:
                knn_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)


            pred_set = set(knn_pred) | seed_pos_set
            mp_set = set(knn_mp) | pred_set

            pred_edges = sorted(list(pred_set))
            mp_edges = sorted(list(mp_set))

            if add_reverse_pred_eff:
                pred_edges = sorted(list(set(GraphConstructionUtils._add_reverse_edges(pred_edges))))
            if add_reverse_mp:
                mp_edges = sorted(list(set(GraphConstructionUtils._add_reverse_edges(mp_edges))))
            if add_self_loop_mp:
                mp_edges = sorted(list(set(GraphConstructionUtils._add_self_loops(n_nodes, mp_edges))))

            pred_edges = GraphConstructionUtils._filter_self_loops(pred_edges)

            if len(pred_edges) > 0:
                edge_index = torch.tensor(pred_edges, dtype=torch.long, device=device).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            if len(mp_edges) > 0:
                mp_edge_index = torch.tensor(mp_edges, dtype=torch.long, device=device).t().contiguous()
            else:
                mp_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            y_list: List[int] = []
            wid_list: List[str] = []
            for (si, ti) in pred_edges:
                s_name = all_activities[si]
                t_name = all_activities[ti]
                key = (s_name, t_name)
                yv = int(seed_label_map.get(key, 0))
                y_list.append(yv)
                wid_list.append(str(seed_widget_map.get(key, "NONE_WIDGET")))

            y = torch.tensor(np.asarray(y_list), dtype=torch.long, device=device) if len(y_list) > 0 \
                else torch.empty((0,), dtype=torch.long, device=device)

            edge_attr = None
            if widget_emb is not None:
                sample_w = next(iter(widget_emb.values()))
                none_w = widget_emb.get("NONE_WIDGET", np.zeros_like(sample_w))
                edge_attributes: List[np.ndarray] = [widget_emb.get(wid, none_w) for wid in wid_list]
                edge_attr = torch.tensor(np.asarray(edge_attributes), dtype=torch.float, device=device) \
                    if len(edge_attributes) > 0 else torch.empty((0, len(none_w)), dtype=torch.float, device=device)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.mp_edge_index = mp_edge_index
            data.seed_edge_index = seed_edge_index
            data.knn_edge_index = knn_edge_index
            data.knn_only_recall = float(knn_only_recall)

            data.activity_names = all_activities

            return data

        edge_pairs: List[List[int]] = []
        edge_attributes: Optional[List[np.ndarray]] = [] if widget_emb is not None else None

        for row in atg_df.itertuples(index=False):
            # row: source, target, widget, is_transition
            src = activity_to_idx[getattr(row, "source")]
            tgt = activity_to_idx[getattr(row, "target")]
            edge_pairs.append([src, tgt])

            if widget_emb is not None:
                wid = getattr(row, "widget")
                if wid not in widget_emb:
                    raise KeyError(f"Missing widget embedding: {wid}")
                edge_attributes.append(widget_emb[wid])

        edge_index = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()

        edge_attr = None
        if edge_attributes is not None:
            edge_attr = torch.tensor(np.asarray(edge_attributes), dtype=torch.float, device=device)

        y = torch.tensor(atg_df["is_transition"].values, dtype=torch.long, device=device)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.mp_edge_index = edge_index
        data.candidate_mode = "atg"
        return data

    @staticmethod
    def get_dimensions(train_graphs: Union[List[Data], Iterator[Data]]) -> Tuple[int, int]:
        first_graph = next(iter(train_graphs))
        activity_dim = first_graph.x.shape[1]
        widget_dim = first_graph.edge_attr.shape[1] if first_graph.edge_attr is not None else 0
        return activity_dim, widget_dim


class DataPacker:
    def __init__(self, config: BaseConfig, pack_dir_name: str = "packed_data_tmp"):
        self.config = config
        self.pack_dir = os.path.join(config.ROOT_DIR, pack_dir_name)
        os.makedirs(self.pack_dir, exist_ok=True)

    def get_pack_path(self, task_name: str, reduction_method: str) -> str:
        return os.path.join(self.pack_dir, f"{task_name}_{reduction_method}")

    def pack_graphs(
            self,
            graphs: List[Data],
            task_name: str,
            reduction_method: str,
            graphs_per_file: int = 100,
    ) -> str:
        pack_path = self.get_pack_path(task_name, reduction_method)
        os.makedirs(pack_path, exist_ok=True)


        for i in tqdm(range(0, len(graphs), graphs_per_file), desc=""):
            batch_graphs = graphs[i: i + graphs_per_file]
            batch_file = os.path.join(pack_path, f"batch_{i // graphs_per_file:06d}.pkl")
            with open(batch_file, "wb") as f:
                pickle.dump(batch_graphs, f)


        return pack_path

    def load_packed_graphs(self, pack_path: str) -> List[Data]:

        graphs = []
        batch_files = sorted([f for f in os.listdir(pack_path) if f.endswith('.pkl')])

        for batch_file in tqdm(batch_files, desc=""):
            batch_path = os.path.join(pack_path, batch_file)
            with open(batch_path, 'rb') as f:
                batch_graphs = pickle.load(f)
                graphs.extend(batch_graphs)

        return graphs


class SeedAtgBuilderPacker:

    def __init__(
            self,
            config: BaseConfig,
            reduction_method: str = "LINEAR",
            use_packed_data: bool = True,
            pack_dir_name: str = "packed_data_tmp",
            graphs_per_file: int = 100,
            candidate_mode: str = "knn",
            candidate_k_pred: int = 20,
            candidate_k_mp: int = 60,
            candidate_intra_first: bool = True,
            candidate_allow_cross_pkg_fill: bool = True,
            candidate_add_reverse_pred: bool = True,
            candidate_add_reverse_mp: bool = True,
            candidate_add_self_loop_mp: bool = False,
    ):
        self.config = config
        self.supported_tasks = TaskConfig.ABLATION_TASKS

        self.reduction_method = (reduction_method or "LINEAR").upper()
        self.use_packed_data = use_packed_data
        self.graphs_per_file = graphs_per_file

        self.device = torch.device("cpu")

        self.rng = random.Random(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)

        self.data_packer = DataPacker(config, pack_dir_name=pack_dir_name)

        self.progress_logger = self._setup_progress_logger()
        self.failure_logger = self._setup_failure_logger()

        self.candidate_cfg = {
            "mode": str(candidate_mode).lower(),
            "k_pred": int(candidate_k_pred),
            "k_mp": int(candidate_k_mp),
            "intra_first": bool(candidate_intra_first),
            "allow_cross_pkg_fill": bool(candidate_allow_cross_pkg_fill),
            "add_reverse_pred": bool(candidate_add_reverse_pred),
            "add_reverse_mp": bool(candidate_add_reverse_mp),
            "add_self_loop_mp": bool(candidate_add_self_loop_mp),
            "small_n": 20,
            "small_k_pred_ratio": 0.30,
            "small_k_mp_ratio": 0.60,
            "small_allow_cross_pkg_fill": True,
            "small_add_reverse_pred": False,
        }

    def run_pipeline(
            self,
            ablation_tasks: Optional[Union[str, List[str]]] = None,
            force_rebuild: bool = False,
    ) -> Dict[str, str]:
        valid_tasks = TaskUtils.process_task_list(ablation_tasks, self.supported_tasks)

        app_atg_paths = FileUtils.get_files_by_suffix(self.config.ATG_INDEX_DIR, ".txt")

        results: Dict[str, str] = {}
        for task in valid_tasks:
            pack_path = self.data_packer.get_pack_path(task, self.reduction_method)
            if self.use_packed_data and (not force_rebuild) and os.path.exists(pack_path):
                results[task] = pack_path
                self._cleanup_resources()
                continue

            graphs = self._build_graphs_for_task(task, app_atg_paths)
            if not graphs:
                self._cleanup_resources()
                continue

            results[task] = self.data_packer.pack_graphs(
                graphs=graphs,
                task_name=task,
                reduction_method=self.reduction_method,
                graphs_per_file=self.graphs_per_file,
            )
            self._cleanup_resources()

        return results

    def _build_graphs_for_task(self, task: str, app_atg_paths: List[str]) -> List[Data]:
        _, widget_components = self._split_task_components_by_entity(task)
        need_widget = len(widget_components) > 0  # ab_widget_none => False

        graphs: List[Data] = []
        for app_path in app_atg_paths:
            app_name = os.path.splitext(os.path.basename(app_path))[0]
            try:
                atg_df = self._read_atg_as_df(app_name)

                activity_emb, resolved_name = self._load_embedding_with_correction(app_name, task, "activity")

                cand_activity_emb, _ = self._load_embedding_with_correction(resolved_name, "ab_complete", "activity")

                widget_emb = None
                if need_widget:
                    widget_emb, _ = self._load_embedding_with_correction(resolved_name, task, "widget")

                pyg = GraphConstructionUtils.create_pyg_data(
                    activity_emb=activity_emb,
                    atg_df=atg_df,
                    widget_emb=widget_emb,
                    device=self.device,
                    candidate_cfg=self.candidate_cfg,
                    candidate_activity_emb=cand_activity_emb,
                )

                pyg.app_name = app_name
                pyg.task_name = task
                pyg.reduction_method = self.reduction_method

                graphs.append(pyg)

            except Exception as e:
                self._record_failed_app(task, app_name, str(e))
                continue

        return graphs

    def _read_atg_as_df(self, app_name: str) -> pd.DataFrame:
        atg_file = os.path.join(self.config.ATG_INDEX_DIR, f"{app_name}.txt")


        rows = []
        with open(atg_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(";")
                if len(parts) != 4:
                    continue
                source, target, widget, is_trans = [p.strip() for p in parts]
                is_trans_int = int(is_trans)
                if is_trans_int not in (0, 1):
                    continue
                rows.append(
                    {"source": source, "target": target, "widget": widget, "is_transition": is_trans_int}
                )


        return pd.DataFrame(rows)

    def _split_task_components_by_entity(self, task_name: str) -> Tuple[List[str], List[str]]:
        task_components = self.supported_tasks[task_name]["task_names"]
        activity_components = [c for c in task_components if c.startswith("activity")]
        widget_components = [c for c in task_components if c.startswith("widget")]
        return activity_components, widget_components

    def _correct_app_name_for_embedding(self, app_name: str) -> str:
        corrections = {
            "com.ivymobi.aurorareader.ebook.pdf.epub.txt.reader--21":
                "com.ivymobi.aurorareader.ebook.pdf.epub.reader--21"
        }
        return corrections.get(app_name, app_name)

    def _load_embedding_with_correction(
            self,
            app_name: str,
            task: str,
            entity_type: str,
    ) -> Tuple[Dict[str, np.ndarray], str]:
        try:
            emb, _ = EmbeddingReader.load_embedding(
                self.config,
                app_name,
                task,
                entity_type,
                reduction_method=self.reduction_method,
            )
            if not emb:
                raise FileNotFoundError(f"{entity_type} embedding : app={app_name}, task={task}")
            return emb, app_name
        except Exception as e1:
            corrected = self._correct_app_name_for_embedding(app_name)
            if corrected != app_name:
                emb, _ = EmbeddingReader.load_embedding(
                    self.config,
                    corrected,
                    task,
                    entity_type,
                    reduction_method=self.reduction_method,
                )
                if not emb:
                    raise FileNotFoundError(f"{entity_type} embedding : app={corrected}, task={task}")
                return emb, corrected
            raise Exception(f"加载 {entity_type} embedding : {e1}")

    def _setup_failure_logger(self) -> logging.Logger:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger("FailureRecorder")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fp = os.path.join(log_dir, f"failed_apps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(fh)
            logger.propagate = False
        return logger

    def _setup_progress_logger(self) -> logging.Logger:
        logger = logging.getLogger("ProgressLogger")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(ch)
        return logger

    def _record_failed_app(self, task: str, app_name: str, reason: str) -> None:
        self.failure_logger.info(
            f"FAILED - Task: {task}, Method: {self.reduction_method}, App: {app_name}, Reason: {reason}"
        )

    def _cleanup_resources(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
