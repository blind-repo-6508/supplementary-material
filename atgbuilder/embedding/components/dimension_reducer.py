import os
from typing import List, Dict, Union, Callable, Optional, TypedDict
import numpy as np
from tqdm import tqdm
from src.embedding.config.base_config import BaseConfig
from src.embedding.config.task_registry import TaskConfig, TaskUtils
from src.embedding.utils.file_utils import FileUtils


class EntityDirInfo(TypedDict):
    base: str
    tasks: Dict[str, str]

class MethodDirInfo(TypedDict):
    activity: EntityDirInfo
    widget: EntityDirInfo


class DimensionReducer:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.supported_tasks = TaskConfig.ABLATION_TASKS
        self.dim_reduction_config = config.DIMENSION_REDUCTION
        self.enable_linear = self.dim_reduction_config.get("ENABLE_LINEAR", True)
        self.target_dims = self.dim_reduction_config.get("TARGET_DIMS", {})
        self.input_activity_dir = config.FUSION_ACTIVITY_EMBEDDING_DIR
        self.input_widget_dir = config.FUSION_WIDGET_EMBEDDING_DIR
        self.base_output_dir = config.DIMENSION_REDUCED_BASE_DIR
        self.output_dirs: Dict[str, MethodDirInfo] = self._build_output_dirs()
        self.reducer_cache: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._validate_config()
        self._create_output_dirs()

    def _build_output_dirs(self) -> Dict[str, MethodDirInfo]:
        output_dirs: Dict[str, MethodDirInfo] = {}
        method = "LINEAR"
        method_root_dir = os.path.join(self.base_output_dir, method)
        output_dirs[method] = {
            "activity": {
                "base": os.path.join(method_root_dir, "activity"),
                "tasks": {}
            },
            "widget": {
                "base": os.path.join(method_root_dir, "widget"),
                "tasks": {}
            }
        }
        for task_name in self.supported_tasks.keys():
            output_dirs[method]["activity"]["tasks"][task_name] = os.path.join(
                output_dirs[method]["activity"]["base"], task_name
            )
            output_dirs[method]["widget"]["tasks"][task_name] = os.path.join(
                output_dirs[method]["widget"]["base"], task_name
            )
        return output_dirs

    def _validate_config(self) -> None:
        required_components = [
            "activity_name",
            "widget_name",
            "activity_summary",
            "widget_summary",
            "activity_simple_name",
            "activity_remove_suffix"
        ]
        missing_dims = [comp for comp in required_components if comp not in self.target_dims]
        for comp, dim in self.target_dims.items():
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"")

    def _create_output_dirs(self) -> None:
        for method, method_dir_info in self.output_dirs.items():
            for entity_type, entity_dir_info in method_dir_info.items():
                base_dir: str = entity_dir_info["base"]
                task_dirs: Dict[str, str] = entity_dir_info["tasks"]
                FileUtils.create_dir(base_dir)
                for task_name, task_dir_path in task_dirs.items():
                    FileUtils.create_dir(task_dir_path)
        for method in self.output_dirs.keys():
            method_root = os.path.join(self.base_output_dir, method.upper())

    def _get_component_original_dim(self, comp_name: str) -> int:
        summary_components = ["activity_summary", "widget_summary"]
        original_dim = 384 if comp_name in summary_components else 768
        return original_dim

    def _init_reducer(self, input_dim: int, output_dim: int) -> Callable[[np.ndarray], np.ndarray]:
        cache_key = f"LINEAR_{input_dim}_{output_dim}"
        if cache_key in self.reducer_cache:
            return self.reducer_cache[cache_key]
        np.random.seed(self.config.RANDOM_SEED)
        weight_matrix = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        reducer_: Callable[[np.ndarray], np.ndarray] = (lambda x: x @ weight_matrix)
        self.reducer_cache[cache_key] = reducer_
        return reducer_

    def _reduce_single_embedding(self, embedding: np.ndarray, components: List[str],
                                 entity_type: str) -> np.ndarray:
        reduced_components: List[np.ndarray] = []
        current_idx = 0
        actual_dim = embedding.shape[0]
        expected_total_dim = sum([self._get_component_original_dim(comp) for comp in components])
        if actual_dim != expected_total_dim:
            target_total_dim = sum([self.target_dims[comp] for comp in components])
            overall_reducer = self._init_reducer(actual_dim, target_total_dim)
            embedding = overall_reducer(embedding)
            split_sizes = [self.target_dims[comp] for comp in components]
            split_embeddings = np.split(embedding, np.cumsum(split_sizes)[:-1])
            return np.concatenate(split_embeddings, axis=0)
        for comp in components:
            comp_original_dim = self._get_component_original_dim(comp)
            comp_target_dim = self.target_dims[comp]
            comp_embedding = embedding[current_idx:current_idx + comp_original_dim]
            current_idx += comp_original_dim
            reducer = self._init_reducer(comp_original_dim, comp_target_dim)
            comp_reduced = reducer(comp_embedding)
            reduced_components.append(comp_reduced)
        final_embedding = np.concatenate(reduced_components, axis=0)
        return final_embedding

    def _process_single_entity_type(self, task_name: str, entity_type: str) -> None:

        input_base_dir = self.input_activity_dir if entity_type == "activity" else self.input_widget_dir
        input_task_dir = os.path.join(input_base_dir, task_name)
        output_task_dir = self.output_dirs["LINEAR"][entity_type]["tasks"][task_name]
        task_components = self.supported_tasks[task_name]["task_names"]
        activity_components = [comp for comp in task_components if comp.startswith("activity")]
        widget_components = [comp for comp in task_components if comp.startswith("widget")]
        current_components = activity_components if entity_type == "activity" else widget_components
        if not current_components:
            return
        app_files = FileUtils.get_files_by_suffix(input_task_dir, ".npz")
        if not app_files:
            return

        for app_file in tqdm(app_files, desc=f"  {entity_type}", leave=False):
            app_name = os.path.splitext(os.path.basename(app_file))[0]
            output_path = os.path.join(output_task_dir, f"{app_name}.npz")
            if os.path.exists(output_path):
                continue
            try:
                emb_dict = FileUtils.load_npz_dict(app_file)
                if not emb_dict:
                    continue
                reduced_emb_dict: Dict[str, np.ndarray] = {}
                for entity_id, original_emb in emb_dict.items():
                    reduced_emb = self._reduce_single_embedding(
                        embedding=original_emb,
                        components=current_components,
                        entity_type=entity_type
                    )
                    reduced_emb_dict[entity_id] = reduced_emb
                np.savez(output_path, **reduced_emb_dict)
            except Exception as e:
                continue

    def _process_single_task(self, task_name: str) -> None:
        self._process_single_entity_type(task_name, entity_type="activity")
        self._process_single_entity_type(task_name, entity_type="widget")

    def run_pipeline(self, tasks: Optional[Union[str, List[str]]] = None) -> None:

        if not self.enable_linear:
            return
        valid_tasks = TaskUtils.process_task_list(task_names=tasks, supported_tasks=self.supported_tasks)

        for task_idx, task_name in enumerate(valid_tasks, 1)
            try:
                self._process_single_task(task_name=task_name)
            except Exception as e:
                continue