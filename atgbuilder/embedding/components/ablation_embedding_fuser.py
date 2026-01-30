import os
from typing import Tuple, Optional, List, Dict, Union, Callable

import numpy as np
from tqdm import tqdm

from src.embedding.config.base_config import BaseConfig
from src.embedding.config.task_registry import TaskConfig, TaskUtils
from src.embedding.utils.embedding_utils import EntityProcessor, EmbeddingTaskUtils, EmbeddingReader
from src.embedding.utils.file_utils import FileUtils


class AblationEmbeddingFuser:

    def __init__(self, config: BaseConfig):
        self.config = config
        self.supported_tasks = TaskConfig.ABLATION_TASKS
        self._task_text_pos_cache = {}
        self._task_index_cache = {}

        self.activity_processor = EntityProcessor(
            entity_type="activity",
            config=self.config,
            split_number=2,
            keyword_extractor=EmbeddingTaskUtils.extract_activity_keywords
        )
        self.widget_processor = EntityProcessor(
            entity_type="widget",
            config=self.config,
            split_number=2,
            keyword_extractor=EmbeddingTaskUtils.extract_widget_keywords
        )

    def _get_task_text_pos_map(self, task_name: str) -> Dict[str, Tuple[str, int]]:
        if task_name in self._task_text_pos_cache:
            return self._task_text_pos_cache[task_name]
        if task_name not in self._task_index_cache:
            try:
                self._task_index_cache[task_name] = EmbeddingReader.load_index(self.config, task_name)
            except Exception as e:
                print(f"❌  [{task_name}] ：{str(e)}")
                self._task_text_pos_cache[task_name] = {}
                return {}
        text_pos_map = {}
        for _, row in self._task_index_cache[task_name].iterrows():
            text_pos_map[row["text"]] = (row["batch_idx"], row["array_index"])
        self._task_text_pos_cache[task_name] = text_pos_map
        return text_pos_map

    def _batch_load_embeddings(self, task_name: str, text_keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        text_pos_map = self._get_task_text_pos_map(task_name)
        if not text_pos_map:
            return {key: None for key in text_keys}
        valid_texts = [key for key in text_keys if key in text_pos_map]
        if not valid_texts:
            return {key: None for key in text_keys}
        batch_group: Dict[str, List[Tuple[str, int]]] = {}
        for key in valid_texts:
            batch_idx, array_idx = text_pos_map[key]
            batch_group.setdefault(batch_idx, []).append((key, array_idx))
        text_emb_map = {key: None for key in text_keys}
        for batch_idx, items in batch_group.items():
            batch_file = EmbeddingReader.get_batch_file_path(self.config, task_name, batch_idx)
            try:
                batch_emb = np.load(batch_file, allow_pickle=False)
                for key, array_idx in items:
                    text_emb_map[key] = batch_emb[array_idx].astype(self.config.PRECISION)
            except Exception as e:
                continue
        return text_emb_map

    def _fuse_entity_embeddings(self,
                                entity_pairs: List[Dict[str, str]],
                                target_tasks: List[str],
                                split_number: int,
                                entity_type: str,
                                keyword_extractor: Callable[[List[str]], Dict[str, str]]) -> Dict[str, np.ndarray]:
        fused_result = {}
        if not target_tasks or not entity_pairs:
            return fused_result
        task_keywords = {task: [] for task in target_tasks}
        entity_task_map = {}
        for pair in entity_pairs:
            entity_id, data_line = pair["id"], pair["data"]
            data_parts = data_line.split(";", maxsplit=split_number - 1)
            if len(data_parts) != split_number:
                print(f"⚠️  {entity_type}（ID：{entity_id}）：{split_number}/{len(data_parts)}个")
                continue
            try:
                all_keywords = keyword_extractor(data_parts)
            except Exception as e:
                print(f"⚠️  {entity_type}（ID：{entity_id}）：{str(e)}")
                continue
            entity_keyword_map = {}
            for task in target_tasks:
                if task not in all_keywords or not all_keywords[task]:
                    continue
                keyword = all_keywords[task]
                entity_keyword_map[task] = keyword
                if keyword not in task_keywords[task]:
                    task_keywords[task].append(keyword)
            if entity_keyword_map:
                entity_task_map[entity_id] = entity_keyword_map
        task_embeddings = {task: self._batch_load_embeddings(task, task_keywords[task]) for task in target_tasks}
        for entity_id, task_key_map in entity_task_map.items():
            emb_list = []
            is_valid = True
            for task in target_tasks:
                keyword = task_key_map[task]
                embedding = task_embeddings[task].get(keyword)
                if embedding is None:
                    is_valid = False
                    break
                emb_list.append(embedding)
            if is_valid and emb_list:
                fused_result[entity_id] = np.concatenate(emb_list, axis=0)
        return fused_result

    def _process_single_file_fusion(self, app_name: str, ablation_config: Dict) -> Tuple[
        Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        activity_file = os.path.join(self.config.ACTIVITY_TXT_DIR, f"{app_name}.txt")
        widget_file = os.path.join(self.config.WIDGET_TXT_DIR, f"{app_name}.txt")

        activity_pairs = self.activity_processor.parse_data(activity_file)
        widget_pairs = self.widget_processor.parse_data(widget_file)

        activity_tasks = [t for t in ablation_config["task_names"] if t.startswith("activity")]
        widget_tasks = [t for t in ablation_config["task_names"] if t.startswith("widget")]

        activity_fused = self._fuse_entity_embeddings(
            entity_pairs=activity_pairs,
            target_tasks=activity_tasks,
            split_number=self.activity_processor.split_number,
            entity_type=self.activity_processor.entity_type,
            keyword_extractor=self.activity_processor.keyword_extractor
        )
        widget_fused = self._fuse_entity_embeddings(
            entity_pairs=widget_pairs,
            target_tasks=widget_tasks,
            split_number=self.widget_processor.split_number,
            entity_type=self.widget_processor.entity_type,
            keyword_extractor=self.widget_processor.keyword_extractor
        )

        return activity_fused, widget_fused

    def _process_single_ablation_task(self, task_name: str, app_names: List[str]) -> None:
        ablation_config = self.supported_tasks[task_name]

        for app_idx, app_name in enumerate(tqdm(
                app_names,
                desc=f"[{task_name}]",
                ncols=150,
                colour='green',
                dynamic_ncols=False
        ), 1):

            activity_fused, widget_fused = self._process_single_file_fusion(app_name=app_name,
                                                                            ablation_config=ablation_config)
            if activity_fused:
                activity_output_path = self.activity_processor.get_output_path(app_name, task_name)
                np.savez_compressed(activity_output_path, **activity_fused)
            if widget_fused:
                widget_output_path = self.widget_processor.get_output_path(app_name, task_name)
                np.savez_compressed(widget_output_path, **widget_fused)

    def run_pipeline(self, task_names: Union[str, List[str]]) -> None:
        valid_tasks = TaskUtils.process_task_list(
            task_names=task_names,
            supported_tasks=self.supported_tasks
        )
        if not valid_tasks:
            return

        apps_absolute_dir = FileUtils.get_files_by_suffix(self.config.ACTIVITY_TXT_DIR, ".txt")
        app_names = [os.path.splitext(os.path.basename(app_dir))[0] for app_dir in apps_absolute_dir]

        if not app_names:
            return


        for task_idx, task_name in enumerate(valid_tasks, 1):
            self._process_single_ablation_task(task_name=task_name, app_names=app_names)

