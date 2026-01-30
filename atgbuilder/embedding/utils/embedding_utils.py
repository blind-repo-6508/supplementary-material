from typing import Optional, Callable, List, Dict

from src.embedding.config.base_config import BaseConfig
from src.embedding.utils.file_utils import FileUtils


class EntityProcessor:
    def __init__(
            self,
            entity_type: str,
            config: BaseConfig,
            split_number: int,
            keyword_extractor: Callable[[List[str]], Dict[str, str]],
    ):
        self.entity_type = entity_type.lower()
        self.config = config
        self.split_number = int(split_number)
        self.keyword_extractor = keyword_extractor

        if self.entity_type not in ("activity", "widget"):
            raise ValueError(f"activity/widget:{entity_type}")
        if self.split_number <= 0:
            raise ValueError(f"split_number  > 0，: {split_number}")

    def _placeholder_lines(self) -> List[str]:

        if self.entity_type == "widget":
            entity_id = "NONE_WIDGET"
            defaults = ["NONE_WIDGET", "NONE_SUMMARY"]
        else:
            entity_id = "NONE_ACTIVITY"
            defaults = ["NONE_ACTIVITY", "NONE_SUMMARY"]

        parts = (defaults + ["NONE"] * self.split_number)[: self.split_number]
        data_line = ";".join(parts)
        return [f">{entity_id}", data_line]

    def parse_data(self, file_path: str) -> Optional[List[Dict[str, str]]]:
        if self.entity_type == "widget" and not os.path.exists(file_path):
            file_lines = self._placeholder_lines()
        else:
            file_lines = FileUtils.load_data(file_path)

        id_data_pairs = EmbeddingTaskUtils.parse_id_data_pairs(file_lines)
        if not id_data_pairs:
            return None
        return id_data_pairs

    def get_task_output_dir(self, task_name: str) -> str:
        out_dir = os.path.join(self.config.FUSION_EMBEDDING_DIR, self.entity_type, task_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def get_output_path(self, app_name: str, task_name: str) -> str:

        if not isinstance(app_name, str) or not app_name.strip():
            raise ValueError(f"Invalid app_name: {app_name}")

        out_dir = self.get_task_output_dir(task_name)
        return os.path.join(out_dir, f"{app_name.strip()}.npz")


import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.embedding.config.base_config import BaseConfig


class EmbeddingReader:

    @staticmethod
    def _get_embedding_dir(config: BaseConfig, task_name: str) -> str:
        return os.path.join(config.EMBEDDING_OUTPUT_DIR, task_name)

    @staticmethod
    def load_index(config: BaseConfig, task_name: str) -> pd.DataFrame:
        embedding_dir = EmbeddingReader._get_embedding_dir(config, task_name)
        index_file = os.path.join(embedding_dir, "total_index.txt")

        df = pd.read_csv(
                index_file,
                sep="\t",
                encoding="utf-8",
                dtype={"batch_idx": str, "array_index": int}
            )
        return df

    @staticmethod
    def _build_text_position_map(config: BaseConfig, task_name: str) -> Dict[str, Tuple[str, int]]:
        index_df = EmbeddingReader.load_index(config, task_name)
        text_map = {}
        for _, row in index_df.iterrows():
            text = row["text"]
            batch_idx = row["batch_idx"]
            array_idx = row["array_index"]
            text_map[text] = (batch_idx, array_idx)
        return text_map

    @staticmethod
    def get_batch_file_path(config: BaseConfig, task_name: str, batch_idx: str) -> str:
        embedding_dir = EmbeddingReader._get_embedding_dir(config, task_name)
        return os.path.join(embedding_dir, f"embeddings_batch_{batch_idx}.npy")

    @staticmethod
    def read_npz_emb_to_dict(npz_path: str) -> Dict[str, np.ndarray]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as npz_file:
            return {key: npz_file[key] for key in npz_file.files}

    @staticmethod
    def load_embedding(
            config: BaseConfig,
            app_name: str,
            task: str,
            entity_type: str,
            reduction_method: str = "LINEAR",
    ) -> Tuple[Dict[str, np.ndarray], str]:

        reduction_method = (reduction_method or "LINEAR").upper()
        entity_type = entity_type.lower()

        if entity_type not in ("activity", "widget"):
            raise ValueError(f"entity_type activity/widget: {entity_type}")

        if reduction_method == "ORIGINAL":
            base_dir = os.path.join(config.FUSION_EMBEDDING_DIR, entity_type)
        else:
            base_dir = os.path.join(config.DIMENSION_REDUCED_BASE_DIR, reduction_method, entity_type)

        emb_path = os.path.join(base_dir, task, f"{app_name}.npz")

        return EmbeddingReader.read_npz_emb_to_dict(emb_path), emb_path


from typing import List, Dict, Tuple


class EmbeddingTaskUtils:

    @staticmethod
    def parse_id_data_pairs(file_lines: List[str]) -> List[Dict[str, str]]:
        parsed_pairs = []
        current_entity_id = None

        for line_idx, line in enumerate(file_lines, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_entity_id = line.lstrip(">").strip()
                continue
            if current_entity_id:
                parsed_pairs.append({"id": current_entity_id, "data": line})
                current_entity_id = None

        return parsed_pairs

    @staticmethod
    def generate_activity_derived_attrs(original_activity_name: str) -> Tuple[str, str]:
        simple_name = original_activity_name.split(".")[-1]

        activity_suffixes = {"Activity", "activity"}
        for suffix in activity_suffixes:
            if original_activity_name.endswith(suffix) and len(original_activity_name) > len(suffix):
                original_activity_name = original_activity_name[:-len(suffix)]
                break
        remove_suffix = original_activity_name

        return simple_name, remove_suffix

    @staticmethod
    def extract_activity_keywords(data_parts: List[str]) -> Dict[str, str]:
        name, summary = data_parts[0].strip(), data_parts[1].strip()
        simple_name, remove_suffix = EmbeddingTaskUtils.generate_activity_derived_attrs(name)
        return {
            "activity_name": name,
            "activity_summary": summary,
            "activity_simple_name": simple_name,
            "activity_remove_suffix": remove_suffix
        }

    @staticmethod
    def extract_widget_keywords(data_parts: List[str]) -> Dict[str, str]:
        name = data_parts[0].strip() if len(data_parts) > 0 else "NONE_WIDGET"
        summary = data_parts[1].strip() if len(data_parts) > 1 else "NONE_SUMMARY"
        return {"widget_name": name, "widget_summary": summary}
