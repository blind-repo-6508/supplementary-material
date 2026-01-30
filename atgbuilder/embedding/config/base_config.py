# src/embedding/config/base_config.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Any

from src.embedding.utils.file_utils import FileUtils

TargetDimsType = Dict[str, int]
DimensionReductionConfig = Dict[str, Union[bool, str, TargetDimsType]]


def _repo_root() -> Path:
    # .../src/embedding/config/base_config.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_embedding_cfg_candidates() -> list[str]:
    repo = _repo_root()
    pkg_cfg = Path(__file__).resolve().parent  # src/embedding/config
    return [
        str(pkg_cfg / "embedding_config.yaml"),
        str(repo / "config" / "embedding.config.yaml"),
        str(repo / "config" / "embedding_config.yaml"),
        str(repo / "config" / "config.yaml"),  # legacy
    ]


@dataclass(frozen=True)
class BaseConfig:
    # ========== Paths ==========
    ROOT_DIR: str
    BERT_MODEL_PATH: str
    ALL_MINILM_L12_V2_MODEL_PATH: str

    DATA_DIR: str
    EMBEDDING_OUTPUT_DIR: str
    FUSION_EMBEDDING_DIR: str
    FUSION_ACTIVITY_EMBEDDING_DIR: str
    FUSION_WIDGET_EMBEDDING_DIR: str

    RAW_DATA_TXT_DIR: str
    ACTIVITY_TXT_DIR: str
    WIDGET_TXT_DIR: str
    ATG_INDEX_DIR: str

    DIMENSION_REDUCED_BASE_DIR: str

    # ========== Runtime ==========
    DEVICE_ID: int
    BATCH_SIZE: int
    PRECISION: str

    DEFAULT_MAX_LENGTH: int
    TASK_MAX_LENGTH: Dict[str, int]

    RANDOM_SEED: int
    DIMENSION_REDUCTION: DimensionReductionConfig

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "BaseConfig":
        if not yaml_path:
            candidates = _default_embedding_cfg_candidates()
            yaml_path = next((p for p in candidates if os.path.exists(p)), None)

        cfg_dict = FileUtils.load_config(yaml_path=yaml_path, resolve_variables=True) or {}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k != "MODEL"}  # legacy compat

        cls._validate_and_prepare_dirs(cfg_dict)
        return cls(**cfg_dict)

    @staticmethod
    def _validate_and_prepare_dirs(cfg: Dict[str, Any]) -> None:
        for k in [
            "EMBEDDING_OUTPUT_DIR",
            "FUSION_EMBEDDING_DIR",
            "FUSION_ACTIVITY_EMBEDDING_DIR",
            "FUSION_WIDGET_EMBEDDING_DIR",
            "DIMENSION_REDUCED_BASE_DIR",
        ]:
            v = cfg.get(k)
            if v:
                os.makedirs(v, exist_ok=True)

        raw_dir = cfg.get("RAW_DATA_TXT_DIR")
        atg_dir = cfg.get("ATG_INDEX_DIR")
        if not raw_dir or not os.path.exists(raw_dir):
            raise FileNotFoundError(f"RAW_DATA_TXT_DIR : {raw_dir}")
        if not atg_dir or not os.path.exists(atg_dir):
            raise FileNotFoundError(f"ATG_INDEX_DIR : {atg_dir}")

        dr = cfg.get("DIMENSION_REDUCTION") or {}

        target_dims = dr.get("TARGET_DIMS") or {}
        required = [
            "activity_name", "widget_name",
            "activity_summary", "widget_summary",
            "activity_simple_name", "activity_remove_suffix",
        ]
        missing = [k for k in required if k not in target_dims]
        if missing:
            raise KeyError(f"DIMENSION_REDUCTION.TARGET_DIMS: {missing}")

    def get_max_length(self, task_name: Optional[str] = None) -> int:
        if task_name and task_name in self.TASK_MAX_LENGTH:
            return int(self.TASK_MAX_LENGTH[task_name])
        return int(self.DEFAULT_MAX_LENGTH)