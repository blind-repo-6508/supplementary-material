# src/embedding/pipeline/run_pipeline.py
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Any

from src.embedding.config.base_config import BaseConfig
from src.embedding.pipeline.atg_pipeline import ATGPipeline, PipelineRunConfig
from src.embedding.utils.file_utils import FileUtils


def _repo_root() -> Path:
    # .../src/embedding/pipeline/run_pipeline.py -> repo root = parents[3]
    return Path(__file__).resolve().parents[3]


def _default_pipeline_cfg_candidates() -> list[str]:
    repo = _repo_root()
    pkg_cfg = repo / "src" / "embedding" / "config"
    return [
        str(pkg_cfg / "pipeline_config.yaml"),
        str(repo / "config" / "pipeline_config.yaml"),
    ]


def _normalize_task_list(v):
    # None/"null" => None（表示跑全部）
    if v is None or v == "null":
        return None
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return v
    return [str(v)]


def load_pipeline_config(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    if not yaml_path:
        candidates = _default_pipeline_cfg_candidates()
        yaml_path = next((p for p in candidates if os.path.exists(p)), None)

    cfg = FileUtils.load_config(yaml_path, resolve_variables=False) or {}

    defaults = {
        "run_generate": True,
        "generation_tasks": None,

        "run_ablation_fusion": True,
        "ablation_tasks": None,

        "run_dim_reduction": True,
        "dim_reduction_tasks": None,

        "run_build_pack": False,
        "build_pack_tasks": None,

        "reduction_method": "LINEAR",

        "pack_dir_name": "packed_data_tmp",
        "graphs_per_file": 100,
        "use_packed_data": True,
        "force_rebuild_pack": False,

        "require_confirm": True,
    }
    for k, dv in defaults.items():
        cfg.setdefault(k, dv)

    cfg["generation_tasks"] = _normalize_task_list(cfg.get("generation_tasks"))
    cfg["ablation_tasks"] = _normalize_task_list(cfg.get("ablation_tasks"))
    cfg["dim_reduction_tasks"] = _normalize_task_list(cfg.get("dim_reduction_tasks"))
    cfg["build_pack_tasks"] = _normalize_task_list(cfg.get("build_pack_tasks"))

    return cfg


def print_execution_plan(cfg: Dict[str, Any]) -> None:
    def show_tasks(x):
        return "task" if x is None else x

    print(f"   {'✅' if cfg['run_generate'] else '❌'}  tasks={show_tasks(cfg['generation_tasks'])}")
    print(f"   {'✅' if cfg['run_ablation_fusion'] else '❌'}  tasks={show_tasks(cfg['ablation_tasks'])}")
    print(f"   {'✅' if cfg['run_dim_reduction'] else '❌'}  tasks={show_tasks(cfg['dim_reduction_tasks'])}")
    print(f"   {'✅' if cfg['run_build_pack'] else '❌'}  tasks={show_tasks(cfg['build_pack_tasks'])}")
    print(f"     reduction_method={cfg['reduction_method']}, graphs_per_file={cfg['graphs_per_file']}, pack_dir={cfg['pack_dir_name']}")
    print(f"     use_packed_data={cfg['use_packed_data']}, force_rebuild_pack={cfg['force_rebuild_pack']}")


def setup_environment():
    repo = str(_repo_root())
    if repo not in sys.path:
        sys.path.insert(0, repo)

    import random
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def main(pipeline_config_yaml_path: Optional[str] = None):

    setup_environment()
    run_params = load_pipeline_config(pipeline_config_yaml_path)
    print_execution_plan(run_params)


    base_config = BaseConfig.from_yaml()
    pipeline = ATGPipeline(config=base_config)

    pipeline_run_config = PipelineRunConfig(**run_params)
    results = pipeline.run(pipeline_run_config)

    print("\n🎉 全流程执行完成！")
    return results


if __name__ == "__main__":
    main(None)