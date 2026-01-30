# src/embedding/pipeline/atg_pipeline.py
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any

from src.embedding.config.base_config import BaseConfig
from src.embedding.components.embedding_generator import EmbeddingGenerator
from src.embedding.components.ablation_embedding_fuser import AblationEmbeddingFuser
from src.embedding.components.dimension_reducer import DimensionReducer
from src.embedding.components.seed_atg_builder_packer import SeedAtgBuilderPacker


@dataclass
class PipelineRunConfig:
    run_generate: bool = False
    run_ablation_fusion: bool = False
    run_dim_reduction: bool = False
    run_build_pack: bool = False

    generation_tasks: Optional[Union[str, List[str]]] = None
    ablation_tasks: Optional[Union[str, List[str]]] = None
    dim_reduction_tasks: Optional[Union[str, List[str]]] = None
    build_pack_tasks: Optional[Union[str, List[str]]] = None

    reduction_method: str = "LINEAR"

    pack_dir_name: str = "packed_data_tmp"
    graphs_per_file: int = 100
    use_packed_data: bool = True
    force_rebuild_pack: bool = False

    require_confirm: bool = False


def _print_step(step_num: int, total_steps: int, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"【{step_num}/{total_steps}】{name}")
    print("=" * 60)


def _normalize_tasks(tasks: Optional[Union[str, List[str]]]) -> Optional[List[str]]:

    if tasks is None:
        return None
    if isinstance(tasks, str):
        return [tasks]
    return list(tasks)


class ATGPipeline:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.embedding_generator = EmbeddingGenerator(config=self.config)
        self.ablation_embedding_fuser = AblationEmbeddingFuser(config=self.config)
        self.dimension_reducer = DimensionReducer(config=self.config)

    # ---------- Step 1 ----------
    def step1_generate(self, tasks: Optional[Union[str, List[str]]] = None) -> Any:
        tasks = _normalize_tasks(tasks)
        return self.embedding_generator.run_pipeline(tasks=tasks)

    # ---------- Step 2 ----------
    def step2_ablation(self, tasks: Optional[Union[str, List[str]]] = None) -> Any:
        tasks = _normalize_tasks(tasks)
        return self.ablation_embedding_fuser.run_pipeline(task_names=tasks)

    # ---------- Step 3 ----------
    def step3_reduce(self, tasks: Optional[Union[str, List[str]]] = None) -> Any:
        tasks = _normalize_tasks(tasks)
        return self.dimension_reducer.run_pipeline(tasks=tasks)

    # ---------- Step 4 ----------
    def step4_build_pack(
        self,
        tasks: Optional[Union[str, List[str]]] = None,
        reduction_method: str = "LINEAR",
        pack_dir_name: str = "packed_data_tmp",
        graphs_per_file: int = 100,
        use_packed_data: bool = True,
        force_rebuild_pack: bool = False,
    ) -> Dict[str, str]:
        tasks = _normalize_tasks(tasks)
        builder = SeedAtgBuilderPacker(
            config=self.config,
            reduction_method=reduction_method,
            use_packed_data=use_packed_data,
            pack_dir_name=pack_dir_name,
            graphs_per_file=graphs_per_file,
            candidate_mode="atg",
        )
        return builder.run_pipeline(ablation_tasks=tasks, force_rebuild=force_rebuild_pack)

    def run(self, run_config: PipelineRunConfig) -> Dict[str, Any]:

        steps = []
        if run_config.run_generate:
            steps.append(("embed", self.step1_generate, {"tasks": run_config.generation_tasks}))
        if run_config.run_ablation_fusion:
            steps.append(("fusion", self.step2_ablation, {"tasks": run_config.ablation_tasks}))
        if run_config.run_dim_reduction:
            steps.append(("reduce", self.step3_reduce, {"tasks": run_config.dim_reduction_tasks}))
        if run_config.run_build_pack:
            steps.append(("package", self.step4_build_pack, {
                "tasks": run_config.build_pack_tasks,
                "reduction_method": run_config.reduction_method,
                "pack_dir_name": run_config.pack_dir_name,
                "graphs_per_file": run_config.graphs_per_file,
                "use_packed_data": run_config.use_packed_data,
                "force_rebuild_pack": run_config.force_rebuild_pack,
            }))

        total = len(steps)
        if total == 0:
            return {}

        results: Dict[str, Any] = {}
        for i, (name, fn, kwargs) in enumerate(steps, 1):
            _print_step(i, total, name)
            results[name] = fn(**kwargs)

        return results