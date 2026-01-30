import os
from typing import Optional, Union, List

from src.embedding.config.base_config import BaseConfig
from src.embedding.config.task_registry import TaskConfig, TaskUtils


class EmbeddingGenerator:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.supported_tasks = TaskConfig.GENERATOR_TASKS

    def _generate_single_task(self, task_name: str) -> None:
        meta = self.supported_tasks[task_name]
        embedder_cls = meta["embedder_cls"]
        filename = meta["filename"]

        input_file = os.path.join(self.config.RAW_DATA_TXT_DIR, filename)
        if not os.path.exists(input_file):
            return
        embedder = embedder_cls(config=self.config, task_name=task_name)
        embedder.process(input_file=input_file)

    def run_pipeline(self, tasks: Optional[Union[str, List[str]]] = None) -> None:

        valid_tasks = TaskUtils.process_task_list(task_names=tasks, supported_tasks=self.supported_tasks)
        if not valid_tasks:
            return
        for task in valid_tasks:
            self._generate_single_task(task)
