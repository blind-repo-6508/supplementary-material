from typing import Dict, Optional, Union, List, Any

from src.embedding.core.bert_embedding import BertEmbedding
from src.embedding.core.sentence_bert_embedding import SentenceBertEmbedding


class TaskConfig:

    GENERATOR_TASKS: Dict[str, Dict] = {
        "activity_name": {
            "embedder_cls": BertEmbedding,
            "filename": "extracted_activities.txt"
        },
        "widget_name": {
            "embedder_cls": BertEmbedding,
            "filename": "extract_widget_names.txt"
        },
        "activity_summary": {
            "embedder_cls": SentenceBertEmbedding,
            "filename": "extracted_activity_summaries.txt"
        },
        "widget_summary": {
            "embedder_cls": SentenceBertEmbedding,
            "filename": "extract_widget_summaries.txt"
        },
        "activity_simple_name": {
            "embedder_cls": BertEmbedding,
            "filename": "extracted_activity_simple_names.txt"
        },
        "activity_remove_suffix": {
            "embedder_cls": BertEmbedding,
            "filename": "extracted_activity_remove_suffix.txt"
        }
    }

    ABLATION_TASKS: Dict[str, Dict] = {
        # baseline
        "ab_baseline": {
            "task_names": ["activity_name", "widget_name"],
        },
        # complete task
        "ab_complete": {
            "task_names": ["activity_name", "activity_summary", "widget_name", "widget_summary"],
        },
        # widget ablations
        "ab_widget_none": {
            "task_names": ["activity_name", "activity_summary"],
        },
        "ab_widget_only_name": {
            "task_names": ["activity_name", "activity_summary", "widget_name"],
        },
        "ab_widget_only_summary": {
            "task_names": ["activity_name", "activity_summary", "widget_summary"],
        },
        # activity ablations
        "ab_activity_only_name": {
            "task_names": ["activity_name", "widget_name", "widget_summary"],
        },
        "ab_activity_only_summary": {
            "task_names": ["activity_summary", "widget_name", "widget_summary"],
        },
        # simple name
        "ab_activity_simple_name": {
            "task_names": ["activity_simple_name", "activity_summary", "widget_name", "widget_summary"],
        },
        # remove activity suffix
        "ab_activity_remove_suffix": {
            "task_names": ["activity_remove_suffix", "activity_summary", "widget_name", "widget_summary"],
        }
    }


class TaskUtils:

    @staticmethod
    def process_task_list(task_names: Optional[Union[str, List[str]]], supported_tasks: Dict[str, Any]) -> Optional[
        List[str]]:
        if task_names is None:
            return list(supported_tasks.keys())

        if isinstance(task_names, str):
            task_names = [task_names]

        task_names = list(set(task_names))
        valid_tasks = [task for task in task_names if task in supported_tasks]

        return valid_tasks
