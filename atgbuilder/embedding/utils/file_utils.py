import os
import re
from typing import Dict, List, Any, Optional, Union

import numpy as np
import yaml


class FileUtils:
    @staticmethod
    def load_data(file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.rstrip("\n") for line in f.readlines()]

    @staticmethod
    def get_files_by_suffix(dir_path: str, suffix: str = ".txt") -> List[str]:
        if not os.path.exists(dir_path):
            return []
        file_list = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(suffix) and os.path.isfile(os.path.join(dir_path, f))
        ]
        return sorted(file_list)

    @staticmethod
    def read_yaml(yaml_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML：{yaml_path}")
        with open(yaml_path, "r", encoding=encoding) as f:
            return yaml.safe_load(f)

    @staticmethod
    def resolve_yaml_variables(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        pattern = re.compile(r"\$\{(\w+)}")

        def resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                while True:
                    match = pattern.search(value)
                    if not match:
                        break
                    var_name = match.group(1)
                    value = value.replace(match.group(0), str(config_dict[var_name]))
                return value
            if isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        return resolve_value(config_dict.copy())

    @staticmethod
    def load_config(yaml_path: str, resolve_variables: bool = True) -> Dict[str, Any]:
        config_dict = FileUtils.read_yaml(yaml_path)
        return FileUtils.resolve_yaml_variables(config_dict) if resolve_variables else config_dict

    @staticmethod
    def load_npz_dict(npz_path: str) -> Dict[str, np.ndarray]:
        with np.load(npz_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}

    @staticmethod
    def create_dir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)


