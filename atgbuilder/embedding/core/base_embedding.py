import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.embedding.config.base_config import BaseConfig


class BaseEmbedding(ABC):

    def __init__(self, config: BaseConfig, task_name: str):
        self.config = config
        self.task_name = task_name
        self.output_dir = os.path.join(config.EMBEDDING_OUTPUT_DIR, task_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.DEVICE_ID}")
            torch.cuda.empty_cache()
        else:
            device = torch.device("cpu")
        return device

    @abstractmethod
    def load_data(self, input_file: str) -> List[str]:
        pass

    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        pass

    def save_batch(self, batch_texts: List[str], batch_embeddings: np.ndarray, batch_idx: int) -> Tuple[
        bool, Tuple[str, str]]:
        embedding_file = os.path.join(self.output_dir, f"embeddings_batch_{batch_idx:04d}.npy")
        meta_file = os.path.join(self.output_dir, f"meta_batch_{batch_idx:04d}.txt")

        try:
            # ✅ 强制按配置保存精度
            prec = str(getattr(self.config, "PRECISION", "float32")).lower()
            if prec in ("float16", "fp16", "half"):
                batch_embeddings = batch_embeddings.astype(np.float16, copy=False)
            elif prec in ("bfloat16", "bf16"):
                batch_embeddings = batch_embeddings.astype(np.float16, copy=False)
            else:
                batch_embeddings = batch_embeddings.astype(np.float32, copy=False)

            np.save(embedding_file, batch_embeddings)

            with open(meta_file, "w", encoding="utf-8") as f:
                for text in batch_texts:
                    f.write(f"{text}\n")
            return True, (embedding_file, meta_file)
        except Exception as e:
            if os.path.exists(embedding_file):
                os.remove(embedding_file)
            if os.path.exists(meta_file):
                os.remove(meta_file)
            return False, (embedding_file, meta_file)

    def generate_total_index(self, all_texts: List[str]) -> bool:
        index_file = os.path.join(self.output_dir, "total_index.txt")
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                f.write("text\tbatch_idx\tarray_index\n")
                for global_idx, text in enumerate(all_texts):
                    batch_idx = global_idx // self.config.BATCH_SIZE
                    array_idx = global_idx % self.config.BATCH_SIZE
                    f.write(f"{text}\t{batch_idx:04d}\t{array_idx}\n")
            return True
        except Exception as e:
            return False

    def process(self, input_file: str) -> None:
        texts = self.load_data(input_file)
        total_count = len(texts)
        if total_count == 0:
            return

        texts = [text.strip() for text in texts]
        batch_count = (total_count + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        success_batches = 0

        for batch_idx in tqdm(
                range(batch_count),
                desc=f"📋 {self.task_name}",
                ncols=150,
                colour='green',
                unit='batch',
                dynamic_ncols=False
        ):
            start = batch_idx * self.config.BATCH_SIZE
            end = min((batch_idx + 1) * self.config.BATCH_SIZE, total_count)
            batch_texts = texts[start:end]

            try:
                batch_embeddings = self.generate_embeddings(batch_texts)
                save_success, _ = self.save_batch(batch_texts, batch_embeddings, batch_idx)
                if save_success:
                    success_batches += 1
            except Exception as e:
                print(f"\n❌ 批次 {batch_idx} 失败: {str(e)[:50]}...")
                continue

        if success_batches > 0:
            self.generate_total_index(texts)
