from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.embedding.config.base_config import BaseConfig
from src.embedding.core.base_embedding import BaseEmbedding
from src.embedding.utils.file_utils import FileUtils


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceBertEmbedding(BaseEmbedding):

    def __init__(self, config: BaseConfig, task_name: str):
        super().__init__(config, task_name)
        self.device = super()._init_device()
        self.tokenizer, self.model = self._load_sentence_model()
        self.max_length = self.config.get_max_length(task_name)

    def _load_sentence_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        model_path = self.config.ALL_MINILM_L12_V2_MODEL_PATH
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model = model.to(self.device).eval()
        return tokenizer, model

    def load_data(self, input_file: str) -> List[str]:
        return FileUtils.load_data(input_file)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        encoded_input = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = _mean_pooling(model_output, encoded_input["attention_mask"])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        if self.config.PRECISION == "float16":
            sentence_embeddings = sentence_embeddings.half()
        else:
            sentence_embeddings = sentence_embeddings.float()

        embeddings_np = sentence_embeddings.cpu().numpy()
        return embeddings_np

