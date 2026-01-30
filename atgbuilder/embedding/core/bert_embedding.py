import torch
import numpy as np
from typing import List, Tuple
from transformers import BertTokenizer, BertModel
from src.embedding.config.base_config import BaseConfig
from src.embedding.core.base_embedding import BaseEmbedding
from src.embedding.utils.file_utils import FileUtils


class BertEmbedding(BaseEmbedding):

    def __init__(self, config: BaseConfig, task_name: str):
        super().__init__(config, task_name)
        self.device = super()._init_device()
        self.tokenizer, self.model = self._load_model()
        self.max_length = self.config.get_max_length(task_name)

    def _load_model(self) -> Tuple[BertTokenizer, BertModel]:

        model_path = self.config.BERT_MODEL_PATH
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)
        model = model.to(self.device).eval()
        return tokenizer, model

    def load_data(self, input_file: str) -> List[str]:
        return FileUtils.load_data(input_file)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        inputs = self.tokenizer(
            text=texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            cls_vectors = outputs.last_hidden_state[:, 0, :]

        if self.config.PRECISION == "float16":
            cls_vectors = cls_vectors.half()
        else:
            cls_vectors = cls_vectors.float()

        return cls_vectors.cpu().numpy()