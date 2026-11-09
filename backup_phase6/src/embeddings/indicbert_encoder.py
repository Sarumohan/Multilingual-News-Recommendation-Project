"""
Assamese and Indic Multilingual Text Representation Module using AI4Bharat IndicBERT.
Implements attention-mask-aware mean pooling and L2 normalization for genuine
multilingual representation across Assamese, Hindi, and English.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel

from src.utils.helpers import setup_logger

logger = setup_logger("IndicBERTEncoder")

INDICBERT_MODEL_NAME = "ai4bharat/IndicBERTv2-SS"
INDICBERT_EMBEDDING_DIM = 768

class IndicBERTEncoder:
    """
    Multilingual encoder leveraging AI4Bharat IndicBERTv2.
    Explicitly supports Assamese ('as'), Hindi ('hi'), and English ('en')
    with 200,000 vocabulary tokens and attention-mask-aware mean pooling.
    """

    def __init__(
        self,
        model_name: str = INDICBERT_MODEL_NAME,
        device: Optional[str] = None,
        max_seq_length: int = 128
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.embedding_dim = INDICBERT_EMBEDDING_DIM

        if "HF_HUB_DISABLE_SYMLINKS_WARNING" not in os.environ:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        logger.info(f"Initializing IndicBERT Encoder with '{self.model_name}' on device '{self.device}'...")

        try:
            # Attempt loading from local cache first to ensure offline reliability
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            logger.info("Loaded IndicBERT model from local cache.")
        except Exception:
            # Fall back to online loading if cache miss
            logger.info("Loading IndicBERT model from Hugging Face hub...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"IndicBERT initialized successfully. Hidden size: {self.embedding_dim}")

    def mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-mask-aware mean pooling over token representations.
        Prevents padding tokens from skewing the final sentence embedding.
        """
        token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_size)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode single text or list of texts into dense 768-dimensional NumPy embeddings.
        Applies attention-mask-aware mean pooling and L2 normalization.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        # Sanitize empty/whitespace inputs
        sanitized = []
        empty_indices = set()
        for idx, t in enumerate(text_list):
            if t is None or not isinstance(t, str) or t.strip() == "":
                sanitized.append("")
                empty_indices.add(idx)
            else:
                sanitized.append(t.strip())

        all_embeddings = []

        # Batch iteration
        for i in range(0, len(sanitized), batch_size):
            batch_texts = sanitized[i:i + batch_size]
            
            # Replace empty strings with single period for valid tokenization
            clean_batch = [t if t != "" else "." for t in batch_texts]

            encoded_inputs = self.tokenizer(
                clean_batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            input_ids = encoded_inputs["input_ids"].to(self.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = self.mean_pooling(model_output, attention_mask)

                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)

                batch_embs = pooled.cpu().numpy()
                all_embeddings.append(batch_embs)

        final_embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Zero out empty string positions
        if empty_indices:
            for idx in empty_indices:
                final_embeddings[idx] = np.zeros(self.embedding_dim, dtype=np.float32)

        return final_embeddings[0] if is_single else final_embeddings

