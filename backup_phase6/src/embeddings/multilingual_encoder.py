"""
Multilingual Text Encoder Module.
Wraps pretrained transformer-based sentence embedding models (e.g., paraphrase-multilingual-MiniLM-L12-v2)
to extract dense, normalized semantic representations across English, Hindi, and Assamese.
"""

import sys
import os
import torch
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

from config.config import DEFAULT_TEXT_MODEL, TEXT_EMBEDDING_DIM
from src.utils.helpers import setup_logger

logger = setup_logger("MultilingualEncoder")

class MultilingualTextEncoder:
    """Encodes multilingual text into dense, normalized semantic embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 128
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        logger.info(f"Loading multilingual model '{self.model_name}' on device '{self.device}'...")
        if "HF_HUB_DISABLE_SYMLINKS_WARNING" not in os.environ:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        try:
            # First attempt loading from local cache to prevent unnecessary network retries in offline/sandboxed environments
            self.model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            self.model.max_seq_length = self.max_seq_length
            self.embedding_dim = (
                self.model.get_embedding_dimension()
                if hasattr(self.model, "get_embedding_dimension")
                else self.model.get_sentence_embedding_dimension()
            )
            logger.info(f"Model loaded from local cache. Embedding dimension: {self.embedding_dim}")
        except Exception as local_err:
            logger.info(f"Local cache miss or offline load error ({local_err}); attempting standard load...")
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.model.max_seq_length = self.max_seq_length
                self.embedding_dim = (
                self.model.get_embedding_dimension()
                if hasattr(self.model, "get_embedding_dimension")
                else self.model.get_sentence_embedding_dimension()
            )
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer '{self.model_name}': {e}")
                raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode single text or list of texts into dense NumPy embeddings.
        Handles empty/null strings safely and normalizes vectors to unit length.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        # Sanitize texts: replace empty or whitespace-only with a placeholder token
        sanitized = []
        empty_indices = set()
        for idx, t in enumerate(text_list):
            if t is None or not isinstance(t, str) or t.strip() == "":
                sanitized.append(" ")
                empty_indices.add(idx)
            else:
                sanitized.append(t.strip())

        # Encode using the SentenceTransformer pipeline
        embeddings = self.model.encode(
            sanitized,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )

        # Ensure correct shape and float32 dtype
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # If any input was empty, zero out its embedding vector
        if empty_indices:
            for idx in empty_indices:
                embeddings[idx] = np.zeros(self.embedding_dim, dtype=np.float32)

        return embeddings[0] if is_single else embeddings
