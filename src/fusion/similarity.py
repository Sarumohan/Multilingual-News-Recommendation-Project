"""
Multimodal Similarity and Modality Comparison Module.
Computes cosine similarity across fused representations and compares
text-only, image-only, and combined multimodal rankings.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class MultimodalSimilarityEngine:
    """Calculates cosine similarity on multimodal representations and benchmarks modality contributions."""

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two 1D vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    @staticmethod
    def compute_similarity_matrix(
        embeddings_a: np.ndarray,
        embeddings_b: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Pairwise cosine similarity matrix between rows of embeddings_a and embeddings_b.
        """
        if embeddings_b is None:
            embeddings_b = embeddings_a

        norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        norm_a = np.where(norm_a == 0, 1e-9, norm_a)
        norm_b = np.where(norm_b == 0, 1e-9, norm_b)

        a_hat = embeddings_a / norm_a
        b_hat = embeddings_b / norm_b

        return np.dot(a_hat, b_hat.T)

    @staticmethod
    def compare_article_pair_modalities(
        aid_a: str,
        aid_b: str,
        text_emb_a: np.ndarray,
        text_emb_b: np.ndarray,
        img_emb_a: Optional[np.ndarray],
        img_emb_b: Optional[np.ndarray],
        has_img_a: bool,
        has_img_b: bool,
        fused_emb_a: np.ndarray,
        fused_emb_b: np.ndarray,
        alpha: float = 0.7,
        beta: float = 0.3
    ) -> Dict[str, Any]:
        """
        Compare text-only, image-only, and combined multimodal similarity for a pair of articles.
        """
        # Text similarity
        s_text = MultimodalSimilarityEngine.compute_similarity(text_emb_a, text_emb_b)

        # Image similarity (only valid if both images exist)
        if has_img_a and has_img_b and img_emb_a is not None and img_emb_b is not None:
            s_image = MultimodalSimilarityEngine.compute_similarity(img_emb_a, img_emb_b)
            img_status = "both_available"
        else:
            s_image = None
            img_status = "missing_one_or_both"

        # Fused similarity
        s_fused = MultimodalSimilarityEngine.compute_similarity(fused_emb_a, fused_emb_b)

        return {
            "pair": f"{aid_a} <-> {aid_b}",
            "article_a": aid_a,
            "article_b": aid_b,
            "text_similarity": round(s_text, 4),
            "image_similarity": round(s_image, 4) if s_image is not None else None,
            "fused_similarity": round(s_fused, 4),
            "image_status": img_status,
            "alpha": alpha,
            "beta": beta
        }

