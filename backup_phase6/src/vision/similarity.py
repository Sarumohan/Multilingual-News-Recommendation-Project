"""
Visual Embedding Similarity Module.
Computes cosine similarity, pairwise visual similarity matrices, and nearest-neighbor visual search.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class ImageSimilarityEngine:
    """Computes cosine similarity and visual rankings on dense image representations."""

    @staticmethod
    def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two 1D image feature vectors."""
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
        Compute pairwise cosine similarity matrix between visual embeddings.
        Normalizes rows to unit L2 norm.
        """
        if embeddings_b is None:
            embeddings_b = embeddings_a

        norm_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

        norm_a = np.where(norm_a == 0, 1e-9, norm_a)
        norm_b = np.where(norm_b == 0, 1e-9, norm_b)

        a_norm = embeddings_a / norm_a
        b_norm = embeddings_b / norm_b

        return np.dot(a_norm, b_norm.T)

    @staticmethod
    def find_top_k(
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_ids: List[str],
        top_k: int = 5,
        exclude_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Find top-K visually closest candidates to query_embedding.
        Returns list of (candidate_id, similarity_score).
        """
        sims = ImageSimilarityEngine.compute_similarity_matrix(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        ranked_indices = np.argsort(sims)[::-1]

        results = []
        for idx in ranked_indices:
            cid = candidate_ids[idx]
            if exclude_id and cid == exclude_id:
                continue
            results.append((cid, float(sims[idx])))
            if len(results) >= top_k:
                break

        return results

