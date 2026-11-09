"""
Multimodal Feature Fusion Module.
Implements weighted feature fusion combining multilingual text embeddings (MiniLM, IndicBERT)
and visual representations (ResNet-50) with missing-modality resilience.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

from config.config import DEFAULT_FUSION_WEIGHTS, TEXT_EMBEDDING_DIM
from src.utils.helpers import setup_logger

logger = setup_logger("MultimodalFusion")

class MultimodalFusionEngine:
    """
    Combines textual and visual embeddings using weighted fusion strategies:
    1. Weighted Normalized Concatenation: Fused = [sqrt(alpha)*text || sqrt(beta)*image]
       Preserves full 2048-d visual feature fidelity while ensuring exact convex similarity.
    2. Projected Common Dimension Fusion: Projects 2048-d visual features to text space
       using a deterministic orthogonal projection matrix.
    Handles missing modalities by automatically falling back to normalized text-only embeddings.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_FUSION_WEIGHTS.get("text", 0.7),
        beta: float = DEFAULT_FUSION_WEIGHTS.get("image", 0.3),
        seed: int = 42
    ):
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(f"Weights alpha ({alpha}) and beta ({beta}) must be non-negative")
        
        # Normalize weights so alpha + beta = 1.0
        total = alpha + beta
        if total == 0:
            alpha, beta = 0.7, 0.3
            total = 1.0
        self.alpha = float(alpha / total)
        self.beta = float(beta / total)
        self.seed = seed

        # Deterministic projection matrix cache
        self._projection_matrices: Dict[Tuple[int, int], np.ndarray] = {}

    def get_orthogonal_projection_matrix(self, in_dim: int, out_dim: int) -> np.ndarray:
        """
        Generate a deterministic semi-orthogonal projection matrix using QR decomposition.
        Maps in_dim (e.g. 2048) to out_dim (e.g. 384 or 768) deterministically.
        """
        key = (in_dim, out_dim)
        if key in self._projection_matrices:
            return self._projection_matrices[key]

        rng = np.random.RandomState(self.seed)
        gaussian_matrix = rng.randn(in_dim, out_dim)
        # QR decomposition to obtain orthonormal basis columns
        q, _ = np.linalg.qr(gaussian_matrix)
        proj_matrix = q[:, :out_dim].astype(np.float32)

        self._projection_matrices[key] = proj_matrix
        return proj_matrix

    def fuse_article_vectors(
        self,
        text_vec: np.ndarray,
        image_vec: Optional[np.ndarray],
        has_image: bool = True,
        strategy: str = "weighted_concat"
    ) -> Tuple[np.ndarray, str]:
        """
        Fuse a single article's text vector and image vector.
        Returns: (fused_vector, fusion_mode: 'multimodal' or 'text_only')
        """
        # Ensure text vector is unit-norm
        t_norm = np.linalg.norm(text_vec)
        t_hat = text_vec / (t_norm + 1e-9)

        # Missing modality handling: fallback to text-only
        if not has_image or image_vec is None or not np.any(image_vec):
            if strategy == "weighted_concat":
                # Zero out image dimension while keeping text vector unit norm
                v_zeros = np.zeros(2048, dtype=np.float32)
                fused = np.concatenate([t_hat, v_zeros], axis=0)
                return fused, "text_only"
            elif strategy == "projected":
                return t_hat.copy(), "text_only"
            else:
                raise ValueError(f"Unknown fusion strategy: {strategy}")

        # Ensure image vector is unit-norm
        v_norm = np.linalg.norm(image_vec)
        v_hat = image_vec / (v_norm + 1e-9)

        if strategy == "weighted_concat":
            # Exact convex combination on unit hypersphere:
            # || [sqrt(alpha)*t || sqrt(beta)*v] ||^2 = alpha + beta = 1.0
            fused = np.concatenate([
                np.sqrt(self.alpha) * t_hat,
                np.sqrt(self.beta) * v_hat
            ], axis=0).astype(np.float32)
            return fused, "multimodal"

        elif strategy == "projected":
            # Project image vector to text vector dimension
            d_text = t_hat.shape[0]
            d_img = v_hat.shape[0]
            proj_matrix = self.get_orthogonal_projection_matrix(d_img, d_text)
            
            v_proj = np.dot(v_hat, proj_matrix)
            v_proj_hat = v_proj / (np.linalg.norm(v_proj) + 1e-9)

            fused_unnorm = (self.alpha * t_hat) + (self.beta * v_proj_hat)
            fused = fused_unnorm / (np.linalg.norm(fused_unnorm) + 1e-9)
            return fused.astype(np.float32), "multimodal"

        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")

    def fuse_datasets(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray,
        has_image_mask: np.ndarray,
        article_ids: List[str],
        strategy: str = "weighted_concat"
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Fuse complete dataset matrices across articles.
        Returns: (fused_embeddings_matrix, fusion_modes, statistics)
        """
        N = len(article_ids)
        if len(text_embeddings) != N or len(image_embeddings) != N:
            raise ValueError("Row count mismatch between text, image embeddings, and article IDs")

        fused_list = []
        modes = []

        for i in range(N):
            t_vec = text_embeddings[i]
            v_vec = image_embeddings[i]
            has_img = bool(has_image_mask[i])

            fused_vec, mode = self.fuse_article_vectors(
                t_vec,
                v_vec,
                has_image=has_img,
                strategy=strategy
            )
            fused_list.append(fused_vec)
            modes.append(mode)

        fused_matrix = np.vstack(fused_list).astype(np.float32)
        
        multimodal_count = sum(1 for m in modes if m == "multimodal")
        text_only_count = N - multimodal_count

        stats = {
            "total_articles": N,
            "multimodal_count": multimodal_count,
            "text_only_count": text_only_count,
            "fusion_strategy": strategy,
            "alpha": self.alpha,
            "beta": self.beta,
            "fused_dimension": int(fused_matrix.shape[1])
        }

        return fused_matrix, modes, stats
