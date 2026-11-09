"""
Diversity-Aware Ranking (MMR) and Popularity Bias Reduction Module (Phase 6 - Tasks 3 & 4).
Implements Maximal Marginal Relevance (MMR) trade-off, popularity dampening,
and exposure/diversity metrics (Shannon Entropy, Gini Index).
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from config.config import DEFAULT_MMR_LAMBDA, DEFAULT_POPULARITY_DEBIAS_GAMMA
from src.fusion.similarity import MultimodalSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("DiversityEngine")

class DiversityEngine:
    """
    Provides diversity re-ranking and debiasing operations:
    1. Maximal Marginal Relevance (MMR): lambda*Relevance - (1-lambda)*Similarity
    2. Popularity-Aware Ranking Adjustment
    3. Exposure & Diversity Analysis Metrics
    """

    def __init__(
        self,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        popularity_gamma: float = DEFAULT_POPULARITY_DEBIAS_GAMMA
    ):
        if not (0.0 <= mmr_lambda <= 1.0):
            raise ValueError(f"MMR lambda must be in [0, 1], got {mmr_lambda}")
        if popularity_gamma < 0:
            raise ValueError(f"Popularity gamma must be non-negative, got {popularity_gamma}")

        self.mmr_lambda = mmr_lambda
        self.popularity_gamma = popularity_gamma

    def apply_popularity_debiasing(
        self,
        score: float,
        popularity_count: int,
        gamma: Optional[float] = None
    ) -> float:
        """
        Dampen score based on article popularity to prevent runaway popularity bias:
        Score_debiased = Score / (1 + log2(1 + popularity))^gamma
        """
        g = gamma if gamma is not None else self.popularity_gamma
        if g <= 0.0 or popularity_count <= 0:
            return float(score)

        dampener = float((1.0 + np.log2(1.0 + popularity_count)) ** g)
        return float(score / dampener)

    def rank_with_mmr(
        self,
        candidates: List[Dict[str, Any]],
        embeddings_map: Dict[str, np.ndarray],
        top_k: int = 5,
        mmr_lambda: Optional[float] = None,
        score_key: str = "hybrid_score"
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidates using Maximal Marginal Relevance (MMR).
        MMR = argmax_{c in C \\ S} [ lambda * Rel(c) - (1 - lambda) * max_{s in S} Sim(c, s) ]
        """
        lam = mmr_lambda if mmr_lambda is not None else self.mmr_lambda

        if not candidates or top_k <= 0:
            return []

        # Sort input candidates by score to record original ranking
        sorted_candidates = sorted(candidates, key=lambda x: x.get(score_key, 0.0), reverse=True)
        for rank, c in enumerate(sorted_candidates, 1):
            c["original_rank"] = rank

        # Normalize relevance scores into [0, 1] for balanced MMR trade-off
        scores = [c.get(score_key, 0.0) for c in sorted_candidates]
        max_score = max(scores) if scores else 1.0
        rel_scores = {
            c["article_id"]: (c.get(score_key, 0.0) / max_score) if max_score > 0 else 0.0
            for c in sorted_candidates
        }

        candidate_pool = list(sorted_candidates)
        selected_slate: List[Dict[str, Any]] = []

        while len(selected_slate) < top_k and candidate_pool:
            if not selected_slate:
                # First item: select highest relevance candidate
                best_cand = candidate_pool.pop(0)
                best_cand["mmr_rank"] = len(selected_slate) + 1
                best_cand["mmr_score"] = round(float(rel_scores[best_cand["article_id"]]), 4)
                selected_slate.append(best_cand)
                continue

            best_mmr_score = -float("inf")
            best_idx = -1

            for idx, cand in enumerate(candidate_pool):
                aid = cand["article_id"]
                cand_vec = embeddings_map.get(aid)
                rel = rel_scores[aid]

                # Compute maximum similarity to any currently selected article
                max_sim = 0.0
                for sel in selected_slate:
                    sel_id = sel["article_id"]
                    sel_vec = embeddings_map.get(sel_id)

                    sim = 0.0
                    if cand_vec is not None and sel_vec is not None and cand_vec.shape == sel_vec.shape:
                        sim = max(0.0, float(MultimodalSimilarityEngine.compute_similarity(cand_vec, sel_vec)))
                    else:
                        # Category-level fallback similarity if vector not available
                        if cand.get("category") == sel.get("category"):
                            sim = 0.50

                    if sim > max_sim:
                        max_sim = sim

                # MMR Formulation: lambda * Rel - (1 - lambda) * max_sim
                mmr_val = (lam * rel) - ((1.0 - lam) * max_sim)

                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = idx

            if best_idx >= 0:
                best_cand = candidate_pool.pop(best_idx)
                best_cand["mmr_rank"] = len(selected_slate) + 1
                best_cand["mmr_score"] = round(float(best_mmr_score), 4)
                selected_slate.append(best_cand)
            else:
                break

        return selected_slate

    @staticmethod
    def calculate_category_entropy(articles: List[Dict[str, Any]]) -> float:
        """
        Compute Shannon Entropy over categories of recommended articles.
        Higher value indicates greater diversity across categories.
        """
        if not articles:
            return 0.0

        cats = [a.get("category", "unknown") for a in articles]
        _, counts = np.unique(cats, return_counts=True)
        probs = counts / len(cats)
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
        return max(0.0, entropy)

    @staticmethod
    def calculate_average_pairwise_similarity(
        articles: List[Dict[str, Any]],
        embeddings_map: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculates average cosine similarity between all pairs in the recommended slate.
        Lower value indicates greater intra-list divergence/diversity.
        """
        n = len(articles)
        if n <= 1:
            return 0.0

        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                v_i = embeddings_map.get(articles[i]["article_id"])
                v_j = embeddings_map.get(articles[j]["article_id"])
                if v_i is not None and v_j is not None and v_i.shape == v_j.shape:
                    sim = max(0.0, float(MultimodalSimilarityEngine.compute_similarity(v_i, v_j)))
                    sims.append(sim)

        return float(np.mean(sims)) if sims else 0.0

    def compare_rankings(
        self,
        original_slate: List[Dict[str, Any]],
        mmr_slate: List[Dict[str, Any]],
        embeddings_map: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compare diversity and redundancy metrics before and after MMR ranking.
        """
        orig_entropy = self.calculate_category_entropy(original_slate)
        mmr_entropy = self.calculate_category_entropy(mmr_slate)

        orig_pair_sim = self.calculate_average_pairwise_similarity(original_slate, embeddings_map)
        mmr_pair_sim = self.calculate_average_pairwise_similarity(mmr_slate, embeddings_map)

        orig_cats = len(set(a.get("category") for a in original_slate))
        mmr_cats = len(set(a.get("category") for a in mmr_slate))

        return {
            "original_category_entropy": round(orig_entropy, 4),
            "mmr_category_entropy": round(mmr_entropy, 4),
            "original_pairwise_similarity": round(orig_pair_sim, 4),
            "mmr_pairwise_similarity": round(mmr_pair_sim, 4),
            "original_unique_categories": orig_cats,
            "mmr_unique_categories": mmr_cats,
            "diversity_gain_pct": round(((mmr_entropy - orig_entropy) / (orig_entropy + 1e-9)) * 100.0, 2),
            "similarity_reduction_pct": round(((orig_pair_sim - mmr_pair_sim) / (orig_pair_sim + 1e-9)) * 100.0, 2)
        }
