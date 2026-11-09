"""
Recommendation Stability Evaluation Module (Phase 6 - Task 5).
Measures recommendation ranking consistency under controlled perturbations:
- Interaction weight jitter (+/- 10%)
- Single interaction dropout
- Missing visual modality injection
Metrics: Top-K Jaccard Overlap and Spearman Rank Correlation.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from database.db_manager import DatabaseManager
from src.utils.helpers import setup_logger

logger = setup_logger("StabilityEvaluator")

class StabilityEvaluator:
    """Evaluates recommendation robustness and rank stability under controlled perturbations."""

    def __init__(self, engine: Any):
        self.engine = engine

    @staticmethod
    def calculate_topk_overlap(list_a: List[str], list_b: List[str]) -> float:
        """Calculate Jaccard overlap between two recommendation lists."""
        set_a = set(list_a)
        set_b = set(list_b)
        if not set_a and not set_b:
            return 1.0
        union = set_a.union(set_b)
        if not union:
            return 0.0
        return float(len(set_a.intersection(set_b)) / len(union))

    @staticmethod
    def calculate_spearman_rank_correlation(list_a: List[str], list_b: List[str]) -> float:
        """
        Calculate Spearman rank correlation on intersecting items between two ranked lists.
        Returns correlation in [-1.0, 1.0].
        """
        common = [item for item in list_a if item in list_b]
        n = len(common)
        if n <= 1:
            return 1.0 if list_a == list_b else 0.0

        from scipy.stats import spearmanr
        ranks_a = [list_a.index(x) for x in common]
        ranks_b = [list_b.index(x) for x in common]

        corr, _ = spearmanr(ranks_a, ranks_b)
        if np.isnan(corr):
            return 1.0 if ranks_a == ranks_b else 0.0
        return float(corr)

    def evaluate_weight_jitter_stability(
        self,
        user_id: str,
        top_k: int = 5,
        jitter_factor: float = 0.10
    ) -> Dict[str, Any]:
        """
        Evaluate ranking stability under +/- 10% interaction weight perturbation.
        """
        # Baseline recommendations
        base_recs = self.engine.recommend_for_user(user_id=user_id, top_k=top_k)
        base_ids = [r["article_id"] for r in base_recs]

        # Perturbed weights
        original_weights = dict(self.engine.profile_modeler.weights)
        perturbed_weights = {
            k: v * (1.0 + jitter_factor if i % 2 == 0 else 1.0 - jitter_factor)
            for i, (k, v) in enumerate(original_weights.items())
        }

        try:
            self.engine.profile_modeler.weights = perturbed_weights
            pert_recs = self.engine.recommend_for_user(user_id=user_id, top_k=top_k)
            pert_ids = [r["article_id"] for r in pert_recs]
        finally:
            self.engine.profile_modeler.weights = original_weights

        overlap = self.calculate_topk_overlap(base_ids, pert_ids)
        correlation = self.calculate_spearman_rank_correlation(base_ids, pert_ids)

        return {
            "test": "interaction_weight_jitter",
            "jitter_factor": jitter_factor,
            "baseline_ids": base_ids,
            "perturbed_ids": pert_ids,
            "topk_overlap": round(overlap, 4),
            "rank_correlation": round(correlation, 4),
            "is_stable": overlap >= 0.70
        }

    def evaluate_dropout_stability(
        self,
        user_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate ranking stability when a single historical interaction is omitted from the interest profile.
        Candidate set keeps the dropped item excluded to fairly evaluate unread recommendations stability.
        """
        # Baseline
        base_recs = self.engine.recommend_for_user(user_id=user_id, top_k=top_k)
        base_ids = [r["article_id"] for r in base_recs]

        # Get user history
        prof = self.engine.profile_modeler.build_user_profile(user_id=user_id)
        if len(prof.interacted_article_ids) < 2:
            return {"test": "interaction_dropout", "skipped": True, "reason": "Insufficient history"}

        # Drop one interaction from profile vector
        drop_aid = list(prof.interacted_article_ids)[-1]
        pert_profile = self.engine.profile_modeler.build_user_profile(
            user_id=user_id,
            text_embeddings_map=self.engine.content_recommender.text_embeddings,
            multimodal_embeddings_map=self.engine.content_recommender.multimodal_concat,
            article_metadata_map=self.engine.content_recommender.article_metadata,
            exclude_article_ids={drop_aid}
        )
        # Keep dropped item in interacted set for candidate masking
        pert_profile.interacted_article_ids.add(drop_aid)

        pert_recs = self.engine.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            user_profile=pert_profile
        )
        pert_ids = [r["article_id"] for r in pert_recs]

        overlap = self.calculate_topk_overlap(base_ids, pert_ids)
        correlation = self.calculate_spearman_rank_correlation(base_ids, pert_ids)

        return {
            "test": "interaction_dropout",
            "dropped_article_id": drop_aid,
            "baseline_ids": base_ids,
            "perturbed_ids": pert_ids,
            "topk_overlap": round(overlap, 4),
            "rank_correlation": round(correlation, 4),
            "is_stable": overlap >= 0.60
        }

    def evaluate_modality_dropout_stability(
        self,
        user_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate ranking stability when switching between multimodal and pure-text representations.
        Verifies that visual absence gracefully degrades without chaotic ranking inversions.
        """
        base_recs = self.engine.recommend_for_user(user_id=user_id, top_k=top_k, representation="multimodal")
        base_ids = [r["article_id"] for r in base_recs]

        text_recs = self.engine.recommend_for_user(user_id=user_id, top_k=top_k, representation="text_minilm")
        text_ids = [r["article_id"] for r in text_recs]

        overlap = self.calculate_topk_overlap(base_ids, text_ids)
        correlation = self.calculate_spearman_rank_correlation(base_ids, text_ids)

        return {
            "test": "modality_dropout_text_fallback",
            "baseline_ids": base_ids,
            "text_only_ids": text_ids,
            "topk_overlap": round(overlap, 4),
            "rank_correlation": round(correlation, 4),
            "is_stable": overlap >= 0.60
        }

    def evaluate_all(self, user_id: str = "usr_001", top_k: int = 5) -> Dict[str, Any]:
        """Run all stability tests for a user."""
        res_jitter = self.evaluate_weight_jitter_stability(user_id=user_id, top_k=top_k)
        res_dropout = self.evaluate_dropout_stability(user_id=user_id, top_k=top_k)
        res_modality = self.evaluate_modality_dropout_stability(user_id=user_id, top_k=top_k)

        all_stable = (
            res_jitter.get("is_stable", False) and
            res_dropout.get("is_stable", True) and
            res_modality.get("is_stable", True)
        )
        return {
            "user_id": user_id,
            "top_k": top_k,
            "weight_jitter_test": res_jitter,
            "interaction_dropout_test": res_dropout,
            "modality_dropout_test": res_modality,
            "overall_stability_pass": all_stable
        }
