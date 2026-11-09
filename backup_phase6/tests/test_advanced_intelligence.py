"""
Comprehensive Unit and Integration Tests for Phase 6 Advanced Intelligence.
Validates all 13 required test categories:
1. Exponential time decay.
2. Recent versus older interactions.
3. Long-term user interests.
4. Article freshness.
5. Explanation generation.
6. Explanation correctness.
7. MMR ranking.
8. Category diversity.
9. Source diversity.
10. Popularity-bias reduction.
11. Recommendation stability.
12. Missing timestamp handling.
13. Integration with Phase 5.
"""

import unittest
from datetime import datetime, timedelta
import numpy as np

from database.db_manager import DatabaseManager
from src.recommendation.hybrid_engine import HybridRecommendationEngine
from src.recommendation.temporal import TemporalModeler
from src.recommendation.diversity import DiversityEngine
from src.explainability.explainer import RecommendationExplainer
from src.evaluation.stability import StabilityEvaluator

class TestAdvancedIntelligencePhase6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager()
        cls.ref_time = datetime(2026, 11, 7, 12, 0, 0)
        cls.temporal_modeler = TemporalModeler(
            temporal_lambda=0.23105,  # 3.0 days half life
            freshness_lambda=0.13863, # 5.0 days half life
            reference_time=cls.ref_time
        )
        cls.diversity_engine = DiversityEngine(mmr_lambda=0.70, popularity_gamma=0.10)
        cls.engine = HybridRecommendationEngine(
            db_manager=cls.db,
            temporal_modeler=cls.temporal_modeler,
            diversity_engine=cls.diversity_engine
        )
        cls.stability_evaluator = StabilityEvaluator(cls.engine)

    def test_01_exponential_time_decay(self):
        """1. Verify exponential time decay formula: Weight = exp(-lambda * delta_days)."""
        lam = 0.23105
        # At delta = 0, decay must be exactly 1.0
        self.assertAlmostEqual(self.temporal_modeler.calculate_exponential_decay(0.0, lam), 1.0, places=5)
        # At delta = 3.0 days (half life), decay must be ~0.50
        decay_half = self.temporal_modeler.calculate_exponential_decay(3.0, lam)
        self.assertAlmostEqual(decay_half, 0.50, places=3)
        # At delta = 6.0 days, decay must be ~0.25
        decay_quarter = self.temporal_modeler.calculate_exponential_decay(6.0, lam)
        self.assertAlmostEqual(decay_quarter, 0.25, places=3)

    def test_02_recent_vs_older_interactions(self):
        """2. Verify that recent interactions receive strictly higher weight than older ones."""
        t_recent = "2026-11-07 06:00:00"  # 6 hours ago
        t_older = "2026-11-01 12:00:00"   # 6 days ago

        w_recent = self.temporal_modeler.compute_interaction_temporal_weight(t_recent)
        w_older = self.temporal_modeler.compute_interaction_temporal_weight(t_older)

        self.assertGreater(w_recent, w_older)
        self.assertGreater(w_recent, 0.90)

    def test_03_long_term_user_interests(self):
        """3. Verify long-term baseline floor is preserved even for very old interactions."""
        t_very_old = "2025-01-01 00:00:00"  # > 1 year ago
        w_old = self.temporal_modeler.compute_interaction_temporal_weight(t_very_old, long_term_ratio=0.30)
        # Should not drop below 0.30 baseline floor
        self.assertGreaterEqual(w_old, 0.30)
        self.assertLess(w_old, 0.35)

    def test_04_article_freshness(self):
        """4. Verify article freshness score decays gracefully with publication timestamp."""
        t_pub_new = "2026-11-07 11:00:00"
        t_pub_old = "2026-11-02 12:00:00"  # 5 days old (half life)

        score_new = self.temporal_modeler.compute_article_freshness_score(t_pub_new)
        score_old = self.temporal_modeler.compute_article_freshness_score(t_pub_old)

        self.assertGreater(score_new, score_old)
        self.assertAlmostEqual(score_old, 0.50, places=2)

    def test_05_explanation_generation(self):
        """5. Verify readable explanation generation for recommendations."""
        meta = {"title": "Quantum Computing AI", "category": "technology", "language": "en"}
        expl = RecommendationExplainer.generate_detailed_explanation(
            article_meta=meta,
            content_score=0.85,
            cf_score=0.20,
            freshness_score=0.90,
            user_cat_affinity=0.80
        )
        self.assertIn("summary", expl)
        self.assertIn("primary_factor", expl)
        self.assertIn("signals", expl)
        self.assertIsInstance(expl["summary"], str)
        self.assertGreater(len(expl["summary"]), 15)

    def test_06_explanation_correctness(self):
        """6. Verify explanation reflects dominant mathematical signal without fabrication."""
        meta = {"title": "Cricket Finals", "category": "sports", "language": "en"}
        # Case A: CF signal dominates
        expl_cf = RecommendationExplainer.generate_detailed_explanation(
            article_meta=meta,
            content_score=0.30,
            cf_score=0.85
        )
        self.assertEqual(expl_cf["primary_factor"], "collaborative_consensus")
        self.assertIn("Collaborative signal", expl_cf["summary"])

        # Case B: Popularity fallback
        expl_pop = RecommendationExplainer.generate_detailed_explanation(
            article_meta=meta,
            content_score=0.0,
            cf_score=0.0,
            is_popular_fallback=True
        )
        self.assertEqual(expl_pop["primary_factor"], "popularity_fallback")
        self.assertIn("Trending", expl_pop["summary"])

    def test_07_mmr_ranking(self):
        """7. Verify Maximal Marginal Relevance trade-off selects diverse items."""
        cands = [
            {"article_id": "c1", "hybrid_score": 0.95, "category": "tech"},
            {"article_id": "c2", "hybrid_score": 0.94, "category": "tech"},
            {"article_id": "c3", "hybrid_score": 0.85, "category": "business"},
            {"article_id": "c4", "hybrid_score": 0.80, "category": "science"}
        ]
        # Vectors where c1 and c2 are identical, c3 and c4 are orthogonal
        embs = {
            "c1": np.array([1.0, 0.0, 0.0]),
            "c2": np.array([1.0, 0.0, 0.0]),
            "c3": np.array([0.0, 1.0, 0.0]),
            "c4": np.array([0.0, 0.0, 1.0]),
        }
        mmr_slate = self.diversity_engine.rank_with_mmr(
            candidates=cands,
            embeddings_map=embs,
            top_k=3,
            mmr_lambda=0.70
        )
        self.assertEqual(len(mmr_slate), 3)
        # c1 should be chosen first
        self.assertEqual(mmr_slate[0]["article_id"], "c1")
        # Second item should be c3 (business) rather than redundant c2 (tech)
        self.assertEqual(mmr_slate[1]["article_id"], "c3")

    def test_08_category_diversity(self):
        """8. Verify category entropy calculation and diversity gain under MMR."""
        monopolized_slate = [
            {"category": "tech"}, {"category": "tech"}, {"category": "tech"}
        ]
        diverse_slate = [
            {"category": "tech"}, {"category": "business"}, {"category": "science"}
        ]
        entropy_mono = self.diversity_engine.calculate_category_entropy(monopolized_slate)
        entropy_div = self.diversity_engine.calculate_category_entropy(diverse_slate)

        self.assertEqual(entropy_mono, 0.0)
        self.assertGreater(entropy_div, 1.0)

    def test_09_source_diversity(self):
        """9. Verify pairwise redundancy calculation in recommendation slates."""
        embs = {
            "a1": np.array([1.0, 0.0]),
            "a2": np.array([1.0, 0.0]),
            "a3": np.array([0.0, 1.0])
        }
        slate_redundant = [{"article_id": "a1"}, {"article_id": "a2"}]
        slate_diverse = [{"article_id": "a1"}, {"article_id": "a3"}]

        sim_red = self.diversity_engine.calculate_average_pairwise_similarity(slate_redundant, embs)
        sim_div = self.diversity_engine.calculate_average_pairwise_similarity(slate_diverse, embs)

        self.assertAlmostEqual(sim_red, 1.0, places=5)
        self.assertAlmostEqual(sim_div, 0.0, places=5)

    def test_10_popularity_bias_reduction(self):
        """10. Verify popularity dampening reduces score of hyper-popular items."""
        score_unpopular = self.diversity_engine.apply_popularity_debiasing(score=1.0, popularity_count=0)
        score_popular = self.diversity_engine.apply_popularity_debiasing(score=1.0, popularity_count=50)

        self.assertEqual(score_unpopular, 1.0)
        self.assertLess(score_popular, 1.0)
        self.assertGreater(score_popular, 0.70)

    def test_11_recommendation_stability(self):
        """11. Verify stability under 10% weight jitter and modality fallback."""
        res_jitter = self.stability_evaluator.evaluate_weight_jitter_stability("usr_001", top_k=5)
        self.assertTrue(res_jitter["is_stable"])
        self.assertGreaterEqual(res_jitter["topk_overlap"], 0.70)

        res_mod = self.stability_evaluator.evaluate_modality_dropout_stability("usr_001", top_k=5)
        self.assertTrue(res_mod["is_stable"])

    def test_12_missing_timestamp_handling(self):
        """12. Verify safe handling of None, empty, or unparseable timestamps."""
        # None timestamp
        delta_none = self.temporal_modeler.calculate_time_delta_days(None)
        self.assertEqual(delta_none, 0.0)
        fresh_none = self.temporal_modeler.compute_article_freshness_score(None)
        self.assertEqual(fresh_none, 1.0)

        # Malformed string timestamp
        delta_bad = self.temporal_modeler.calculate_time_delta_days("invalid-date-string")
        self.assertEqual(delta_bad, 0.0)

    def test_13_integration_with_phase5(self):
        """13. Verify full pipeline produces recommendations integrating temporal, CF, CB, and MMR."""
        recs = self.engine.recommend_for_user("usr_001", top_k=4, use_mmr=True)
        self.assertEqual(len(recs), 4)
        for r in recs:
            self.assertIn("hybrid_score", r)
            self.assertIn("content_score", r)
            self.assertIn("collaborative_score", r)
            self.assertIn("freshness_score", r)
            self.assertIn("explanation", r)
            self.assertIn("explanation_details", r)
            self.assertTrue(0.0 <= r["hybrid_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

