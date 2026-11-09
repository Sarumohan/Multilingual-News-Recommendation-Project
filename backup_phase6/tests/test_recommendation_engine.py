"""
Comprehensive Unit and Integration Tests for Recommendation Engine (Phase 5).
Validates all 12 key functional requirements:
1. Content-based recommendation.
2. User profile generation.
3. Collaborative filtering.
4. Hybrid scoring.
5. Cross-lingual candidate retrieval.
6. Language filtering.
7. Cold-start users.
8. New articles.
9. Missing modalities.
10. Recommendation explanations.
11. Recommendation ranking.
12. Duplicate recommendation prevention.
"""

import unittest
import numpy as np
from database.db_manager import DatabaseManager
from src.recommendation.user_profile import UserProfileModeler, UserProfile
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative import CollaborativeFilteringEngine
from src.recommendation.hybrid_engine import HybridRecommendationEngine

class TestRecommendationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager()
        cls.content_rec = ContentBasedRecommender(cls.db)
        cls.cf_engine = CollaborativeFilteringEngine(cls.db)
        cls.profile_modeler = UserProfileModeler(cls.db)
        cls.hybrid_engine = HybridRecommendationEngine(
            db_manager=cls.db,
            content_recommender=cls.content_rec,
            cf_engine=cls.cf_engine,
            profile_modeler=cls.profile_modeler,
            content_weight=0.65,
            collaborative_weight=0.35,
            max_per_category=2
        )

    def test_01_content_based_recommendation(self):
        """1. Test content-based recommendation produces valid scores and similar items."""
        recs = self.content_rec.recommend_similar_articles("art_en_001", top_k=3)
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 3)
        for r in recs:
            self.assertIn("similarity_score", r)
            self.assertTrue(0.0 <= r["similarity_score"] <= 1.0)
            self.assertNotEqual(r["article_id"], "art_en_001")

    def test_02_user_profile_generation(self):
        """2. Test user profile generation aggregates interactions, categories, and embeddings."""
        prof = self.profile_modeler.build_user_profile(
            user_id="usr_001",
            text_embeddings_map=self.content_rec.text_embeddings,
            multimodal_embeddings_map=self.content_rec.multimodal_concat,
            article_metadata_map=self.content_rec.article_metadata
        )
        self.assertEqual(prof.user_id, "usr_001")
        self.assertTrue(prof.has_history)
        self.assertGreater(prof.total_interactions, 0)
        self.assertIsNotNone(prof.text_vector)
        self.assertEqual(prof.text_vector.shape, (384,))
        # Unit norm verification
        self.assertAlmostEqual(float(np.linalg.norm(prof.text_vector)), 1.0, places=5)
        # Category affinities sum to ~1.0
        self.assertAlmostEqual(sum(prof.category_affinity.values()), 1.0, places=5)

    def test_03_collaborative_filtering(self):
        """3. Test collaborative filtering computes interaction matrix and predicts scores."""
        self.assertTrue(self.cf_engine.IS_SIMULATED_DATASET)
        sparsity = self.cf_engine.get_matrix_sparsity()
        self.assertTrue(0.0 <= sparsity <= 100.0)

        scores = self.cf_engine.get_collaborative_scores("usr_001", method="item_item")
        self.assertEqual(len(scores), len(self.cf_engine.article_ids))
        for aid, s in scores.items():
            self.assertTrue(0.0 <= s <= 1.0)

        cf_recs = self.cf_engine.recommend("usr_001", top_k=3)
        self.assertIsInstance(cf_recs, list)

    def test_04_hybrid_scoring(self):
        """4. Test hybrid scoring combines content-based and collaborative filtering convexly."""
        recs = self.hybrid_engine.recommend_for_user("usr_001", top_k=5)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn("hybrid_score", r)
            self.assertIn("content_score", r)
            self.assertIn("collaborative_score", r)
            self.assertTrue(0.0 <= r["hybrid_score"] <= 1.0)
            self.assertTrue(0.0 <= r["content_score"] <= 1.0)
            self.assertTrue(0.0 <= r["collaborative_score"] <= 1.0)

    def test_05_cross_lingual_candidate_retrieval(self):
        """5. Test cross-lingual candidate retrieval across English, Hindi, and Assamese."""
        # Test English seed -> Hindi and Assamese targets
        recs_en = self.hybrid_engine.recommend_cross_lingual(
            seed_article_id="art_en_001",
            target_languages=["hi", "as"],
            top_k=4
        )
        self.assertGreater(len(recs_en), 0)
        for r in recs_en:
            self.assertEqual(r["source_language"], "en")
            self.assertIn(r["target_language"], ["hi", "as"])
            self.assertNotEqual(r["target_language"], "en")
            self.assertTrue(0.0 <= r["semantic_similarity"] <= 1.0)

        # Test Indic-to-Indic (Assamese -> Hindi)
        recs_as = self.hybrid_engine.recommend_cross_lingual(
            seed_article_id="art_as_001",
            target_languages=["hi"],
            top_k=2
        )
        self.assertGreater(len(recs_as), 0)
        for r in recs_as:
            self.assertEqual(r["source_language"], "as")
            self.assertEqual(r["target_language"], "hi")

    def test_06_language_filtering(self):
        """6. Test language filtering restricts recommendations to specified languages."""
        recs_hi = self.hybrid_engine.recommend_for_user(
            "usr_001",
            preferred_languages=["hi"],
            top_k=5
        )
        for r in recs_hi:
            self.assertEqual(r["language"], "hi")

        recs_as = self.hybrid_engine.recommend_for_user(
            "usr_002",
            preferred_languages=["as"],
            top_k=5
        )
        for r in recs_as:
            self.assertEqual(r["language"], "as")

    def test_07_cold_start_users(self):
        """7. Test recommendations for cold-start users (declared interests vs pure popularity)."""
        # Case A: Declared preferences
        declared_profile = UserProfile(
            user_id="new_declared_user",
            preferred_languages=["hi"],
            preferred_categories=["science"],
            category_affinity={"science": 1.0},
            has_history=False
        )
        recs_dec = self.hybrid_engine._recommend_cold_start_user(
            user_id="new_declared_user",
            profile=declared_profile,
            preferred_languages=["hi"],
            top_k=3
        )
        self.assertGreater(len(recs_dec), 0)
        self.assertEqual(recs_dec[0]["recommendation_type"], "cold_start_declared")
        self.assertEqual(recs_dec[0]["category"], "science")

        # Case B: Completely blank user -> popularity fallback
        blank_profile = UserProfile(user_id="brand_new_user", has_history=False)
        recs_pop = self.hybrid_engine._recommend_cold_start_user(
            user_id="brand_new_user",
            profile=blank_profile,
            preferred_languages=["en", "hi", "as"],
            top_k=3
        )
        self.assertGreater(len(recs_pop), 0)
        self.assertEqual(recs_pop[0]["recommendation_type"], "cold_start_popularity")

    def test_08_new_articles_cold_item(self):
        """8. Test that a new article with 0 interactions can be recommended via content."""
        # Check an article with 0 interactions in user 1's collaborative matrix
        scores = self.cf_engine.get_collaborative_scores("usr_001")
        # art_en_006 has 0 collaborative score for usr_001
        self.assertEqual(scores.get("art_en_006", 0.0), 0.0)

        # But in hybrid/content it receives non-zero content score
        cb_scores = self.content_rec.score_candidates_for_user_profile(
            profile=self.profile_modeler.build_user_profile(
                user_id="usr_001",
                text_embeddings_map=self.content_rec.text_embeddings,
                article_metadata_map=self.content_rec.article_metadata
            ),
            candidate_ids=["art_en_006"]
        )
        self.assertIn("art_en_006", cb_scores)
        self.assertGreater(cb_scores["art_en_006"], 0.0)

    def test_09_missing_modalities_handling(self):
        """9. Test that articles with missing images are handled properly in recommendations."""
        # art_en_002 has missing image
        self.assertFalse(self.content_rec.has_image_map.get("art_en_002", True))
        recs = self.hybrid_engine.recommend_similar_articles("art_en_002", top_k=3)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertTrue(0.0 <= r["similarity_score"] <= 1.0)

    def test_10_recommendation_explanations(self):
        """10. Test that every recommendation generates a non-empty, factual explanation."""
        recs = self.hybrid_engine.recommend_for_user("usr_001", top_k=5)
        for r in recs:
            self.assertIn("explanation", r)
            self.assertIsInstance(r["explanation"], str)
            self.assertGreater(len(r["explanation"]), 10)
            # Ensure no fake placeholder text
            self.assertNotIn("TODO", r["explanation"])
            self.assertNotIn("fake", r["explanation"].lower())

    def test_11_recommendation_ranking(self):
        """11. Test that candidate recommendations are sorted in descending order of hybrid score."""
        recs = self.hybrid_engine.recommend_for_user("usr_001", top_k=6, apply_diversity=False)
        scores = [r["hybrid_score"] for r in recs]
        # Check monotonicity
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_12_duplicate_recommendation_prevention(self):
        """12. Test that unread recommendations contain no duplicate article IDs and exclude read items."""
        recs = self.hybrid_engine.recommend_for_user("usr_001", top_k=10)
        rec_ids = [r["article_id"] for r in recs]
        # No duplicates in recommendations list
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

        # Exclude articles already read/interacted with by usr_001
        prof = self.profile_modeler.build_user_profile("usr_001")
        interacted = prof.interacted_article_ids
        for aid in rec_ids:
            self.assertNotIn(aid, interacted)


if __name__ == "__main__":
    unittest.main()

