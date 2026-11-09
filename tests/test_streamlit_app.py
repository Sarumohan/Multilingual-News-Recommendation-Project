"""
Integration Tests for Streamlit Dashboard Components and Logic (Phase 8).
Validates that:
1. Database and Hybrid Recommendation Engine load cleanly into the app.
2. Recommendation generation for all users executes with top-K and MMR options.
3. User interaction logging records events correctly into SQLite.
4. Multilingual and cross-lingual recommendations generate valid outputs.
5. XAI attribution signals generate non-empty payloads.
6. Chart components generate valid Plotly figures without exceptions.
"""

import unittest
from database.db_manager import DatabaseManager
from src.recommendation.hybrid_engine import HybridRecommendationEngine
from app.components.charts import (
    plot_language_distribution,
    plot_category_distribution,
    plot_baseline_comparison,
    plot_diversity_gain
)

class TestStreamlitAppIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = DatabaseManager()
        cls.engine = HybridRecommendationEngine(cls.db)

    def test_01_engine_and_db_ready(self):
        """1. Verify DB connection and engine readiness for Streamlit."""
        articles = self.db.get_all_articles()
        self.assertEqual(len(articles), 16)
        users = [u["user_id"] for u in self.db.execute_query("SELECT user_id FROM users")]
        self.assertEqual(len(users), 5)

    def test_02_personalized_recommendation_generation(self):
        """2. Verify personalized recommendations generate cleanly for dashboard."""
        recs = self.engine.recommend_for_user("usr_001", top_k=4, use_mmr=True)
        self.assertEqual(len(recs), 4)
        for r in recs:
            self.assertIn("title", r)
            self.assertIn("hybrid_score", r)
            self.assertIn("explanation", r)
            self.assertIn("explanation_details", r)

    def test_03_user_interaction_logging(self):
        """3. Verify interaction logging records events and refits CF, with cleanup."""
        test_uid = "usr_001"
        test_aid = "art_hi_003"
        initial_count = len(self.db.get_user_interactions(test_uid))

        try:
            self.db.record_user_interaction(test_uid, test_aid, "like", 2.0)
            new_count = len(self.db.get_user_interactions(test_uid))
            self.assertEqual(new_count, initial_count + 1)

            # Refit CF engine
            self.engine.cf_engine.fit()
            self.assertTrue(self.engine.cf_engine._is_fitted)
        finally:
            self.db.execute_update(
                "DELETE FROM user_interactions WHERE user_id = ? AND article_id = ? AND interaction_type = 'like' AND rating = 2.0",
                (test_uid, test_aid)
            )
            self.engine.cf_engine.fit()

    def test_04_cross_lingual_recommendation_flow(self):
        """4. Verify cross-lingual discovery logic generates valid recommendations."""
        cross_recs = self.engine.recommend_cross_lingual("art_en_001", target_languages=["hi", "as"], top_k=2)
        self.assertEqual(len(cross_recs), 2)
        for r in cross_recs:
            self.assertIn(r["target_language"], ["hi", "as"])
            self.assertGreater(r["semantic_similarity"], 0.0)

    def test_05_xai_attribution_details(self):
        """5. Verify XAI explanation details contain all 4 attribution signals."""
        recs = self.engine.recommend_for_user("usr_001", top_k=2)
        expl_dict = recs[0].get("explanation_details", {})
        self.assertIn("primary_factor", expl_dict)
        self.assertIn("signals", expl_dict)
        signals = expl_dict["signals"]
        self.assertIn("semantic_content_similarity", signals)
        self.assertIn("collaborative_community_signal", signals)
        self.assertIn("user_category_affinity", signals)
        self.assertIn("article_freshness", signals)

    def test_06_chart_generation(self):
        """6. Verify Plotly charts generate valid figure instances."""
        articles = self.db.get_all_articles()
        fig_lang = plot_language_distribution(articles)
        self.assertIsNotNone(fig_lang)

        fig_cat = plot_category_distribution(articles)
        self.assertIsNotNone(fig_cat)

        baselines = [
            {"config_name": "Popularity", "ndcg@5": 0.53, "mrr@5": 0.45},
            {"config_name": "Proposed", "ndcg@5": 0.91, "mrr@5": 0.88}
        ]
        fig_base = plot_baseline_comparison(baselines)
        self.assertIsNotNone(fig_base)

        fig_div = plot_diversity_gain()
        self.assertIsNotNone(fig_div)

    def test_07_card_image_dimensions_and_placeholders(self):
        """7. Verify all card images return exact (400, 240) dimensions, including fallbacks."""
        from app.components.cards import get_card_image
        articles = self.db.get_all_articles()
        for art in articles:
            img = get_card_image(art["article_id"], art["category"])
            self.assertEqual(img.size, (400, 240))
        # Test unknown / missing article ID fallback
        img_missing = get_card_image("art_missing_999", "science")
        self.assertEqual(img_missing.size, (400, 240))

    def test_08_multilingual_news_explorer_filters(self):
        """8. Verify News Explorer filtering logic across English, Hindi, and Assamese."""
        articles = self.db.get_all_articles()
        
        # Language filters
        en_arts = [a for a in articles if a["language"] == "en"]
        hi_arts = [a for a in articles if a["language"] == "hi"]
        as_arts = [a for a in articles if a["language"] == "as"]
        self.assertEqual(len(en_arts), 6)
        self.assertEqual(len(hi_arts), 5)
        self.assertEqual(len(as_arts), 5)

        # Category filter
        tech_arts = [a for a in articles if a["category"].lower() == "technology"]
        self.assertGreater(len(tech_arts), 0)

        # Keyword search
        search_query = "transformer"
        matched = [a for a in articles if search_query in a["title"].lower() or search_query in a.get("abstract", "").lower()]
        self.assertTrue(any(a["article_id"] == "art_en_001" for a in matched))


if __name__ == "__main__":
    unittest.main()

