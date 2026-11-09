"""
Explainable AI (XAI) Module for News Recommendations (Phase 6 - Task 2).
Generates transparent, multi-signal, human-interpretable justifications based on genuine
algorithmic scores (semantic content, collaborative consensus, category affinity, recency, freshness).
"""

from typing import Dict, List, Optional, Any
from config.config import SUPPORTED_LANGUAGES
from src.utils.helpers import setup_logger

logger = setup_logger("RecommendationExplainer")

class RecommendationExplainer:
    """
    Produces transparent, mathematically grounded justifications for recommended news articles.
    """

    @staticmethod
    def generate_detailed_explanation(
        article_meta: Dict[str, Any],
        content_score: float,
        cf_score: float,
        freshness_score: float = 0.0,
        user_cat_affinity: float = 0.0,
        seed_article_title: Optional[str] = None,
        is_cross_lingual: bool = False,
        source_lang: Optional[str] = None,
        is_cold_start: bool = False,
        is_popular_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a multi-signal explanation payload and human-readable text justification.
        """
        category = article_meta.get("category", "General").title()
        lang_code = article_meta.get("language", "en")
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
        title = article_meta.get("title", "Article")

        signals: Dict[str, float] = {
            "semantic_content_similarity": round(float(content_score), 4),
            "collaborative_community_signal": round(float(cf_score), 4),
            "user_category_affinity": round(float(user_cat_affinity), 4),
            "article_freshness": round(float(freshness_score), 4)
        }

        # 1. Cold-start cases
        if is_popular_fallback:
            return {
                "summary": f"Trending story in {category} popular among readers (General reader recommendation in {lang_name}).",
                "primary_factor": "popularity_fallback",
                "signals": signals
            }

        if is_cold_start:
            return {
                "summary": f"Matches your declared interest in {category} news (Available in {lang_name}).",
                "primary_factor": "declared_category_preference",
                "signals": signals
            }

        # 2. Cross-lingual cases
        if is_cross_lingual and source_lang:
            src_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang.upper())
            return {
                "summary": f"Cross-lingual recommendation in {lang_name} matching themes you read in {src_name}.",
                "primary_factor": "cross_lingual_transfer",
                "signals": signals
            }

        # 3. Item-to-item seed similarity
        if seed_article_title:
            return {
                "summary": f"Semantically similar to '{seed_article_title[:45]}...' which you recently engaged with.",
                "primary_factor": "seed_article_similarity",
                "signals": signals
            }

        # 4. Multi-signal personalized hybrid ranking
        # Identify dominant signal
        if cf_score >= 0.40 and cf_score >= content_score:
            summary = (
                f"Recommended because readers with reading patterns similar to yours "
                f"also engaged with this {category} story (Collaborative signal: {cf_score:.2f})."
            )
            primary_factor = "collaborative_consensus"

        elif freshness_score >= 0.85 and content_score >= 0.35:
            summary = (
                f"Recent breaking news in {category} matching your topical interests "
                f"(Freshness: {freshness_score:.2f}, Relevance: {content_score:.2f})."
            )
            primary_factor = "recent_breaking_freshness"

        elif user_cat_affinity >= 0.30 and content_score >= 0.35:
            summary = (
                f"Matches your high reading affinity for {category} news "
                f"(Topic relevance: {content_score:.2f}, Category interest: {user_cat_affinity:.2f})."
            )
            primary_factor = "user_category_affinity"

        elif content_score >= 0.50:
            summary = (
                f"Highly aligned with the semantic themes and entities in your reading history "
                f"(Semantic score: {content_score:.2f}, {lang_name})."
            )
            primary_factor = "semantic_content_similarity"

        else:
            summary = f"Relevant {category} news matching your multilingual reading profile ({lang_name})."
            primary_factor = "general_profile_relevance"

        return {
            "summary": summary,
            "primary_factor": primary_factor,
            "signals": signals
        }

