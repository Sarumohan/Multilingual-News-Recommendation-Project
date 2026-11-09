"""
Hybrid and Cross-Lingual News Recommendation Engine (Phase 5 - Tasks 4, 6, 7, 8).
Combines Content-Based semantic matching, Memory-Based Collaborative Filtering,
Cross-Lingual discovery, Cold-Start resilience, Diversity re-ranking, and Explainable AI justifications.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from config.config import (
    CONTENT_WEIGHT,
    COLLABORATIVE_WEIGHT,
    DEFAULT_TOP_K,
    SUPPORTED_LANGUAGES,
    DEFAULT_FRESHNESS_WEIGHT,
    DEFAULT_POPULARITY_DEBIAS_GAMMA,
    DEFAULT_MMR_LAMBDA
)
from database.db_manager import DatabaseManager
from src.recommendation.user_profile import UserProfileModeler, UserProfile
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.collaborative import CollaborativeFilteringEngine
from src.recommendation.temporal import TemporalModeler
from src.recommendation.diversity import DiversityEngine
from src.explainability.explainer import RecommendationExplainer
from src.utils.helpers import setup_logger

logger = setup_logger("HybridRecommendationEngine")

class HybridRecommendationEngine:
    """
    Main Recommendation Orchestrator combining:
    - Content-Based Filtering (MiniLM, IndicBERT, Multimodal ResNet-50)
    - Collaborative Filtering (Item-Item / User-User interaction matrix)
    - Temporal User-Interest & Article Freshness Modeling
    - Popularity Bias Reduction & Maximal Marginal Relevance (MMR)
    - Cross-Lingual Knowledge Transfer across English, Hindi, and Assamese
    - Cold-Start User and Item handling
    - Transparent Multi-Signal Explainable AI (XAI)
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        content_recommender: Optional[ContentBasedRecommender] = None,
        cf_engine: Optional[CollaborativeFilteringEngine] = None,
        profile_modeler: Optional[UserProfileModeler] = None,
        temporal_modeler: Optional[TemporalModeler] = None,
        diversity_engine: Optional[DiversityEngine] = None,
        content_weight: float = CONTENT_WEIGHT,
        collaborative_weight: float = COLLABORATIVE_WEIGHT,
        freshness_weight: float = DEFAULT_FRESHNESS_WEIGHT,
        popularity_gamma: float = DEFAULT_POPULARITY_DEBIAS_GAMMA,
        mmr_lambda: float = DEFAULT_MMR_LAMBDA,
        max_per_category: int = 2
    ):
        self.db = db_manager or DatabaseManager()
        self.content_recommender = content_recommender or ContentBasedRecommender(self.db)
        self.cf_engine = cf_engine or CollaborativeFilteringEngine(self.db)
        self.temporal_modeler = temporal_modeler or TemporalModeler()
        self.diversity_engine = diversity_engine or DiversityEngine(
            mmr_lambda=mmr_lambda,
            popularity_gamma=popularity_gamma
        )

        self.profile_modeler = profile_modeler or UserProfileModeler(
            db_manager=self.db,
            temporal_modeler=self.temporal_modeler
        )
        if self.profile_modeler.temporal_modeler is None:
            self.profile_modeler.temporal_modeler = self.temporal_modeler

        # Normalize weights
        tot = content_weight + collaborative_weight
        if tot <= 0:
            content_weight, collaborative_weight = 0.65, 0.35
            tot = 1.0
        self.alpha_content = content_weight / tot
        self.beta_cf = collaborative_weight / tot
        self.freshness_weight = freshness_weight
        self.max_per_category = max_per_category

        # Cache article popularity counts from interactions
        pop_rows = self.db.get_popular_articles(limit=100)
        self.popularity_counts = {r["article_id"]: int(r.get("interaction_count", 0)) for r in pop_rows}

    def generate_explanation(
        self,
        article_meta: Dict[str, Any],
        content_score: float,
        cf_score: float,
        hybrid_score: float,
        user_profile: Optional[UserProfile] = None,
        recommendation_type: str = "hybrid",
        freshness_score: float = 0.0
    ) -> str:
        """
        Generate human-interpretable, factual XAI explanation for a recommendation.
        """
        user_cat_aff = user_profile.category_affinity.get(article_meta.get("category", ""), 0.0) if user_profile else 0.0
        expl = RecommendationExplainer.generate_detailed_explanation(
            article_meta=article_meta,
            content_score=content_score,
            cf_score=cf_score,
            freshness_score=freshness_score,
            user_cat_affinity=user_cat_aff,
            is_cold_start=(recommendation_type == "cold_start_declared"),
            is_popular_fallback=(recommendation_type == "cold_start_popularity")
        )
        return expl["summary"]

        if recommendation_type == "cold_start_popularity":
            return (
                f"Popular and trending in {category} among readers "
                f"(Trending recommendation in {lang_name})."
            )

        if recommendation_type == "cold_start_declared":
            return (
                f"Recommended based on your declared interest in {category} "
                f"and preferred language ({lang_name})."
            )

        if recommendation_type == "cross_lingual":
            return (
                f"Cross-lingual recommendation in {lang_name} matching topics "
                f"from your reading history in another language."
            )

        # Standard personalized hybrid recommendations
        user_cat_affinity = user_profile.category_affinity.get(article_meta.get("category", ""), 0.0) if user_profile else 0.0

        if cf_score > 0.4 and cf_score >= content_score:
            return (
                f"Recommended because readers with similar engagement patterns "
                f"also read this {category} article (Collaborative signal)."
            )
        elif user_cat_affinity > 0.3:
            return (
                f"Matches your high reading affinity for {category} news "
                f"(Content relevance: {content_score:.2f})."
            )
        elif content_score > 0.5:
            return (
                f"Highly relevant to semantic themes in your recent reading history "
                f"(Available in {lang_name})."
            )
        else:
            return f"Relevant {category} news matching your multilingual reading profile."

    def apply_diversity_reranking(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        max_per_category: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks top candidate items to prevent category monopolization.
        Ensures a diverse mix of categories while prioritizing higher-scored items.
        """
        limit = max_per_category if max_per_category is not None else self.max_per_category
        selected: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        category_counts: Dict[str, int] = {}

        for cand in candidates:
            cat = cand.get("category", "general")
            current_count = category_counts.get(cat, 0)
            if current_count < limit:
                selected.append(cand)
                category_counts[cat] = current_count + 1
            else:
                deferred.append(cand)

            if len(selected) == top_k:
                break

        # If not enough items satisfy diversity threshold, fill remainder from deferred candidates
        if len(selected) < top_k and deferred:
            remainder = deferred[:(top_k - len(selected))]
            selected.extend(remainder)

        return selected

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = DEFAULT_TOP_K,
        preferred_languages: Optional[List[str]] = None,
        representation: str = "multimodal",
        apply_diversity: bool = True,
        use_mmr: bool = True,
        mmr_lambda: Optional[float] = None,
        apply_popularity_debiasing: bool = True,
        apply_freshness: bool = True,
        log_to_db: bool = False,
        user_profile: Optional[UserProfile] = None,
        exclude_article_ids: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate top-K personalized hybrid recommendations for a user.
        Incorporates:
        - Content Relevance
        - Collaborative Filtering
        - Temporal User Preferences
        - Article Freshness
        - Popularity Debiasing
        - Maximal Marginal Relevance (MMR) Diversity
        - Transparent XAI Explanations
        """
        # 1. Build User Interest Profile (with temporal decay applied to interactions)
        if user_profile is not None:
            profile = user_profile
        else:
            profile = self.profile_modeler.build_user_profile(
                user_id=user_id,
                text_embeddings_map=self.content_recommender.text_embeddings,
                multimodal_embeddings_map=self.content_recommender.multimodal_concat,
                article_metadata_map=self.content_recommender.article_metadata,
                exclude_article_ids=exclude_article_ids,
                apply_temporal_decay=True
            )

        # Language preference filter
        langs = preferred_languages or profile.preferred_languages or list(SUPPORTED_LANGUAGES.keys())

        # ---------------------------------------------------------
        # Case A: Cold-Start User with No Interaction History
        # ---------------------------------------------------------
        if not profile.has_history:
            return self._recommend_cold_start_user(
                user_id=user_id,
                profile=profile,
                preferred_languages=langs,
                top_k=top_k,
                apply_diversity=apply_diversity,
                log_to_db=log_to_db
            )

        # ---------------------------------------------------------
        # Case B: Warm User with Interaction History (Hybrid Pipeline)
        # ---------------------------------------------------------
        interacted_ids = profile.interacted_article_ids

        # 1. Content-Based Scores
        cb_scores = self.content_recommender.score_candidates_for_user_profile(
            profile=profile,
            candidate_ids=[aid for aid in self.content_recommender.article_ids if aid not in interacted_ids],
            representation=representation,
            target_languages=langs
        )

        # 2. Collaborative Filtering Scores
        cf_scores = self.cf_engine.get_collaborative_scores(user_id=user_id, method="item_item")

        # Determine effective fusion weights
        # If user has no CF signal at all, fallback to pure Content-Based (alpha=1.0, beta=0.0)
        has_cf_signal = any(cf_scores.get(aid, 0.0) > 0 for aid in cb_scores.keys())
        alpha = self.alpha_content if has_cf_signal else 1.0
        beta = self.beta_cf if has_cf_signal else 0.0

        candidates = []
        for aid, s_cb in cb_scores.items():
            meta = self.content_recommender.article_metadata.get(aid, {})
            s_cf = cf_scores.get(aid, 0.0)

            # 1. Base Convex Hybrid Scoring
            base_score = (alpha * s_cb) + (beta * s_cf)

            # 2. Article Publication Freshness Scoring
            s_fresh = 1.0
            if apply_freshness and self.temporal_modeler:
                s_fresh = self.temporal_modeler.compute_article_freshness_score(meta.get("published_at"))
                integrated_score = ((1.0 - self.freshness_weight) * base_score) + (self.freshness_weight * s_fresh)
            else:
                integrated_score = base_score

            # 3. Popularity Bias Debiasing
            pop_cnt = self.popularity_counts.get(aid, 0)
            if apply_popularity_debiasing and self.diversity_engine:
                final_score = self.diversity_engine.apply_popularity_debiasing(integrated_score, pop_cnt)
            else:
                final_score = integrated_score

            # 4. Multi-Signal Transparent XAI Explanation
            user_cat_aff = profile.category_affinity.get(meta.get("category", ""), 0.0)
            expl_dict = RecommendationExplainer.generate_detailed_explanation(
                article_meta=meta,
                content_score=s_cb,
                cf_score=s_cf,
                freshness_score=s_fresh,
                user_cat_affinity=user_cat_aff,
                is_cold_start=False,
                is_popular_fallback=False
            )

            candidates.append({
                "article_id": aid,
                "title": meta.get("title", ""),
                "language": meta.get("language", ""),
                "category": meta.get("category", ""),
                "hybrid_score": round(float(final_score), 4),
                "base_score": round(float(base_score), 4),
                "content_score": round(float(s_cb), 4),
                "collaborative_score": round(float(s_cf), 4),
                "freshness_score": round(float(s_fresh), 4),
                "popularity_count": pop_cnt,
                "has_image": self.content_recommender.has_image_map.get(aid, False),
                "explanation": expl_dict["summary"],
                "explanation_details": expl_dict,
                "recommendation_type": "hybrid"
            })

        # Sort by hybrid score descending as baseline
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Apply Diversity Re-ranking (MMR or category quota)
        if apply_diversity:
            if use_mmr and self.diversity_engine:
                emb_map = self.content_recommender.multimodal_concat if representation == "multimodal" else self.content_recommender.text_embeddings
                recommendations = self.diversity_engine.rank_with_mmr(
                    candidates=candidates,
                    embeddings_map=emb_map,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                    score_key="hybrid_score"
                )
            else:
                recommendations = self.apply_diversity_reranking(candidates, top_k=top_k)
        else:
            recommendations = candidates[:top_k]

        # Log recommendations to SQLite for auditing if enabled
        if log_to_db:
            for rec in recommendations:
                self.db.log_recommendation(
                    user_id=user_id,
                    article_id=rec["article_id"],
                    algorithm="Hybrid-CB+CF",
                    relevance_score=rec["hybrid_score"],
                    diversity_score=None,
                    explanation=rec["explanation"]
                )

        return recommendations

    def _recommend_cold_start_user(
        self,
        user_id: str,
        profile: UserProfile,
        preferred_languages: List[str],
        top_k: int = DEFAULT_TOP_K,
        apply_diversity: bool = True,
        log_to_db: bool = False
    ) -> List[Dict[str, Any]]:
        """Handles cold-start recommendations when user has zero history."""
        # Check if user declared category preferences
        has_declared_cats = len(profile.preferred_categories) > 0
        allowed_langs = set(preferred_languages)

        candidates = []

        if has_declared_cats:
            # Cold-Start Declared Interest Matching
            declared_set = set(profile.preferred_categories)
            for aid, meta in self.content_recommender.article_metadata.items():
                if meta.get("language") in allowed_langs:
                    cat = meta.get("category", "")
                    match_score = 1.0 if cat in declared_set else 0.2
                    expl = self.generate_explanation(
                        article_meta=meta,
                        content_score=match_score,
                        cf_score=0.0,
                        hybrid_score=match_score,
                        user_profile=profile,
                        recommendation_type="cold_start_declared"
                    )
                    candidates.append({
                        "article_id": aid,
                        "title": meta.get("title", ""),
                        "language": meta.get("language", ""),
                        "category": cat,
                        "hybrid_score": round(match_score, 4),
                        "content_score": round(match_score, 4),
                        "collaborative_score": 0.0,
                        "has_image": self.content_recommender.has_image_map.get(aid, False),
                        "explanation": expl,
                        "recommendation_type": "cold_start_declared"
                    })
            candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        else:
            # Completely Cold User: Popularity Fallback
            pop_articles = self.db.get_popular_articles(limit=20)
            for row in pop_articles:
                aid = row["article_id"]
                if row.get("language") in allowed_langs:
                    cnt = row.get("interaction_count", 0)
                    pop_score = round(min(1.0, cnt / 10.0), 4)
                    expl = self.generate_explanation(
                        article_meta=row,
                        content_score=0.0,
                        cf_score=0.0,
                        hybrid_score=pop_score,
                        user_profile=profile,
                        recommendation_type="cold_start_popularity"
                    )
                    candidates.append({
                        "article_id": aid,
                        "title": row.get("title", ""),
                        "language": row.get("language", ""),
                        "category": row.get("category", ""),
                        "hybrid_score": pop_score,
                        "content_score": 0.0,
                        "collaborative_score": 0.0,
                        "has_image": self.content_recommender.has_image_map.get(aid, False),
                        "explanation": expl,
                        "recommendation_type": "cold_start_popularity"
                    })

        if apply_diversity:
            recommendations = self.apply_diversity_reranking(candidates, top_k=top_k)
        else:
            recommendations = candidates[:top_k]

        if log_to_db:
            for rec in recommendations:
                self.db.log_recommendation(
                    user_id=user_id,
                    article_id=rec["article_id"],
                    algorithm=rec["recommendation_type"],
                    relevance_score=rec["hybrid_score"],
                    diversity_score=None,
                    explanation=rec["explanation"]
                )

        return recommendations

    def recommend_cross_lingual(
        self,
        seed_article_id: str,
        target_languages: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Cross-lingual recommendation with generated explanations.
        """
        recs = self.content_recommender.recommend_cross_lingual(
            seed_article_id=seed_article_id,
            target_languages=target_languages,
            top_k=top_k
        )
        for r in recs:
            meta = self.content_recommender.article_metadata.get(r["article_id"], {})
            r["explanation"] = self.generate_explanation(
                article_meta=meta,
                content_score=r["semantic_similarity"],
                cf_score=0.0,
                hybrid_score=r["semantic_similarity"],
                recommendation_type="cross_lingual"
            )
        return recs

    def recommend_similar_articles(
        self,
        seed_article_id: str,
        top_k: int = 5,
        target_languages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Item-to-Item content recommendations with explanations.
        """
        recs = self.content_recommender.recommend_similar_articles(
            seed_article_id=seed_article_id,
            top_k=top_k,
            target_languages=target_languages
        )
        for r in recs:
            seed_meta = self.content_recommender.article_metadata.get(seed_article_id, {})
            seed_title = seed_meta.get("title", "previous article")
            r["explanation"] = (
                f"Semantically similar to '{seed_title[:45]}...' "
                f"(Relevance: {r['similarity_score']:.2f})."
            )
        return recs
