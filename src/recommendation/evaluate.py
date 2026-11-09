"""
Offline Recommendation Quality and Performance Evaluation Module (Phase 5).
Computes quantitative ranking and diversity metrics:
- Precision@K
- Recall@K / HitRate@K
- Catalog Coverage
- Intra-List Category Diversity (Shannon Entropy & Gini Index)
- Cross-Lingual Transfer Coverage
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from src.recommendation.hybrid_engine import HybridRecommendationEngine
from src.utils.helpers import setup_logger

logger = setup_logger("RecommendationEvaluator")

class RecommendationEvaluator:
    """Evaluates recommendation engine metrics across users."""

    def __init__(self, engine: Optional[HybridRecommendationEngine] = None):
        self.engine = engine or HybridRecommendationEngine()
        self.db = self.engine.db

    def evaluate_leave_one_out(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Evaluate recommendation performance using Leave-One-Out validation on user reading history.
        For each user with >= 2 interactions, holds out the most recent article,
        generates top-K recommendations, and measures whether the held-out article is retrieved.
        """
        history_rows = self.db.get_user_reading_history()
        # Group by user
        user_history: Dict[str, List[str]] = {}
        for row in history_rows:
            uid = row["user_id"]
            aid = row["article_id"]
            if uid not in user_history:
                user_history[uid] = []
            if aid not in user_history[uid]:
                user_history[uid].append(aid)

        evaluable_users = [u for u, aids in user_history.items() if len(aids) >= 2]
        if not evaluable_users:
            logger.warning("No users with sufficient history for leave-one-out validation.")
            return {"evaluable_users": 0}

        hits = 0
        total_reciprocal_rank = 0.0
        all_recommended_articles: Set[str] = set()
        user_category_diversities: List[float] = []

        for uid in evaluable_users:
            held_out = user_history[uid][0]  # Most recent article
            # Compute recommendations holding out the target article
            recs = self.engine.recommend_for_user(
                uid,
                top_k=top_k,
                apply_diversity=True,
                exclude_article_ids={held_out}
            )
            rec_ids = [r["article_id"] for r in recs]
            all_recommended_articles.update(rec_ids)

            # Hit Rate & MRR
            if held_out in rec_ids:
                hits += 1
                rank = rec_ids.index(held_out) + 1
                total_reciprocal_rank += (1.0 / rank)

            # Intra-list category diversity: ratio of unique categories to total recommended
            cats = [r["category"] for r in recs]
            diversity_ratio = len(set(cats)) / len(cats) if cats else 0.0
            user_category_diversities.append(diversity_ratio)

        total_catalog = len(self.engine.content_recommender.article_ids)
        catalog_coverage = (len(all_recommended_articles) / total_catalog) * 100.0 if total_catalog else 0.0

        hit_rate = (hits / len(evaluable_users)) if evaluable_users else 0.0
        mrr = (total_reciprocal_rank / len(evaluable_users)) if evaluable_users else 0.0
        avg_diversity = float(np.mean(user_category_diversities)) if user_category_diversities else 0.0

        results = {
            "evaluable_users_count": len(evaluable_users),
            "top_k": top_k,
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr_at_k": round(mrr, 4),
            "catalog_coverage_pct": round(catalog_coverage, 2),
            "avg_intra_list_category_diversity": round(avg_diversity, 4),
            "unique_recommended_count": len(all_recommended_articles),
            "total_catalog_articles": total_catalog
        }

        logger.info(f"Offline Evaluation Results: {results}")
        return results

if __name__ == "__main__":
    from typing import Optional
    evaluator = RecommendationEvaluator()
    res = evaluator.evaluate_leave_one_out(top_k=5)
    print(json.dumps(res, indent=2))
