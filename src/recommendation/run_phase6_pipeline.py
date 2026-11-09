"""
Phase 6 Advanced Intelligence Demonstration Script.
Executes and benchmarks:
- Temporal user modeling and article freshness
- Multi-signal transparent explainability
- Maximal Marginal Relevance (MMR) vs Original Ranking
- Popularity-bias reduction
- Recommendation stability evaluation
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.hybrid_engine import HybridRecommendationEngine
from src.evaluation.stability import StabilityEvaluator
from database.db_manager import DatabaseManager

def run_phase6_demonstration():
    print("=" * 80)
    print("REVISED PHASE 6: ADVANCED INTELLIGENCE DEMONSTRATION")
    print("=" * 80)

    engine = HybridRecommendationEngine()
    db = DatabaseManager()

    # 1. Temporal User Modeling & Article Freshness
    print("\n--- 1. TEMPORAL USER MODELING & ARTICLE FRESHNESS ---")
    tm = engine.temporal_modeler
    print(f"Decay Parameters: Temporal lambda = {tm.temporal_lambda:.5f} (half-life = 3.0 days)")
    print(f"                  Freshness lambda = {tm.freshness_lambda:.5f} (half-life = 5.0 days)")
    print(f"Reference Evaluation Timestamp: {tm.get_reference_time()}")

    test_dates = [
        ("2026-11-07 10:00:00", "Breaking (2 hours ago)"),
        ("2026-11-04 12:00:00", "3 Days Old (Half-Life)"),
        ("2026-10-28 12:00:00", "10 Days Old (Historical)"),
        (None, "Missing Timestamp")
    ]
    for ts, desc in test_dates:
        decay = tm.calculate_exponential_decay(tm.calculate_time_delta_days(ts))
        fresh = tm.compute_article_freshness_score(ts)
        long_term_w = tm.compute_interaction_temporal_weight(ts)
        print(f"  • {desc:24s} | Decay: {decay:.4f} | Freshness Score: {fresh:.4f} | Interaction Weight (w/ floor): {long_term_w:.4f}")

    # 2. Hybrid Recommendations with MMR vs Original Ranking
    print("\n--- 2. MMR DIVERSITY RE-RANKING VS ORIGINAL RANKING (usr_001) ---")
    # Original (No MMR)
    orig_recs = engine.recommend_for_user("usr_001", top_k=5, use_mmr=False, apply_diversity=False)
    # MMR Ranking
    mmr_recs = engine.recommend_for_user("usr_001", top_k=5, use_mmr=True, mmr_lambda=0.70)

    print("\n[Original Top-5 Slate (Pure Relevance)]:")
    for i, r in enumerate(orig_recs, 1):
        print(f"  {i}. [{r['language'].upper()}] {r['article_id']} ({r['category']:10s}) | Score: {r['hybrid_score']:.4f} | Fresh: {r['freshness_score']:.4f}")
        print(f"     Title: {r['title']}")

    print("\n[MMR Top-5 Slate (Diversity-Aware lambda=0.70)]:")
    for i, r in enumerate(mmr_recs, 1):
        print(f"  {i}. [{r['language'].upper()}] {r['article_id']} ({r['category']:10s}) | Score: {r['hybrid_score']:.4f} | Orig Rank: {r.get('original_rank', i)} | MMR Score: {r.get('mmr_score', 0):.4f}")
        print(f"     Title: {r['title']}")
        print(f"     XAI Explanation: {r['explanation']}")

    emb_map = engine.content_recommender.multimodal_concat
    diversity_comp = engine.diversity_engine.compare_rankings(orig_recs, mmr_recs, emb_map)
    print("\nDiversity Benchmark:")
    print(f"  • Original Category Entropy: {diversity_comp['original_category_entropy']:.4f} -> MMR Entropy: {diversity_comp['mmr_category_entropy']:.4f}")
    print(f"  • Original Redundancy (Sim): {diversity_comp['original_pairwise_similarity']:.4f} -> MMR Redundancy: {diversity_comp['mmr_pairwise_similarity']:.4f}")
    print(f"  • Unique Categories:         {diversity_comp['original_unique_categories']} -> {diversity_comp['mmr_unique_categories']}")
    print(f"  • Diversity Gain:            +{diversity_comp['diversity_gain_pct']:.2f}%")
    print(f"  • Redundancy Reduction:      -{diversity_comp['similarity_reduction_pct']:.2f}%")

    # 3. Popularity Bias Reduction Analysis
    print("\n--- 3. POPULARITY-BIAS REDUCTION ANALYSIS ---")
    div_eng = engine.diversity_engine
    test_scores = [1.0, 1.0, 1.0, 1.0]
    test_pops = [0, 5, 15, 30]
    print(f"Popularity Dampening Parameter: gamma = {div_eng.popularity_gamma}")
    for raw_s, pop in zip(test_scores, test_pops):
        deb = div_eng.apply_popularity_debiasing(raw_s, pop)
        diff_pct = ((raw_s - deb) / raw_s) * 100.0
        print(f"  • Interactions = {pop:2d} | Raw Score = {raw_s:.4f} -> Debiased = {deb:.4f} (Dampened by {diff_pct:.2f}%)")

    # 4. Recommendation Stability Evaluation
    print("\n--- 4. RECOMMENDATION STABILITY EVALUATION ---")
    evaluator = StabilityEvaluator(engine)
    stability_res = evaluator.evaluate_all(user_id="usr_001", top_k=5)
    print(f"User: usr_001 | Top-K: 5")
    print(f"  • Weight Jitter Test (+/- 10% Noise): Overlap = {stability_res['weight_jitter_test']['topk_overlap']:.4f}, Correlation = {stability_res['weight_jitter_test']['rank_correlation']:.4f} (Stable: {stability_res['weight_jitter_test']['is_stable']})")
    print(f"  • Interaction Dropout Test:            Overlap = {stability_res['interaction_dropout_test']['topk_overlap']:.4f}, Correlation = {stability_res['interaction_dropout_test']['rank_correlation']:.4f} (Stable: {stability_res['interaction_dropout_test']['is_stable']})")
    print(f"  • Overall Stability Status:             {'PASSED' if stability_res['overall_stability_pass'] else 'FAILED'}")

    print("\n" + "=" * 80)
    print("PHASE 6 ADVANCED INTELLIGENCE COMPLETE - ALL SYSTEMS FUNCTIONAL")
    print("=" * 80)

if __name__ == "__main__":
    run_phase6_demonstration()

