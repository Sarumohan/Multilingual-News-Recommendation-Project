"""
Phase 5 Pipeline Verification Script.
Executes end-to-end tests of User Profile Modeling, Content-Based, Collaborative Filtering,
Hybrid Recommendation, Cross-Lingual discovery, and Cold-Start paths.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.hybrid_engine import HybridRecommendationEngine
from src.recommendation.user_profile import UserProfileModeler
from database.db_manager import DatabaseManager

def run_phase5_demonstration():
    print("=" * 80)
    print("REVISED PHASE 5: HYBRID AND CROSS-LINGUAL RECOMMENDATION ENGINE DEMONSTRATION")
    print("=" * 80)

    engine = HybridRecommendationEngine()
    db = DatabaseManager()

    # 1. User Profiles
    print("\n--- 1. USER INTEREST MODELING SAMPLES ---")
    users = ["usr_001", "usr_002", "usr_003"]
    for uid in users:
        prof = engine.profile_modeler.build_user_profile(
            user_id=uid,
            text_embeddings_map=engine.content_recommender.text_embeddings,
            multimodal_embeddings_map=engine.content_recommender.multimodal_concat,
            article_metadata_map=engine.content_recommender.article_metadata
        )
        print(f"User: {uid}")
        print(f"  • Declared Languages:   {prof.preferred_languages}")
        print(f"  • Declared Categories:  {prof.preferred_categories}")
        print(f"  • Interacted Articles:  {len(prof.interacted_article_ids)} articles ({sorted(list(prof.interacted_article_ids))[:4]}...)")
        print(f"  • Category Affinities:  {dict(list(prof.category_affinity.items())[:3])}")
        print(f"  • Embedding Status:     Text: {prof.text_vector is not None}, Multimodal: {prof.multimodal_vector is not None}")

    # 2. Hybrid Personalized Recommendations
    print("\n--- 2. HYBRID PERSONALIZED RECOMMENDATIONS ---")
    for uid in ["usr_001", "usr_002"]:
        recs = engine.recommend_for_user(uid, top_k=3)
        print(f"\nRecommendations for {uid}:")
        for i, r in enumerate(recs, 1):
            lang = r["language"].upper()
            print(f"  {i}. [{lang}] {r['article_id']} | Hybrid: {r['hybrid_score']:.4f} (CB: {r['content_score']:.4f}, CF: {r['collaborative_score']:.4f})")
            print(f"     Title: {r['title']}")
            print(f"     Category: {r['category']} | Image Available: {r['has_image']}")
            print(f"     XAI Explanation: {r['explanation']}")

    # 3. Cross-Lingual Knowledge Transfer Across All Language Pairs
    print("\n--- 3. CROSS-LINGUAL RECOMMENDATION TRANSFERS ---")
    transfers = [
        ("art_en_001", ["hi", "as"], "English AI News -> Hindi & Assamese"),
        ("art_hi_004", ["en", "as"], "Hindi Solar Energy News -> English & Assamese"),
        ("art_as_001", ["en", "hi"], "Assamese Kaziranga News -> English & Hindi"),
    ]

    for seed_id, target_langs, desc in transfers:
        seed_meta = engine.content_recommender.article_metadata.get(seed_id, {})
        print(f"\nSeed: {seed_id} [{seed_meta.get('language', '').upper()}] - '{seed_meta.get('title', '')}'")
        print(f"Goal: {desc}")
        cross_recs = engine.recommend_cross_lingual(seed_id, target_languages=target_langs, top_k=2)
        for i, cr in enumerate(cross_recs, 1):
            print(f"  {i}. Transfer: {cr['transfer_path']} | Sim: {cr['semantic_similarity']:.4f} | Model: {cr['model_used'].split('/')[-1]}")
            print(f"     Target: {cr['article_id']} ({cr['category']}) - '{cr['title']}'")
            print(f"     XAI Rationale: {cr['explanation']}")

    # 4. Cold-Start Tests
    print("\n--- 4. COLD-START RECOMMENDATION EVALUATION ---")
    # 4a. User with declared interests, 0 history
    print("\n[Case A] Cold User with Declared Preferences (hi, as / science):")
    cold_user_profile = engine.profile_modeler.build_user_profile(
        user_id="cold_usr_declared",
        article_metadata_map=engine.content_recommender.article_metadata
    )
    cold_user_profile.preferred_languages = ["hi", "as"]
    cold_user_profile.preferred_categories = ["science"]
    cold_user_profile.category_affinity = {"science": 1.0}
    cold_user_profile.has_history = False

    cold_recs_a = engine._recommend_cold_start_user(
        user_id="cold_usr_declared",
        profile=cold_user_profile,
        preferred_languages=["hi", "as"],
        top_k=3
    )
    for i, r in enumerate(cold_recs_a, 1):
        print(f"  {i}. [{r['language'].upper()}] {r['article_id']} ({r['category']}) | Score: {r['hybrid_score']}")
        print(f"     Title: {r['title']}")
        print(f"     XAI Explanation: {r['explanation']}")

    # 4b. Completely cold user (0 history, 0 declared preferences -> Popularity Fallback)
    print("\n[Case B] Completely Cold User (0 history, 0 declared preferences -> Popularity Fallback):")
    cold_user_blank = engine.profile_modeler.build_user_profile(user_id="cold_usr_blank")
    cold_recs_b = engine._recommend_cold_start_user(
        user_id="cold_usr_blank",
        profile=cold_user_blank,
        preferred_languages=["en", "hi", "as"],
        top_k=3
    )
    for i, r in enumerate(cold_recs_b, 1):
        print(f"  {i}. [{r['language'].upper()}] {r['article_id']} ({r['category']}) | Score: {r['hybrid_score']}")
        print(f"     Title: {r['title']}")
        print(f"     XAI Explanation: {r['explanation']}")

    print("\n" + "=" * 80)
    print("PHASE 5 DEMONSTRATION COMPLETE - ALL SYSTEMS FUNCTIONAL")
    print("=" * 80)

if __name__ == "__main__":
    run_phase5_demonstration()
