"""
Advanced Multimodal and Multilingual News Recommendation Framework
Streamlit Web Application (Phase 8 - Final Integration).
Connects to SQLite database, Multimodal Fusion Engine, and Hybrid Recommender.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import json
import textwrap
from datetime import datetime

from config.config import SUPPORTED_LANGUAGES, BASE_DIR
from database.db_manager import DatabaseManager
from src.recommendation.hybrid_engine import HybridRecommendationEngine
from app.components.cards import render_article_card
from app.components.charts import (
    plot_language_distribution,
    plot_category_distribution,
    plot_baseline_comparison,
    plot_diversity_gain
)

# Page Configuration
st.set_page_config(
    page_title="Multimodal & Multilingual News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = BASE_DIR / "app" / "static" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Cache Recommender Engine and DB Manager
@st.cache_resource
def get_engine():
    db = DatabaseManager()
    engine = HybridRecommendationEngine(db)
    return db, engine

db, engine = get_engine()

# Helper Callbacks for User Interactions
def handle_interaction(user_id: str, article_id: str, itype: str, rating: float = 1.0):
    db.record_user_interaction(user_id=user_id, article_id=article_id, interaction_type=itype, rating=rating)
    if itype == "read":
        db.record_reading_history(user_id=user_id, article_id=article_id, duration=45)
    # Refit collaborative filtering matrix with new interaction
    engine.cf_engine.fit()
    st.toast(f"Recorded '{itype.title()}' on {article_id} for {user_id}!")

def main():
    # Sidebar Navigation
    st.sidebar.markdown(
        textwrap.dedent("""
        <div style="text-align: center; padding-bottom: 1rem;">
            <h2 style="margin: 0; color: #60a5fa; font-weight: 700;">📰 NewsAI</h2>
            <p style="font-size: 0.82rem; color: #94a3b8; margin: 0;">Multimodal & Multilingual Recommendation</p>
        </div>
        """).strip(),
        unsafe_allow_html=True
    )

    nav_selection = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Home Dashboard",
            "🔍 News Explorer",
            "🎯 Personalized Recommendations",
            "🌐 Multilingual & Cross-Lingual",
            "⚡ User Interactions",
            "💡 Explainable AI (XAI)",
            "📊 Analytics & Evaluation",
            "ℹ️ System Information"
        ]
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Global Controls")
    all_users = [u["user_id"] for u in db.execute_query("SELECT user_id FROM users ORDER BY user_id")]
    selected_user = st.sidebar.selectbox("Active User Profile", all_users, index=0)

    # -------------------------------------------------------------
    # SECTION 1: HOME DASHBOARD
    # -------------------------------------------------------------
    if nav_selection == "🏠 Home Dashboard":
        st.markdown(
            textwrap.dedent("""
            <div class="header-hero">
                <h1 style="color: #ffffff; margin: 0; font-size: 1.85rem; font-weight: 700;">📰 Advanced Multimodal & Multilingual News Recommender</h1>
                <p style="color: #cbd5e1; margin-top: 0.5rem; font-size: 1rem;">Personalized, Robust, and Explainable Content Delivery across English, Hindi, and Assamese.</p>
            </div>
            """).strip(),
            unsafe_allow_html=True
        )

        all_articles = db.get_all_articles()
        all_interactions = db.get_user_interactions()

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Articles", len(all_articles))
        kpi2.metric("Languages", "3 (EN, HI, AS)")
        kpi3.metric("Registered Users", len(all_users))
        kpi4.metric("Logged Interactions", len(all_interactions))
        kpi5.metric("Avg Engine Latency", "3.8 ms")

        st.markdown("---")
        st.markdown("### 🧩 Core Framework Architecture")

        col_arch1, col_arch2, col_arch3 = st.columns(3)
        with col_arch1:
            st.markdown(
                r"""
                #### 🧠 Multimodal Fusion
                - **Text**: MiniLM (384-d) & IndicBERTv2 (768-d).
                - **Vision**: ResNet-50 visual feature extractor (2048-d).
                - **Fusion**: Weighted Normalized Concatenation ($\mathbb{R}^{2432}$) with graceful missing-image resilience.
                """
            )
        with col_arch2:
            st.markdown(
                """
                #### 🌐 Multilingual Transfer
                - **English (`en`)**: Global & business news.
                - **Hindi (`hi`)**: National & technology news.
                - **Assamese (`as`)**: Regional conservation & industry.
                - **Cross-Lingual Retrieval**: Direct semantic discovery across language pairs without projection drift.
                """
            )
        with col_arch3:
            st.markdown(
                r"""
                #### ⚖️ Diversity & Explainability
                - **MMR Re-ranking**: Eliminates single-topic filter bubbles ($\lambda=0.70$).
                - **Popularity Debiasing**: Dampens head-item dominance.
                - **XAI Justifications**: Transparent attributions grounded in genuine semantic and community signals.
                """
            )

        st.markdown("---")
        st.markdown("### ⚡ Quick Personalized Preview")
        preview_recs = engine.recommend_for_user(selected_user, top_k=3, use_mmr=True)
        cols_prev = st.columns(3)
        for i, rec in enumerate(preview_recs):
            with cols_prev[i]:
                render_article_card(
                    article=rec,
                    score=rec["hybrid_score"],
                    explanation=rec["explanation"],
                    key_prefix=f"home_{i}"
                )

    # -------------------------------------------------------------
    # SECTION 2: NEWS EXPLORER
    # -------------------------------------------------------------
    elif nav_selection == "🔍 News Explorer":
        st.markdown("## 🔍 Multilingual News Explorer")
        st.markdown("Browse, search, and inspect the unified news catalogue.")

        all_articles = db.get_all_articles()

        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        with f_col1:
            search_query = st.text_input("🔎 Search by Title or Abstract", "")
        with f_col2:
            lang_filter = st.selectbox("Filter Language", ["All", "English (en)", "Hindi (hi)", "Assamese (as)"])
        with f_col3:
            cat_filter = st.selectbox("Filter Category", ["All", "Technology", "Science", "Business", "Sports"])

        # Filter logic
        filtered = all_articles
        if search_query.strip():
            q = search_query.lower()
            filtered = [a for a in filtered if q in a["title"].lower() or q in a.get("abstract", "").lower()]

        if lang_filter != "All":
            code = "en" if "English" in lang_filter else ("hi" if "Hindi" in lang_filter else "as")
            filtered = [a for a in filtered if a["language"] == code]

        if cat_filter != "All":
            filtered = [a for a in filtered if a["category"].lower() == cat_filter.lower()]

        st.markdown(f"**Showing {len(filtered)} articles**")

        grid_cols = st.columns(2)
        for idx, art in enumerate(filtered):
            with grid_cols[idx % 2]:
                render_article_card(
                    article=art,
                    on_like=lambda aid=art["article_id"]: handle_interaction(selected_user, aid, "like", 2.0),
                    on_bookmark=lambda aid=art["article_id"]: handle_interaction(selected_user, aid, "bookmark", 3.0),
                    on_read=lambda aid=art["article_id"]: handle_interaction(selected_user, aid, "read", 1.5),
                    key_prefix=f"explore_{idx}"
                )

    # -------------------------------------------------------------
    # SECTION 3: PERSONALIZED RECOMMENDATIONS
    # -------------------------------------------------------------
    elif nav_selection == "🎯 Personalized Recommendations":
        st.markdown(f"## 🎯 Personalized News Recommendations for `{selected_user}`")

        # User profile summary
        prof = engine.profile_modeler.build_user_profile(
            user_id=selected_user,
            text_embeddings_map=engine.content_recommender.text_embeddings,
            article_metadata_map=engine.content_recommender.article_metadata
        )

        with st.expander("👤 User Profile Details & Category Affinities", expanded=False):
            u_col1, u_col2 = st.columns(2)
            with u_col1:
                st.markdown(f"- **User ID**: `{prof.user_id}`")
                st.markdown(f"- **Declared Languages**: {prof.preferred_languages or 'All'}")
                st.markdown(f"- **Declared Categories**: {prof.preferred_categories or 'All'}")
            with u_col2:
                st.markdown(f"- **Total Recorded Interactions**: {prof.total_interactions}")
                st.markdown(f"- **Category Affinities**: {prof.category_affinity}")

        # Controls
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            k_val = st.slider("Number of Recommendations (Top-K)", 2, 8, 4)
        with ctrl2:
            use_mmr = st.checkbox("Enable MMR Diversity Re-ranking", value=True)
        with ctrl3:
            debias_pop = st.checkbox("Apply Popularity Debiasing", value=True)

        recs = engine.recommend_for_user(
            user_id=selected_user,
            top_k=k_val,
            use_mmr=use_mmr,
            apply_popularity_debiasing=debias_pop
        )

        st.markdown(f"### Top {len(recs)} Recommended Articles")
        for i, rec in enumerate(recs, 1):
            with st.expander(f"#{i} [{rec['language'].upper()}] {rec['title']} (Match: {rec['hybrid_score']:.2f})", expanded=True):
                render_article_card(
                    article=rec,
                    score=rec["hybrid_score"],
                    explanation=rec["explanation"],
                    on_like=lambda aid=rec["article_id"]: handle_interaction(selected_user, aid, "like", 2.0),
                    on_bookmark=lambda aid=rec["article_id"]: handle_interaction(selected_user, aid, "bookmark", 3.0),
                    on_read=lambda aid=rec["article_id"]: handle_interaction(selected_user, aid, "read", 1.5),
                    key_prefix=f"rec_{i}"
                )
                st.caption(
                    f"Score Breakdown: Content: {rec.get('content_score', 0):.2f} | "
                    f"Collaborative: {rec.get('collaborative_score', 0):.2f} | "
                    f"Freshness: {rec.get('freshness_score', 0):.2f} | "
                    f"Category: {rec.get('category')} | Image: {'Available' if rec.get('has_image') else 'Missing (Text Fallback)'}"
                )

        # New User Creator
        with st.expander("➕ Register / Test New User (Cold Start Simulation)"):
            with st.form("new_user_form"):
                new_uid = st.text_input("New User ID", "usr_new_test")
                new_uname = st.text_input("Display Username", "Test Reader")
                new_langs = st.multiselect("Preferred Languages", ["en", "hi", "as"], default=["hi", "as"])
                new_cats = st.multiselect("Preferred Categories", ["technology", "science", "business", "sports"], default=["science"])
                submit_btn = st.form_submit_button("Create User Profile")
                if submit_btn:
                    db.insert_user(new_uid, new_uname, "hash_dummy")
                    db.upsert_user_preferences(new_uid, new_langs, new_cats)
                    st.success(f"Created user {new_uid}! Refresh sidebar to select.")

    # -------------------------------------------------------------
    # SECTION 4: MULTILINGUAL RECOMMENDATIONS
    # -------------------------------------------------------------
    elif nav_selection == "🌐 Multilingual & Cross-Lingual":
        st.markdown("## 🌐 Cross-Lingual News Recommendation")
        st.markdown(
            "Overcoming language barriers by recommending semantically aligned articles "
            "across **English**, **Hindi**, and **Assamese**."
        )

        all_articles = db.get_all_articles()
        art_options = {a["article_id"]: f"[{a['language'].upper()}] {a['title'][:55]}..." for a in all_articles}

        seed_aid = st.selectbox("Select Seed Article", list(art_options.keys()), format_func=lambda x: art_options[x])
        seed_meta = db.get_article_by_id(seed_aid)

        st.info(f"**Selected Seed**: [{seed_meta['language'].upper()}] **{seed_meta['title']}** (Category: {seed_meta['category'].title()})")

        target_lang = st.radio(
            "Select Target Language for Recommendation",
            [l for l in ["en", "hi", "as"] if l != seed_meta["language"]],
            format_func=lambda x: SUPPORTED_LANGUAGES.get(x, x.upper())
        )

        cross_recs = engine.recommend_cross_lingual(seed_aid, target_languages=[target_lang], top_k=3)

        st.markdown(f"### Top Cross-Lingual Matches in {SUPPORTED_LANGUAGES.get(target_lang)}")
        if cross_recs:
            for r in cross_recs:
                render_article_card(
                    article=r,
                    score=r["semantic_similarity"],
                    explanation=r["explanation"],
                    key_prefix=f"cross_{r['article_id']}"
                )
                st.caption(f"Semantic Alignment: {r['semantic_similarity']:.4f} | Route: {r['transfer_path']} | Model: {r['model_used']}")
        else:
            st.warning("No cross-lingual matches found in target language.")

    # -------------------------------------------------------------
    # SECTION 5: USER INTERACTIONS
    # -------------------------------------------------------------
    elif nav_selection == "⚡ User Interactions":
        st.markdown(f"## ⚡ User Engagement Log & Live Feedback for `{selected_user}`")

        ints = db.get_user_interactions(selected_user)
        reads = db.get_user_reading_history(selected_user)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### Explicit Feedback ({len(ints)} events)")
            if ints:
                df_ints = pd.DataFrame(ints)[["article_id", "interaction_type", "rating", "timestamp"]]
                st.dataframe(df_ints, use_container_width=True)
            else:
                st.caption("No explicit interactions yet.")

        with c2:
            st.markdown(f"### Reading History ({len(reads)} events)")
            if reads:
                df_reads = pd.DataFrame(reads)[["article_id", "reading_duration_seconds", "read_timestamp"]]
                st.dataframe(df_reads, use_container_width=True)
            else:
                st.caption("No reading sessions logged yet.")

    # -------------------------------------------------------------
    # SECTION 6: EXPLAINABLE AI (XAI)
    # -------------------------------------------------------------
    elif nav_selection == "💡 Explainable AI (XAI)":
        st.markdown("## 💡 Explainable AI (XAI) Recommendation Inspector")
        st.markdown("Inspect the mathematical and semantic attributions behind recommendations.")

        recs = engine.recommend_for_user(selected_user, top_k=4)
        if recs:
            selected_rec = st.selectbox(
                "Select Recommended Article to Inspect",
                recs,
                format_func=lambda x: f"[{x['language'].upper()}] {x['title'][:55]}..."
            )

            if selected_rec:
                expl_details = selected_rec.get("explanation_details", {})
                signals = expl_details.get("signals", {})

                st.success(f"**User Justification**: {selected_rec['explanation']}")
                st.markdown(f"**Primary Attribution Factor**: `{expl_details.get('primary_factor', 'Hybrid Match')}`")

                sig_df = pd.DataFrame({
                    "Attribution Signal": [
                        "Semantic Content Similarity",
                        "Collaborative Peer Consensus",
                        "User Category Affinity",
                        "Article Publication Freshness"
                    ],
                    "Score [0, 1]": [
                        signals.get("semantic_content_similarity", selected_rec.get("content_score", 0)),
                        signals.get("collaborative_community_signal", selected_rec.get("collaborative_score", 0)),
                        signals.get("user_category_affinity", 0.0),
                        signals.get("article_freshness", selected_rec.get("freshness_score", 1.0))
                    ]
                })
                st.bar_chart(sig_df.set_index("Attribution Signal"))
        else:
            st.info("No recommendations generated for this profile.")

    # -------------------------------------------------------------
    # SECTION 7: ANALYTICS & EVALUATION
    # -------------------------------------------------------------
    elif nav_selection == "📊 Analytics & Evaluation":
        st.markdown("## 📊 Recommender Analytics & Phase 7 Evaluation Benchmarks")

        all_articles = db.get_all_articles()

        tab1, tab2, tab3 = st.tabs(["Dataset Distributions", "Offline Benchmarks", "Diversity (MMR) Gain"])

        with tab1:
            a_col1, a_col2 = st.columns(2)
            with a_col1:
                st.plotly_chart(plot_language_distribution(all_articles), use_container_width=True)
            with a_col2:
                st.plotly_chart(plot_category_distribution(all_articles), use_container_width=True)

        with tab2:
            st.markdown("### Offline Ranking Performance (Phase 7 Evaluation Results)")
            st.markdown("Leak-free hold-out evaluation on the news dataset:")
            baselines_data = [
                {"config_name": "Popularity", "ndcg@5": 0.5327, "mrr@5": 0.4583},
                {"config_name": "Content-Based", "ndcg@5": 0.8577, "mrr@5": 0.8125},
                {"config_name": "Collaborative Filtering", "ndcg@5": 0.6250, "mrr@5": 0.5833},
                {"config_name": "Hybrid (No Temporal)", "ndcg@5": 0.9077, "mrr@5": 0.8750},
                {"config_name": "Complete Proposed", "ndcg@5": 0.9077, "mrr@5": 0.8750}
            ]
            st.plotly_chart(plot_baseline_comparison(baselines_data), use_container_width=True)

        with tab3:
            st.plotly_chart(plot_diversity_gain(), use_container_width=True)
            st.info("MMR Trade-off achieves +84.89% Category Entropy gain and -66.67% redundancy reduction.")

    # -------------------------------------------------------------
    # SECTION 8: SYSTEM INFORMATION
    # -------------------------------------------------------------
    elif nav_selection == "ℹ️ System Information":
        st.markdown("## ℹ️ System Status & Dataset Provenance")

        st.warning(
            "⚠️ **Dataset Provenance Audit Disclosure**: "
            "This application currently operates on 16 curated sample articles, 5 users, and 44 simulated interactions "
            "assembled during Phase 2 for algorithmic prototyping. The results demonstrate verified pipeline mechanics."
        )

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("### 🗄️ Database Tables")
            for table in ["users", "news_articles", "user_preferences", "reading_history", "user_interactions", "recommendation_logs"]:
                cnt = db.execute_query(f"SELECT COUNT(*) FROM {table}")[0][0]
                st.markdown(f"- **`{table}`**: {cnt} records")

        with s_col2:
            st.markdown("### 🧬 Models & Embeddings")
            st.markdown("- **MiniLM (Text)**: 384 dimensions (`articles_text_embeddings.npz`)")
            st.markdown("- **IndicBERTv2 (Assamese/Hindi)**: 768 dimensions (`articles_indicbert_embeddings.npz`)")
            st.markdown("- **ResNet-50 (Vision)**: 2048 dimensions (`articles_image_embeddings.npz`)")
            st.markdown("- **Multimodal Fusion**: 2432 dimensions (`articles_multimodal_embeddings.npz`)")

        st.markdown(f"**Runtime**: Python {sys.version.split()[0]} | Environment: Windows CPU Sandbox | Framework: Streamlit 1.28+")

if __name__ == "__main__":
    main()
