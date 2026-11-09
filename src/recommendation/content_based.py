"""
Content-Based and Cross-Lingual Recommendation Module (Phase 5 - Tasks 1 & 5).
Computes semantic content relevance using multilingual text embeddings (MiniLM, IndicBERT)
and multimodal fused embeddings, supporting cross-lingual recommendation across EN, HI, and AS.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import json
from pathlib import Path

from config.config import DATA_EMBEDDINGS_DIR, SUPPORTED_LANGUAGES
from database.db_manager import DatabaseManager
from src.recommendation.user_profile import UserProfile
from src.fusion.similarity import MultimodalSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("ContentBasedRecommender")

class ContentBasedRecommender:
    """
    Computes article-to-profile and article-to-article semantic relevance
    across languages and modalities.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        embeddings_dir: Optional[Path] = None
    ):
        self.db = db_manager or DatabaseManager()
        self.emb_dir = embeddings_dir or DATA_EMBEDDINGS_DIR

        # Load embeddings caches
        self.article_ids: List[str] = []
        self.article_metadata: Dict[str, Dict[str, Any]] = {}

        # 1. MiniLM Multilingual Text Embeddings (384-d) - Validated Shared Space
        self.text_embeddings: Dict[str, np.ndarray] = {}
        # 2. Multimodal Fused Embeddings (2432-d concat & 384-d projected)
        self.multimodal_concat: Dict[str, np.ndarray] = {}
        self.multimodal_projected: Dict[str, np.ndarray] = {}
        # 3. IndicBERT Indic Embeddings (768-d) - Validated Indic Space (HI <-> AS)
        self.indicbert_embeddings: Dict[str, np.ndarray] = {}

        self.has_image_map: Dict[str, bool] = {}

        self.load_all_representations()

    def load_all_representations(self) -> None:
        """Load text, visual, and multimodal embeddings from disk."""
        # 1. Load articles metadata from database
        articles = self.db.get_all_articles()
        self.article_metadata = {a["article_id"]: a for a in articles}
        self.article_ids = sorted(list(self.article_metadata.keys()))

        # 2. Text (MiniLM - 384-d)
        text_path = self.emb_dir / "articles_text_embeddings.npz"
        if text_path.exists():
            data = np.load(text_path, allow_pickle=True)
            a_ids = list(data["article_ids"])
            embs = data["embeddings"]
            for i, aid in enumerate(a_ids):
                self.text_embeddings[aid] = embs[i].astype(np.float32)

        # 3. IndicBERT (768-d)
        indic_path = self.emb_dir / "articles_indicbert_embeddings.npz"
        if indic_path.exists():
            data = np.load(indic_path, allow_pickle=True)
            a_ids = list(data["article_ids"])
            embs = data["embeddings"]
            for i, aid in enumerate(a_ids):
                self.indicbert_embeddings[aid] = embs[i].astype(np.float32)

        # 4. Multimodal Embeddings
        mm_path = self.emb_dir / "articles_multimodal_embeddings.npz"
        if mm_path.exists():
            data = np.load(mm_path, allow_pickle=True)
            a_ids = list(data["article_ids"])
            concat = data["fused_concat"]
            proj = data["fused_projected"]
            has_img = data["has_image"]
            for i, aid in enumerate(a_ids):
                self.multimodal_concat[aid] = concat[i].astype(np.float32)
                self.multimodal_projected[aid] = proj[i].astype(np.float32)
                self.has_image_map[aid] = bool(has_img[i])

        logger.info(
            f"Loaded {len(self.article_ids)} articles for Content-Based Recommender. "
            f"MiniLM: {len(self.text_embeddings)}, Multimodal: {len(self.multimodal_concat)}, "
            f"IndicBERT: {len(self.indicbert_embeddings)}"
        )

    def get_candidate_vector(
        self,
        article_id: str,
        representation: str = "multimodal"
    ) -> Optional[np.ndarray]:
        """
        Retrieve representation vector for an article.
        Ensures strict separation of embedding spaces.
        """
        if representation == "multimodal":
            return self.multimodal_concat.get(article_id, self.text_embeddings.get(article_id))
        elif representation == "multimodal_projected":
            return self.multimodal_projected.get(article_id, self.text_embeddings.get(article_id))
        elif representation == "text_minilm":
            return self.text_embeddings.get(article_id)
        elif representation == "indicbert":
            return self.indicbert_embeddings.get(article_id)
        else:
            raise ValueError(f"Unknown representation type: {representation}")

    def score_candidates_for_user_profile(
        self,
        profile: UserProfile,
        candidate_ids: Optional[List[str]] = None,
        representation: str = "multimodal",
        category_weight: float = 0.15,
        target_languages: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute content-based match scores between user profile and candidate articles.
        Returns dict of {article_id: normalized_content_score in [0, 1]}.
        """
        if candidate_ids is None:
            candidate_ids = self.article_ids

        # Filter candidate languages if target_languages or preferred_languages are given
        allowed_languages = set(target_languages) if target_languages else None
        if allowed_languages is None and profile.preferred_languages:
            allowed_languages = set(profile.preferred_languages)

        scores: Dict[str, float] = {}

        # Determine target vector from profile
        if representation == "multimodal" and profile.multimodal_vector is not None:
            user_vec = profile.multimodal_vector
        elif profile.text_vector is not None:
            user_vec = profile.text_vector
            representation = "text_minilm"
        else:
            user_vec = None

        for aid in candidate_ids:
            meta = self.article_metadata.get(aid, {})
            art_lang = meta.get("language")

            # Language check
            if allowed_languages and art_lang not in allowed_languages:
                continue

            # 1. Semantic Embedding Similarity
            sim_score = 0.0
            if user_vec is not None:
                art_vec = self.get_candidate_vector(aid, representation=representation)
                if art_vec is not None and art_vec.shape == user_vec.shape:
                    sim_score = max(0.0, MultimodalSimilarityEngine.compute_similarity(user_vec, art_vec))

            # 2. Category Affinity Bonus
            cat = meta.get("category", "")
            cat_score = profile.category_affinity.get(cat, 0.0)

            # Combined content score: (1 - cat_weight) * semantic_sim + cat_weight * category_affinity
            if user_vec is not None:
                combined_score = (1.0 - category_weight) * sim_score + category_weight * cat_score
            else:
                # Cold start profile with only category affinities
                combined_score = cat_score

            scores[aid] = float(combined_score)

        # Normalize into [0, 1]
        max_s = max(scores.values()) if scores else 0.0
        if max_s > 0:
            scores = {aid: s / max_s for aid, s in scores.items()}

        return scores

    def recommend_similar_articles(
        self,
        seed_article_id: str,
        top_k: int = 5,
        target_languages: Optional[List[str]] = None,
        representation: str = "multimodal",
        exclude_seed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recommend articles semantically similar to a seed article (Item-to-Item Content Filtering).
        Supports cross-lingual candidate discovery.
        """
        seed_vec = self.get_candidate_vector(seed_article_id, representation=representation)
        if seed_vec is None:
            logger.warning(f"No vector found for seed article {seed_article_id}")
            return []

        seed_meta = self.article_metadata.get(seed_article_id, {})
        allowed_languages = set(target_languages) if target_languages else None

        candidates = []
        for aid in self.article_ids:
            if exclude_seed and aid == seed_article_id:
                continue

            cand_meta = self.article_metadata.get(aid, {})
            cand_lang = cand_meta.get("language")

            if allowed_languages and cand_lang not in allowed_languages:
                continue

            cand_vec = self.get_candidate_vector(aid, representation=representation)
            if cand_vec is None or cand_vec.shape != seed_vec.shape:
                continue

            sim = MultimodalSimilarityEngine.compute_similarity(seed_vec, cand_vec)
            # Bound in [0, 1]
            score = max(0.0, float(sim))

            candidates.append({
                "article_id": aid,
                "title": cand_meta.get("title", ""),
                "language": cand_lang,
                "category": cand_meta.get("category", ""),
                "similarity_score": round(score, 4),
                "seed_article_id": seed_article_id,
                "is_cross_lingual": (cand_lang != seed_meta.get("language"))
            })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]

    def recommend_cross_lingual(
        self,
        seed_article_id: str,
        target_languages: Optional[List[str]] = None,
        top_k: int = 5,
        use_indicbert_for_indic: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate explicitly cross-lingual news recommendations.
        Strictly adheres to embedding space isolation:
        - Uses IndicBERT only when comparing Indic-to-Indic (Hindi <-> Assamese).
        - Uses MiniLM for all English <-> Indic comparisons (EN <-> HI, EN <-> AS).
        """
        seed_meta = self.article_metadata.get(seed_article_id, {})
        source_lang = seed_meta.get("language", "")

        # Default targets: all supported languages except source
        if not target_languages:
            target_languages = [l for l in SUPPORTED_LANGUAGES.keys() if l != source_lang]
        else:
            # Exclude source language to guarantee genuine cross-lingual transfer
            target_languages = [l for l in target_languages if l != source_lang]

        results = []
        for t_lang in target_languages:
            # Select appropriate validated shared representation
            is_indic_pair = (source_lang in ["hi", "as"]) and (t_lang in ["hi", "as"])
            
            if is_indic_pair and use_indicbert_for_indic and seed_article_id in self.indicbert_embeddings:
                rep = "indicbert"
                seed_v = self.indicbert_embeddings[seed_article_id]
                pool = self.indicbert_embeddings
                model_used = "ai4bharat/IndicBERTv2-SS"
            else:
                rep = "text_minilm"
                seed_v = self.text_embeddings[seed_article_id]
                pool = self.text_embeddings
                model_used = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

            candidates_in_lang = []
            for aid, cand_v in pool.items():
                if aid == seed_article_id:
                    continue
                cand_meta = self.article_metadata.get(aid, {})
                if cand_meta.get("language") == t_lang:
                    sim = float(MultimodalSimilarityEngine.compute_similarity(seed_v, cand_v))
                    candidates_in_lang.append({
                        "article_id": aid,
                        "title": cand_meta.get("title", ""),
                        "source_language": source_lang,
                        "target_language": t_lang,
                        "category": cand_meta.get("category", ""),
                        "semantic_similarity": round(max(0.0, sim), 4),
                        "model_used": model_used,
                        "transfer_path": f"{source_lang.upper()} -> {t_lang.upper()}"
                    })

            candidates_in_lang.sort(key=lambda x: x["semantic_similarity"], reverse=True)
            results.extend(candidates_in_lang[:top_k])

        results.sort(key=lambda x: x["semantic_similarity"], reverse=True)
        return results[:top_k]

