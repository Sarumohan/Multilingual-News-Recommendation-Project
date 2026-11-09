"""
User Profile and Interest Modeling Module (Phase 5 - Task 2).
Models user long-term and short-term news preferences from reading history,
interactions (clicks, likes, bookmarks, shares), categories, and language declarations.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import numpy as np

from config.config import DEFAULT_INTERACTION_WEIGHTS
from database.db_manager import DatabaseManager
from src.utils.helpers import setup_logger

logger = setup_logger("UserProfileModeler")

@dataclass
class UserProfile:
    """Represents a computed user interest profile."""
    user_id: str
    preferred_languages: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    category_affinity: Dict[str, float] = field(default_factory=dict)
    interacted_article_ids: Set[str] = field(default_factory=set)
    interaction_scores: Dict[str, float] = field(default_factory=dict)
    text_vector: Optional[np.ndarray] = None          # 384-d composite vector
    multimodal_vector: Optional[np.ndarray] = None    # 2432-d or projected composite vector
    total_interactions: int = 0
    has_history: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable summary dictionary."""
        return {
            "user_id": self.user_id,
            "preferred_languages": self.preferred_languages,
            "preferred_categories": self.preferred_categories,
            "category_affinity": {k: round(v, 4) for k, v in self.category_affinity.items()},
            "interacted_article_ids": sorted(list(self.interacted_article_ids)),
            "interaction_scores": {k: round(v, 2) for k, v in self.interaction_scores.items()},
            "total_interactions": self.total_interactions,
            "has_history": self.has_history,
            "has_text_vector": self.text_vector is not None,
            "has_multimodal_vector": self.multimodal_vector is not None
        }


class UserProfileModeler:
    """
    Builds dynamic user interest representations by aggregating multi-type interaction signals.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        interaction_weights: Optional[Dict[str, float]] = None,
        temporal_modeler: Optional[Any] = None
    ):
        self.db = db_manager or DatabaseManager()
        self.weights = interaction_weights or DEFAULT_INTERACTION_WEIGHTS
        self.temporal_modeler = temporal_modeler

    def build_user_profile(
        self,
        user_id: str,
        text_embeddings_map: Optional[Dict[str, np.ndarray]] = None,
        multimodal_embeddings_map: Optional[Dict[str, np.ndarray]] = None,
        article_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        exclude_article_ids: Optional[Set[str]] = None,
        apply_temporal_decay: bool = True
    ) -> UserProfile:
        """
        Construct a complete UserProfile for the given user_id.
        Optional exclude_article_ids omits specific articles (used during train/test split or leave-one-out validation).
        If apply_temporal_decay is True and temporal_modeler is set, interactions are exponentially decayed.
        """
        profile = UserProfile(user_id=user_id)
        excluded = exclude_article_ids or set()

        # 1. Load declared preferences
        prefs = self.db.get_user_preferences(user_id)
        if prefs:
            profile.preferred_languages = prefs.get("preferred_languages", [])
            profile.preferred_categories = prefs.get("preferred_categories", [])

        # 2. Load reading history & interactions
        history_rows = self.db.get_user_reading_history(user_id)
        interaction_rows = self.db.get_user_interactions(user_id)

        # Aggregate interaction weights per article
        article_scores: Dict[str, float] = {}
        interacted_ids: Set[str] = set()

        for row in history_rows:
            aid = row["article_id"]
            if aid in excluded:
                continue
            interacted_ids.add(aid)
            dur = row.get("reading_duration_seconds", 0)
            # Duration boost: up to +1.0 for longer reading
            duration_boost = min(1.0, dur / 60.0) if dur else 0.0
            base_read_weight = self.weights.get("read", 1.5)
            raw_w = base_read_weight + duration_boost

            # Apply temporal decay if modeler present
            if apply_temporal_decay and self.temporal_modeler:
                temp_factor = self.temporal_modeler.compute_interaction_temporal_weight(
                    row.get("read_timestamp")
                )
                raw_w *= temp_factor

            article_scores[aid] = article_scores.get(aid, 0.0) + raw_w

        for row in interaction_rows:
            aid = row["article_id"]
            if aid in excluded:
                continue
            interacted_ids.add(aid)
            itype = row["interaction_type"].lower()
            rating = float(row.get("rating", 1.0))
            weight = self.weights.get(itype, 1.0) * rating

            # Apply temporal decay if modeler present
            if apply_temporal_decay and self.temporal_modeler:
                temp_factor = self.temporal_modeler.compute_interaction_temporal_weight(
                    row.get("timestamp")
                )
                weight *= temp_factor

            article_scores[aid] = article_scores.get(aid, 0.0) + weight

        profile.interacted_article_ids = interacted_ids
        profile.interaction_scores = article_scores
        profile.total_interactions = len(history_rows) + len(interaction_rows)
        profile.has_history = len(interacted_ids) > 0

        # 3. Compute Category Affinity Distribution
        if article_metadata_map and interacted_ids:
            cat_weights: Dict[str, float] = {}
            for aid in interacted_ids:
                meta = article_metadata_map.get(aid, {})
                cat = meta.get("category", "general")
                score = article_scores.get(aid, 1.0)
                cat_weights[cat] = cat_weights.get(cat, 0.0) + score

            total_cat_weight = sum(cat_weights.values())
            if total_cat_weight > 0:
                profile.category_affinity = {
                    cat: w / total_cat_weight for cat, w in cat_weights.items()
                }
        elif profile.preferred_categories:
            # Cold-start fallback: uniform distribution across declared categories
            n_cats = len(profile.preferred_categories)
            profile.category_affinity = {c: 1.0 / n_cats for c in profile.preferred_categories}

        # 4. Compute Composite User Embedding Vectors (Text & Multimodal)
        if text_embeddings_map and interacted_ids:
            t_acc = None
            total_w = 0.0
            for aid in interacted_ids:
                if aid in text_embeddings_map:
                    vec = text_embeddings_map[aid]
                    w = article_scores.get(aid, 1.0)
                    if t_acc is None:
                        t_acc = np.zeros_like(vec, dtype=np.float32)
                    t_acc += w * vec
                    total_w += w

            if t_acc is not None and total_w > 0:
                norm = np.linalg.norm(t_acc)
                if norm > 1e-9:
                    profile.text_vector = (t_acc / norm).astype(np.float32)

        if multimodal_embeddings_map and interacted_ids:
            m_acc = None
            total_w = 0.0
            for aid in interacted_ids:
                if aid in multimodal_embeddings_map:
                    vec = multimodal_embeddings_map[aid]
                    w = article_scores.get(aid, 1.0)
                    if m_acc is None:
                        m_acc = np.zeros_like(vec, dtype=np.float32)
                    m_acc += w * vec
                    total_w += w

            if m_acc is not None and total_w > 0:
                norm = np.linalg.norm(m_acc)
                if norm > 1e-9:
                    profile.multimodal_vector = (m_acc / norm).astype(np.float32)

        logger.debug(
            f"Built user profile for {user_id}: {len(interacted_ids)} articles, "
            f"{len(profile.category_affinity)} categories."
        )
        return profile
