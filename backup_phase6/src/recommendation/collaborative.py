"""
Collaborative Filtering Recommendation Module (Phase 5 - Task 3).
Implements Memory-based Collaborative Filtering (Item-Item and User-User)
over the user interaction matrix with explicit handling of sparse matrices.

NOTE ON DATA AUTHENTICITY:
The interaction matrix in this phase is constructed from simulated interactions
curated during development for testing and verification purposes.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd

from database.db_manager import DatabaseManager
from config.config import DEFAULT_INTERACTION_WEIGHTS
from src.utils.helpers import setup_logger

logger = setup_logger("CollaborativeFiltering")

class CollaborativeFilteringEngine:
    """
    Memory-Based Collaborative Filtering Engine supporting Item-Item and User-User cosine similarity.
    """

    IS_SIMULATED_DATASET = True

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        interaction_weights: Optional[Dict[str, float]] = None
    ):
        self.db = db_manager or DatabaseManager()
        self.weights = interaction_weights or DEFAULT_INTERACTION_WEIGHTS

        # Matrix structures
        self.user_ids: List[str] = []
        self.article_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.article_to_idx: Dict[str, int] = {}

        self.interaction_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.item_similarity_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.user_similarity_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)

        self._is_fitted = False
        self.fit()

    def fit(self) -> "CollaborativeFilteringEngine":
        """
        Build interaction matrix from database and precompute similarity matrices.
        """
        interactions = self.db.get_user_interactions()
        all_articles = self.db.get_all_articles()

        # Unique user list from interactions and db
        users_in_db = [u["user_id"] for u in self.db.execute_query("SELECT user_id FROM users")]
        users_in_int = list({r["user_id"] for r in interactions})
        self.user_ids = sorted(list(set(users_in_db + users_in_int)))

        # Unique article list from articles table and interactions
        articles_in_db = [a["article_id"] for a in all_articles]
        articles_in_int = list({r["article_id"] for r in interactions})
        self.article_ids = sorted(list(set(articles_in_db + articles_in_int)))

        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.article_to_idx = {aid: i for i, aid in enumerate(self.article_ids)}

        num_users = len(self.user_ids)
        num_articles = len(self.article_ids)

        # 1. Populate Interaction Matrix
        R = np.zeros((num_users, num_articles), dtype=np.float32)

        for row in interactions:
            u_id = row["user_id"]
            a_id = row["article_id"]
            if u_id in self.user_to_idx and a_id in self.article_to_idx:
                u_idx = self.user_to_idx[u_id]
                a_idx = self.article_to_idx[a_id]
                itype = row["interaction_type"].lower()
                rating = float(row.get("rating", 1.0))
                weight = self.weights.get(itype, 1.0) * rating
                R[u_idx, a_idx] += weight

        self.interaction_matrix = R

        # 2. Compute Item-Item Cosine Similarity Matrix (Articles x Articles)
        # item columns: R[:, i]
        item_norms = np.linalg.norm(R, axis=0, keepdims=True)  # (1, num_articles)
        item_norms = np.where(item_norms == 0, 1e-9, item_norms)
        normalized_items = R / item_norms  # (num_users, num_articles)
        self.item_similarity_matrix = np.dot(normalized_items.T, normalized_items).astype(np.float32)
        np.fill_diagonal(self.item_similarity_matrix, 0.0)  # zero self-similarity

        # 3. Compute User-User Cosine Similarity Matrix (Users x Users)
        # user rows: R[u, :]
        user_norms = np.linalg.norm(R, axis=1, keepdims=True)  # (num_users, 1)
        user_norms = np.where(user_norms == 0, 1e-9, user_norms)
        normalized_users = R / user_norms
        self.user_similarity_matrix = np.dot(normalized_users, normalized_users.T).astype(np.float32)
        np.fill_diagonal(self.user_similarity_matrix, 0.0)  # zero self-similarity

        self._is_fitted = True
        logger.info(
            f"Fitted Collaborative Filtering Engine: {num_users} users x {num_articles} articles. "
            f"Sparsity: {self.get_matrix_sparsity():.2f}%"
        )
        return self

    def get_matrix_sparsity(self) -> float:
        """Calculate the percentage of zero entries in the interaction matrix."""
        total = self.interaction_matrix.size
        if total == 0:
            return 100.0
        non_zeros = np.count_nonzero(self.interaction_matrix)
        return float((1.0 - (non_zeros / total)) * 100.0)

    def predict_item_item_score(self, user_idx: int, article_idx: int) -> float:
        """
        Estimate user preference for an article using Item-Item similarity.
        R_hat(u, i) = sum_{j in rated} (sim(i, j) * R_{u, j}) / sum_{j in rated} |sim(i, j)|
        """
        user_ratings = self.interaction_matrix[user_idx]  # (num_articles,)
        rated_indices = np.where(user_ratings > 0)[0]

        if len(rated_indices) == 0:
            return 0.0

        sims = self.item_similarity_matrix[article_idx, rated_indices]
        ratings = user_ratings[rated_indices]

        # Consider positive similarities
        pos_mask = sims > 0
        if not np.any(pos_mask):
            return 0.0

        numerator = np.sum(sims[pos_mask] * ratings[pos_mask])
        denominator = np.sum(sims[pos_mask])

        if denominator < 1e-9:
            return 0.0

        return float(numerator / denominator)

    def predict_user_user_score(self, user_idx: int, article_idx: int) -> float:
        """
        Estimate user preference for an article using User-User similarity.
        R_hat(u, i) = sum_{v != u} (sim(u, v) * R_{v, i}) / sum_{v != u} |sim(u, v)|
        """
        item_ratings = self.interaction_matrix[:, article_idx]  # (num_users,)
        active_users = np.where(item_ratings > 0)[0]
        active_users = active_users[active_users != user_idx]

        if len(active_users) == 0:
            return 0.0

        sims = self.user_similarity_matrix[user_idx, active_users]
        ratings = item_ratings[active_users]

        pos_mask = sims > 0
        if not np.any(pos_mask):
            return 0.0

        numerator = np.sum(sims[pos_mask] * ratings[pos_mask])
        denominator = np.sum(sims[pos_mask])

        if denominator < 1e-9:
            return 0.0

        return float(numerator / denominator)

    def get_collaborative_scores(
        self,
        user_id: str,
        method: str = "item_item"
    ) -> Dict[str, float]:
        """
        Compute normalized collaborative filtering scores in [0, 1] for all articles for a user.
        """
        if not self._is_fitted:
            self.fit()

        if user_id not in self.user_to_idx:
            # Cold-start user not in interaction matrix
            return {aid: 0.0 for aid in self.article_ids}

        u_idx = self.user_to_idx[user_id]
        scores: Dict[str, float] = {}

        raw_scores = []
        for a_idx, aid in enumerate(self.article_ids):
            if method == "item_item":
                score = self.predict_item_item_score(u_idx, a_idx)
            elif method == "user_user":
                score = self.predict_user_user_score(u_idx, a_idx)
            else:
                raise ValueError(f"Unknown collaborative filtering method: {method}")
            scores[aid] = score
            raw_scores.append(score)

        max_score = max(raw_scores) if raw_scores else 0.0
        # Normalize into [0, 1]
        if max_score > 0:
            scores = {aid: s / max_score for aid, s in scores.items()}

        return scores

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
        method: str = "item_item",
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate top-K collaborative recommendations for a user.
        """
        scores = self.get_collaborative_scores(user_id, method=method)

        interacted = set()
        if exclude_interacted and user_id in self.user_to_idx:
            u_idx = self.user_to_idx[user_id]
            interacted = {
                self.article_ids[i]
                for i in np.where(self.interaction_matrix[u_idx] > 0)[0]
            }

        candidates = [
            {"article_id": aid, "cf_score": score, "method": method}
            for aid, score in scores.items()
            if aid not in interacted and score > 0
        ]

        candidates.sort(key=lambda x: x["cf_score"], reverse=True)
        return candidates[:top_k]

