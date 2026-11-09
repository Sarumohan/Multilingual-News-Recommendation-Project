"""
Ranking and Diversity Evaluation Metrics Module (Phase 7 - Task 1 & Task 3).
Implements standard Information Retrieval and Recommender Systems metrics:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Catalog Coverage & Catalog Gini Index
- Intra-List Diversity (Shannon Entropy & Pairwise Distance)
- Novelty (Self-Information)
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np

class RankingMetrics:
    """Computes ranking accuracy, diversity, and coverage metrics without data leakage."""

    @staticmethod
    def precision_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """Precision@K = |actual ∩ predicted[:k]| / k"""
        if k <= 0:
            return 0.0
        pred_k = predicted[:k]
        hits = len(set(actual).intersection(set(pred_k)))
        return float(hits / k)

    @staticmethod
    def recall_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """Recall@K = |actual ∩ predicted[:k]| / |actual|"""
        if not actual or k <= 0:
            return 0.0
        pred_k = predicted[:k]
        hits = len(set(actual).intersection(set(pred_k)))
        return float(hits / len(actual))

    @staticmethod
    def reciprocal_rank(actual: List[str], predicted: List[str], k: Optional[int] = None) -> float:
        """Reciprocal Rank = 1 / rank of first relevant item in predicted list."""
        pred_list = predicted[:k] if k is not None else predicted
        actual_set = set(actual)
        for rank, item in enumerate(pred_list, 1):
            if item in actual_set:
                return float(1.0 / rank)
        return 0.0

    @staticmethod
    def average_precision_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """Average Precision@K = sum_{i=1}^k (P@i * rel_i) / min(|actual|, k)"""
        if not actual or k <= 0:
            return 0.0

        pred_k = predicted[:k]
        actual_set = set(actual)
        score = 0.0
        num_hits = 0

        for i, item in enumerate(pred_k, 1):
            if item in actual_set:
                num_hits += 1
                score += (num_hits / i)

        denom = min(len(actual), k)
        return float(score / denom) if denom > 0 else 0.0

    @staticmethod
    def ndcg_at_k(actual: List[str], predicted: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at K (binary relevance)."""
        if not actual or k <= 0:
            return 0.0

        pred_k = predicted[:k]
        actual_set = set(actual)

        dcg = 0.0
        for i, item in enumerate(pred_k, 1):
            if item in actual_set:
                dcg += 1.0 / np.log2(i + 1)

        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(actual), k) + 1))
        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def catalog_coverage(all_recommended_unique: Set[str], total_catalog_count: int) -> float:
        """Percentage of total catalog items recommended at least once."""
        if total_catalog_count <= 0:
            return 0.0
        return float((len(all_recommended_unique) / total_catalog_count) * 100.0)

    @staticmethod
    def catalog_gini_index(recommendation_counts: List[int]) -> float:
        """
        Calculates Gini coefficient of item recommendation distribution.
        G = 0: perfect equality (every article recommended equally).
        G = 1: total inequality (one article monopolizes all recommendations).
        """
        arr = np.array(recommendation_counts, dtype=np.float64)
        if len(arr) == 0 or np.sum(arr) == 0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        gini = (2.0 * np.sum(index * arr)) / (n * np.sum(arr)) - ((n + 1.0) / n)
        return float(max(0.0, min(1.0, gini)))

    @staticmethod
    def novelty_score(predicted: List[str], popularity_probabilities: Dict[str, float]) -> float:
        """
        Measures recommendation novelty as mean self-information:
        Novelty = - 1/K sum_{i=1}^K log2(P(item))
        Higher score indicates recommending less obvious/long-tail articles.
        """
        if not predicted:
            return 0.0
        self_info = []
        for aid in predicted:
            prob = popularity_probabilities.get(aid, 1e-4)
            prob = max(1e-5, min(1.0, prob))
            self_info.append(-np.log2(prob))
        return float(np.mean(self_info)) if self_info else 0.0

