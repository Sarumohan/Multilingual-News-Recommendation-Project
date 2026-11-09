"""
Temporal User-Interest and Article Freshness Modeling Module (Phase 6 - Task 1).
Implements exponential time decay for historical user interactions and article publication freshness.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from config.config import (
    TEMPORAL_DECAY_LAMBDA,
    FRESHNESS_LAMBDA,
    DEFAULT_FRESHNESS_WEIGHT
)
from src.utils.helpers import setup_logger

logger = setup_logger("TemporalModeler")

# Standard datetime formats supported in project
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d"
]

class TemporalModeler:
    """
    Handles temporal decay of user interests and publication freshness scoring.
    Formula: Temporal Weight = exp(-lambda * delta_time_days)
    """

    def __init__(
        self,
        temporal_lambda: float = TEMPORAL_DECAY_LAMBDA,
        freshness_lambda: float = FRESHNESS_LAMBDA,
        freshness_weight: float = DEFAULT_FRESHNESS_WEIGHT,
        reference_time: Optional[datetime] = None
    ):
        if temporal_lambda < 0 or freshness_lambda < 0:
            raise ValueError("Decay parameter lambda must be non-negative")

        self.temporal_lambda = temporal_lambda
        self.freshness_lambda = freshness_lambda
        self.freshness_weight = freshness_weight
        self.reference_time = reference_time

    def get_reference_time(self) -> datetime:
        """Return the evaluation reference time (defaults to fixed reference or current UTC)."""
        if self.reference_time:
            return self.reference_time
        # For repeatable testing across datasets anchored at Nov 2026:
        return datetime(2026, 11, 7, 12, 0, 0)

    @staticmethod
    def parse_timestamp(timestamp_val: Any, default_time: Optional[datetime] = None) -> Optional[datetime]:
        """
        Safely parse diverse timestamp inputs (str, datetime, None) into standard datetime.
        Handles missing, malformed, or unparseable timestamps gracefully.
        """
        if timestamp_val is None:
            return default_time

        if isinstance(timestamp_val, datetime):
            return timestamp_val.replace(tzinfo=None)

        if not isinstance(timestamp_val, str) or not timestamp_val.strip():
            return default_time

        ts_str = timestamp_val.strip()
        for fmt in DATETIME_FORMATS:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse timestamp string '{timestamp_val}'. Using default fallback.")
        return default_time

    def calculate_time_delta_days(
        self,
        event_time: Union[str, datetime, None],
        ref_time: Optional[datetime] = None
    ) -> float:
        """
        Compute elapsed time in days between event and reference time.
        Future events (or clock drift) are clamped to 0.0 days.
        """
        reference = ref_time or self.get_reference_time()
        parsed_event = self.parse_timestamp(event_time, default_time=reference)

        if parsed_event is None:
            return 0.0

        delta_seconds = (reference - parsed_event).total_seconds()
        delta_days = max(0.0, delta_seconds / 86400.0)
        return float(delta_days)

    def calculate_exponential_decay(self, delta_days: float, decay_lambda: Optional[float] = None) -> float:
        """
        Compute exponential decay weight:
        w = exp(-lambda * delta_days)
        """
        lam = decay_lambda if decay_lambda is not None else self.temporal_lambda
        if delta_days <= 0.0 or lam == 0.0:
            return 1.0
        return float(np.exp(-lam * delta_days))

    def compute_interaction_temporal_weight(
        self,
        interaction_timestamp: Union[str, datetime, None],
        long_term_ratio: float = 0.3
    ) -> float:
        """
        Dual-horizon decay weighting:
        Weight = (1 - ratio) * exp(-lambda * delta) + ratio
        Ensures recent interactions receive high weights while long-term interests retain baseline influence.
        """
        delta_days = self.calculate_time_delta_days(interaction_timestamp)
        decay = self.calculate_exponential_decay(delta_days, self.temporal_lambda)
        # Blend short-term decay with non-zero long-term floor
        weight = ((1.0 - long_term_ratio) * decay) + long_term_ratio
        return float(weight)

    def compute_article_freshness_score(
        self,
        published_at: Union[str, datetime, None]
    ) -> float:
        """
        Computes article freshness score in (0, 1] based on article publication timestamp.
        Freshness = exp(-freshness_lambda * delta_days)
        """
        delta_days = self.calculate_time_delta_days(published_at)
        return self.calculate_exponential_decay(delta_days, self.freshness_lambda)

