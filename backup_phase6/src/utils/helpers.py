"""
General helper functions for logging, math operations, and text formatting.
"""

import logging
import sys
import math
from typing import Optional, List, Union

def setup_logger(name: str = "NewsRecommender", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def cosine_similarity_matrix(a, b=None):
    """
    Compute cosine similarity matrix between row vectors of a and b.
    Supports both NumPy ndarrays and pure Python nested lists.
    """
    try:
        import numpy as np
        if isinstance(a, np.ndarray):
            if b is None:
                b = a
            a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
            b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
            return np.dot(a_norm, b_norm.T)
    except ImportError:
        pass

    # Pure Python fallback
    if b is None:
        b = a

    def norm(v):
        val = math.sqrt(sum(x * x for x in v))
        return val if val > 1e-9 else 1e-9

    def dot(v1, v2):
        return sum(x1 * x2 for x1, x2 in zip(v1, v2))

    matrix = []
    for row_a in a:
        norm_a = norm(row_a)
        res_row = []
        for row_b in b:
            norm_b = norm(row_b)
            res_row.append(dot(row_a, row_b) / (norm_a * norm_b))
        matrix.append(res_row)

    return matrix

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"
