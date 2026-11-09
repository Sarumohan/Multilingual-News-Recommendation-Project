"""
Multimodal Feature Fusion package.
Provides fusion engines, cross-modality similarity comparison, and pipeline orchestration.
"""
from .multimodal_fusion import MultimodalFusionEngine
from .similarity import MultimodalSimilarityEngine

__all__ = ["MultimodalFusionEngine", "MultimodalSimilarityEngine"]

