"""
Multimodal Fusion model wrapper.
Re-exports MultimodalFusionEngine and MultimodalSimilarityEngine from src.fusion for model package interface.
"""
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.similarity import MultimodalSimilarityEngine

__all__ = ["MultimodalFusionEngine", "MultimodalSimilarityEngine"]

