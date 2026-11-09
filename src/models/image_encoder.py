"""Pretrained vision encoder re-export for modular compatibility."""
from src.vision.image_encoder import ResNetImageEncoder, RESNET_EMBEDDING_DIM

__all__ = ["ResNetImageEncoder", "RESNET_EMBEDDING_DIM"]

