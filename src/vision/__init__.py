"""Vision and multimodal image processing package."""
from .image_downloader import NewsImageDownloader, IMAGES_DIR
from .image_encoder import ResNetImageEncoder, RESNET_EMBEDDING_DIM
from .similarity import ImageSimilarityEngine
from .image_feature_pipeline import ImageFeaturePipeline

__all__ = [
    "NewsImageDownloader",
    "ResNetImageEncoder",
    "RESNET_EMBEDDING_DIM",
    "ImageSimilarityEngine",
    "ImageFeaturePipeline",
    "IMAGES_DIR",
]

