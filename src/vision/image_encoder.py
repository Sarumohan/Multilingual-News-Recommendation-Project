"""
ResNet-50 Vision Feature Extractor for Multimodal News Recommendation.
Extracts 2048-dimensional dense visual representations from article images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple

from src.utils.helpers import setup_logger

logger = setup_logger("ResNetImageEncoder")

RESNET_EMBEDDING_DIM = 2048

class ResNetImageEncoder:
    """Extracts 2048-dimensional dense visual representations using pretrained ResNet-50."""

    def __init__(
        self,
        device: Optional[str] = None,
        use_default_weights: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = RESNET_EMBEDDING_DIM

        logger.info(f"Initializing ResNet-50 Image Encoder on device '{self.device}'...")

        if use_default_weights:
            weights = ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
        else:
            base_model = models.resnet50(weights=None)

        # Remove the final classification FC layer to extract 2048-d pooled features
        # Structure: conv1 -> bn1 -> relu -> maxpool -> layer1..4 -> avgpool -> flatten
        modules = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Standard ImageNet preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info(f"ResNet-50 initialized successfully. Feature dimension: {self.embedding_dim}")

    def preprocess_image(self, image_input: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Load, validate, convert to RGB, and apply torchvision transforms.
        Returns tensor of shape (3, 224, 224).
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found at {path}")
            img = Image.open(path)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Ensure 3-channel RGB representation (converts RGBA, Grayscale 'L', CMYK, Palette 'P')
        if img.mode != "RGB":
            img = img.convert("RGB")

        tensor = self.transform(img)
        return tensor

    def extract_features(
        self,
        image_inputs: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        batch_size: int = 16,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Extract 2048-dimensional dense visual embeddings for an image or batch of images.
        """
        is_single = not isinstance(image_inputs, list)
        inputs_list = [image_inputs] if is_single else image_inputs

        all_features = []

        for i in range(0, len(inputs_list), batch_size):
            batch = inputs_list[i:i + batch_size]
            batch_tensors = []

            for item in batch:
                t = self.preprocess_image(item)
                batch_tensors.append(t)

            tensor_stack = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                feats = self.feature_extractor(tensor_stack)
                feats = torch.flatten(feats, 1)  # (batch_size, 2048)

                if normalize_embeddings:
                    feats = F.normalize(feats, p=2, dim=1)

                all_features.append(feats.cpu().numpy())

        final_features = np.vstack(all_features).astype(np.float32)
        return final_features[0] if is_single else final_features

    def extract_features_safe(
        self,
        file_path: Optional[Union[str, Path]]
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Safely extract features from a file path.
        Handles missing files and corrupted images without raising exceptions.
        Returns (success_bool, embedding_or_None, status_msg).
        """
        if file_path is None:
            return False, None, "missing_path"

        path = Path(file_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, None, "file_missing_or_empty"

        try:
            emb = self.extract_features(path, normalize_embeddings=True)
            return True, emb, "extracted"
        except (UnidentifiedImageError, OSError) as e:
            return False, None, f"corrupted_image_{e}"
        except Exception as e:
            return False, None, f"extraction_error_{e}"

