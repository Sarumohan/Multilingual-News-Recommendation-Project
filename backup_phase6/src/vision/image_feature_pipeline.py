"""
Image Feature Pipeline and Multimodal Data Mapping Module.
Coordinates image downloading, visual feature extraction via ResNet-50,
article ID mapping, and storage in data/embeddings/.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from config.config import (
    DATA_PROCESSED_DIR,
    DATA_EMBEDDINGS_DIR,
    BASE_DIR
)
from src.vision.image_downloader import NewsImageDownloader, IMAGES_DIR
from src.vision.image_encoder import ResNetImageEncoder, RESNET_EMBEDDING_DIM
from src.vision.similarity import ImageSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("ImageFeaturePipeline")

IMAGE_EMBEDDINGS_FILE = DATA_EMBEDDINGS_DIR / "articles_image_embeddings.npz"
IMAGE_METADATA_FILE = DATA_EMBEDDINGS_DIR / "articles_image_metadata.json"

class ImageFeaturePipeline:
    """End-to-end pipeline for image acquisition, feature extraction, and multimodal mapping."""

    def __init__(
        self,
        downloader: Optional[NewsImageDownloader] = None,
        encoder: Optional[ResNetImageEncoder] = None
    ):
        self.downloader = downloader or NewsImageDownloader()
        self.encoder = encoder or ResNetImageEncoder()
        self.embeddings: Optional[np.ndarray] = None
        self.article_ids: List[str] = []
        self.has_image_mask: Optional[np.ndarray] = None
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def run_pipeline(
        self,
        articles_csv: Optional[Path] = None,
        auto_download: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full image feature extraction and persistence workflow.
        """
        csv_path = articles_csv or (DATA_PROCESSED_DIR / "multilingual_news_cleaned.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed news CSV not found at {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} articles for visual processing.")

        # Step 1: Check / Download images
        if auto_download:
            manifest = self.downloader.download_dataset_images(csv_path)
            manifest_lookup = {m["article_id"]: m for m in manifest}
        else:
            manifest_lookup = {}
            for _, row in df.iterrows():
                aid = str(row["article_id"])
                p = IMAGES_DIR / f"{aid}.jpg"
                is_valid, status = self.downloader.validate_image(p) if p.exists() else (False, "missing")
                manifest_lookup[aid] = {
                    "article_id": aid,
                    "language": str(row.get("language", "")),
                    "category": str(row.get("category", "")),
                    "image_url": str(row.get("image_url", "")),
                    "is_available": is_valid,
                    "local_path": str(p) if is_valid else None,
                    "status": status
                }

        # Step 2: Extract features per article
        N = len(df)
        D = self.encoder.embedding_dim
        embeddings = np.zeros((N, D), dtype=np.float32)
        has_image = np.zeros(N, dtype=bool)
        article_ids = df["article_id"].tolist()
        metadata_dict = {}

        start_time = time.time()
        success_count = 0
        failed_count = 0

        logger.info(f"Extracting visual features with ResNet-50 for {N} articles...")
        for i, row in df.iterrows():
            aid = str(row["article_id"])
            m = manifest_lookup.get(aid, {})
            is_avail = m.get("is_available", False)
            local_path = m.get("local_path")

            if is_avail and local_path:
                ok, vec, status_msg = self.encoder.extract_features_safe(local_path)
                if ok and vec is not None:
                    embeddings[i] = vec
                    has_image[i] = True
                    success_count += 1
                    status_final = "success"
                else:
                    embeddings[i] = np.zeros(D, dtype=np.float32)
                    has_image[i] = False
                    failed_count += 1
                    status_final = f"extraction_failed_{status_msg}"
            else:
                embeddings[i] = np.zeros(D, dtype=np.float32)
                has_image[i] = False
                failed_count += 1
                status_final = m.get("status", "missing_image")

            metadata_dict[aid] = {
                "article_id": aid,
                "language": str(row["language"]),
                "category": str(row["category"]),
                "title": str(row["title"]),
                "image_url": str(row.get("image_url", "")),
                "local_path": str(local_path) if is_avail else None,
                "is_available": bool(has_image[i]),
                "status": status_final,
                "model_name": "resnet50",
                "embedding_dim": D,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

        elapsed_sec = time.time() - start_time
        logger.info(f"Extracted visual features: {success_count} succeeded, {failed_count} failed in {elapsed_sec:.2f}s.")

        # Step 3: Persist embeddings and metadata
        self.save_embeddings(embeddings, article_ids, has_image, metadata_dict)

        self.embeddings = embeddings
        self.article_ids = article_ids
        self.has_image_mask = has_image
        self.metadata = metadata_dict

        stats = {
            "total_articles": N,
            "images_available": success_count,
            "images_failed": failed_count,
            "embedding_dim": D,
            "model_name": "resnet50",
            "device": self.encoder.device,
            "elapsed_seconds": round(elapsed_sec, 2),
            "npz_path": str(IMAGE_EMBEDDINGS_FILE),
            "metadata_path": str(IMAGE_METADATA_FILE)
        }

        return stats

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        article_ids: List[str],
        has_image: np.ndarray,
        metadata: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save image embeddings to compressed NPZ and metadata to JSON."""
        DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            IMAGE_EMBEDDINGS_FILE,
            embeddings=embeddings,
            article_ids=np.array(article_ids, dtype=object),
            has_image=has_image
        )
        logger.info(f"Saved image embeddings to {IMAGE_EMBEDDINGS_FILE}")

        with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved image metadata to {IMAGE_METADATA_FILE}")

    def load_embeddings(self) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Dict[str, Any]]]:
        """Load persisted image embeddings and metadata from disk."""
        if not IMAGE_EMBEDDINGS_FILE.exists() or not IMAGE_METADATA_FILE.exists():
            raise FileNotFoundError("Image embeddings or metadata file not found.")

        data = np.load(IMAGE_EMBEDDINGS_FILE, allow_pickle=True)
        embeddings = data["embeddings"]
        article_ids = list(data["article_ids"])
        has_image = data["has_image"]

        with open(IMAGE_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.embeddings = embeddings
        self.article_ids = article_ids
        self.has_image_mask = has_image
        self.metadata = metadata

        return embeddings, article_ids, has_image, metadata

    def get_article_image_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Retrieve 2048-d visual feature vector if article image is available."""
        if self.embeddings is None:
            self.load_embeddings()

        try:
            idx = self.article_ids.index(article_id)
            if self.has_image_mask[idx]:
                return self.embeddings[idx]
            return None
        except ValueError:
            return None

    def find_similar_images(
        self,
        article_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find top-K visually closest articles that have valid image embeddings.
        """
        if self.embeddings is None:
            self.load_embeddings()

        query_vec = self.get_article_image_embedding(article_id)
        if query_vec is None:
            return []

        # Only search within candidates that actually have valid images
        valid_indices = [
            i for i, aid in enumerate(self.article_ids)
            if self.has_image_mask[i] and aid != article_id
        ]

        if not valid_indices:
            return []

        candidate_embeddings = self.embeddings[valid_indices]
        candidate_ids = [self.article_ids[i] for i in valid_indices]

        top_results = ImageSimilarityEngine.find_top_k(
            query_embedding=query_vec,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            top_k=top_k
        )

        results = []
        for aid, score in top_results:
            meta = self.metadata.get(aid, {})
            results.append({
                "article_id": aid,
                "title": meta.get("title", ""),
                "language": meta.get("language", ""),
                "category": meta.get("category", ""),
                "similarity_score": round(score, 4)
            })

        return results

