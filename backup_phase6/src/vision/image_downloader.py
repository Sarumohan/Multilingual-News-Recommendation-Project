"""
Image Downloader and Validator Module for Multimodal News Processing.
Handles downloading, caching, PIL validation, and tracking image availability.
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from typing import Dict, List, Optional, Tuple, Any

from config.config import BASE_DIR, DATA_PROCESSED_DIR
from src.utils.helpers import setup_logger

logger = setup_logger("ImageDownloader")

IMAGES_DIR = BASE_DIR / "data" / "images"

class NewsImageDownloader:
    """Manages downloading, validating, and caching news article images."""

    def __init__(self, output_dir: Optional[Path] = None, timeout: int = 10):
        self.output_dir = output_dir or IMAGES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    def fetch_image_for_article(
        self,
        article_id: str,
        image_url: str
    ) -> Tuple[bool, Optional[Path], str]:
        """
        Download and validate image for a given article ID.
        Returns: (success_bool, local_path_or_None, status_message)
        """
        if not image_url or not isinstance(image_url, str) or not image_url.strip():
            return False, None, "missing_url"

        url = image_url.strip()
        local_path = self.output_dir / f"{article_id}.jpg"

        # Check local cache first to avoid repeated downloads
        if local_path.exists():
            is_valid, err = self.validate_image(local_path)
            if is_valid:
                return True, local_path, "cached"
            else:
                try:
                    local_path.unlink()
                except Exception:
                    pass

        # Format Unsplash URLs with crop/dimension parameters if needed
        fetch_url = url
        if "unsplash.com" in url and "?" not in url:
            fetch_url = f"{url}?auto=format&fit=crop&w=400&q=80"

        try:
            req = urllib.request.Request(fetch_url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status != 200:
                    return False, None, f"http_status_{response.status}"
                data = response.read()

            with open(local_path, "wb") as f:
                f.write(data)

            # Validate image with PIL
            is_valid, err = self.validate_image(local_path)
            if is_valid:
                return True, local_path, "downloaded"
            else:
                if local_path.exists():
                    local_path.unlink()
                return False, None, f"invalid_image_{err}"

        except urllib.error.HTTPError as e:
            return False, None, f"http_error_{e.code}"
        except urllib.error.URLError as e:
            return False, None, f"network_error_{e.reason}"
        except Exception as e:
            return False, None, f"error_{str(e)[:50]}"

    def validate_image(self, file_path: Path) -> Tuple[bool, str]:
        """Verify that file exists and can be decoded by PIL without corruption."""
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False, "file_empty_or_missing"

        try:
            with Image.open(file_path) as img:
                img.verify()
            # Double check re-opening to ensure decoding succeeds
            with Image.open(file_path) as img:
                img.load()
            return True, "valid"
        except (UnidentifiedImageError, OSError, Exception) as e:
            return False, str(e)

    def download_dataset_images(
        self,
        articles_csv: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Iterate over news articles, download missing images, and return status manifest.
        """
        csv_path = articles_csv or (DATA_PROCESSED_DIR / "multilingual_news_cleaned.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed news CSV not found at {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8")
        manifest = []

        logger.info(f"Checking images for {len(df)} articles...")
        for _, row in df.iterrows():
            aid = str(row["article_id"])
            url = str(row.get("image_url", ""))
            lang = str(row.get("language", ""))
            cat = str(row.get("category", ""))

            success, path, status = self.fetch_image_for_article(aid, url)
            manifest.append({
                "article_id": aid,
                "language": lang,
                "category": cat,
                "image_url": url,
                "is_available": success,
                "local_path": str(path) if path else None,
                "status": status
            })

        downloaded = sum(1 for m in manifest if m["is_available"])
        failed = len(manifest) - downloaded
        logger.info(f"Image manifest: {downloaded} available, {failed} unavailable/failed.")
        return manifest

