"""
Phase 4 Test Suite: Multimodal Image Feature Extraction & Visual Similarity.
Tests image loading, PIL preprocessing, RGB conversion, ResNet-50 feature extraction,
dimension integrity, missing/corrupted image resilience, cosine similarity,
persistence, and article ID mapping.
"""

import sys
import tempfile
import unittest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_EMBEDDINGS_DIR
from src.vision.image_encoder import ResNetImageEncoder, RESNET_EMBEDDING_DIM
from src.vision.image_downloader import NewsImageDownloader, IMAGES_DIR
from src.vision.image_feature_pipeline import ImageFeaturePipeline
from src.vision.similarity import ImageSimilarityEngine

class TestImageFeatureExtraction(unittest.TestCase):
    """Unit and integration tests for Phase 4 visual representation pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.encoder = ResNetImageEncoder()
        cls.downloader = NewsImageDownloader()
        cls.pipeline = ImageFeaturePipeline(downloader=cls.downloader, encoder=cls.encoder)

    def test_image_loading_and_preprocessing(self):
        """1 & 2: Test image loading and preprocessing transform to (3, 224, 224)."""
        test_img = Image.new("RGB", (300, 400), color=(128, 64, 32))
        tensor = self.encoder.preprocess_image(test_img)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32 if "torch" in sys.modules else tensor.dtype)

    def test_rgb_conversion_from_various_modes(self):
        """3: Test conversion of RGBA and Grayscale (L) images to standard 3-channel RGB."""
        # RGBA image with alpha channel
        rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        tensor_rgba = self.encoder.preprocess_image(rgba_img)
        self.assertEqual(tensor_rgba.shape[0], 3)

        # Grayscale image
        gray_img = Image.new("L", (100, 100), color=128)
        tensor_gray = self.encoder.preprocess_image(gray_img)
        self.assertEqual(tensor_gray.shape[0], 3)

    def test_image_embedding_generation_and_dimensions(self):
        """4 & 5: Test ResNet-50 feature extraction yields normalized 2048-dim vectors."""
        test_img = Image.new("RGB", (224, 224), color=(0, 255, 0))
        vec = self.encoder.extract_features(test_img, normalize_embeddings=True)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (RESNET_EMBEDDING_DIM,))
        self.assertEqual(vec.shape[0], 2048)
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)

    def test_missing_image_handling(self):
        """6: Test that non-existent image paths are caught gracefully without crashing."""
        non_existent_path = Path("data/images/non_existent_art_999.jpg")
        ok, vec, status = self.encoder.extract_features_safe(non_existent_path)

        self.assertFalse(ok)
        self.assertIsNone(vec)
        self.assertEqual(status, "file_missing_or_empty")

    def test_corrupted_image_handling(self):
        """7: Test that corrupted/truncated image files are rejected safely."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"NOT_A_REAL_IMAGE_DATA_CORRUPTED_BYTES_123456")
            corrupt_path = Path(f.name)

        try:
            ok, vec, status = self.encoder.extract_features_safe(corrupt_path)
            self.assertFalse(ok)
            self.assertIsNone(vec)
            self.assertTrue("corrupted" in status or "error" in status)
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    def test_cosine_similarity_calculation(self):
        """8: Test cosine similarity properties on visual features."""
        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img2 = Image.new("RGB", (224, 224), color=(255, 10, 10))
        img3 = Image.new("RGB", (224, 224), color=(0, 0, 255))

        vec1 = self.encoder.extract_features(img1)
        vec2 = self.encoder.extract_features(img2)
        vec3 = self.encoder.extract_features(img3)

        sim_self = ImageSimilarityEngine.compute_similarity(vec1, vec1)
        sim_near = ImageSimilarityEngine.compute_similarity(vec1, vec2)
        sim_far = ImageSimilarityEngine.compute_similarity(vec1, vec3)

        self.assertAlmostEqual(sim_self, 1.0, places=4)
        self.assertGreater(sim_near, sim_far)

    def test_embedding_persistence_and_retrieval(self):
        """9: Verify saved articles_image_embeddings.npz can be loaded accurately."""
        npz_file = DATA_EMBEDDINGS_DIR / "articles_image_embeddings.npz"
        json_file = DATA_EMBEDDINGS_DIR / "articles_image_metadata.json"

        self.assertTrue(npz_file.exists(), "Image embeddings NPZ file must exist")
        self.assertTrue(json_file.exists(), "Image metadata JSON file must exist")

        embs, ids, has_img, meta = self.pipeline.load_embeddings()
        self.assertEqual(len(ids), 16)
        self.assertEqual(embs.shape, (16, 2048))
        self.assertEqual(len(has_img), 16)
        self.assertEqual(sum(has_img), 13)  # 13 available, 3 failed

    def test_article_id_mapping(self):
        """10: Verify article ID mapping preserves language, category, and availability flags."""
        _, ids, has_img, meta = self.pipeline.load_embeddings()

        for idx, aid in enumerate(ids):
            self.assertIn(aid, meta)
            item = meta[aid]
            self.assertEqual(item["article_id"], aid)
            self.assertIn(item["language"], ["en", "hi", "as"])
            self.assertEqual(item["embedding_dim"], 2048)
            self.assertEqual(item["is_available"], bool(has_img[idx]))

        # Specifically check a known successful and a known missing article
        self.assertTrue(meta["art_en_001"]["is_available"])
        self.assertFalse(meta["art_en_002"]["is_available"])  # 404 missing URL
        self.assertFalse(meta["art_hi_001"]["is_available"])  # 404 missing URL

if __name__ == "__main__":
    unittest.main()
