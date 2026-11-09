"""
Unit Tests for Multimodal Feature Fusion Engine (Revised Phase 4).
Tests compatibility, normalization, weighted fusion, missing-modality handling,
IndicBERT dual fusion, cosine similarity computations, and artifact persistence.
"""

import unittest
import json
import numpy as np
from pathlib import Path

from config.config import DATA_EMBEDDINGS_DIR, DEFAULT_FUSION_WEIGHTS
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.similarity import MultimodalSimilarityEngine

class TestMultimodalFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alpha = DEFAULT_FUSION_WEIGHTS.get("text", 0.7)
        cls.beta = DEFAULT_FUSION_WEIGHTS.get("image", 0.3)
        cls.engine = MultimodalFusionEngine(alpha=cls.alpha, beta=cls.beta)
        cls.npz_path = DATA_EMBEDDINGS_DIR / "articles_multimodal_embeddings.npz"
        cls.meta_path = DATA_EMBEDDINGS_DIR / "articles_multimodal_metadata.json"

    def test_fusion_weight_normalization(self):
        """Test that fusion weights are strictly normalized to sum to 1.0."""
        engine = MultimodalFusionEngine(alpha=0.7, beta=0.3)
        self.assertAlmostEqual(engine.alpha + engine.beta, 1.0, places=6)
        self.assertAlmostEqual(engine.alpha, 0.7, places=6)
        self.assertAlmostEqual(engine.beta, 0.3, places=6)

        # Unnormalized inputs
        engine_unnorm = MultimodalFusionEngine(alpha=14.0, beta=6.0)
        self.assertAlmostEqual(engine_unnorm.alpha + engine_unnorm.beta, 1.0, places=6)
        self.assertAlmostEqual(engine_unnorm.alpha, 0.7, places=6)
        self.assertAlmostEqual(engine_unnorm.beta, 0.3, places=6)

    def test_weighted_concat_unit_norm(self):
        """Test that weighted concatenation yields an exact unit-norm fused vector."""
        t_vec = np.random.randn(384).astype(np.float32)
        v_vec = np.random.randn(2048).astype(np.float32)

        fused, mode = self.engine.fuse_article_vectors(
            text_vec=t_vec,
            image_vec=v_vec,
            has_image=True,
            strategy="weighted_concat"
        )
        self.assertEqual(mode, "multimodal")
        self.assertEqual(fused.shape, (2432,))
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)

    def test_weighted_fusion_inner_product_linearity(self):
        """
        Test that dot product of weighted concatenated vectors matches
        alpha * text_similarity + beta * image_similarity.
        """
        t_a = np.random.randn(384).astype(np.float32)
        t_b = np.random.randn(384).astype(np.float32)
        v_a = np.random.randn(2048).astype(np.float32)
        v_b = np.random.randn(2048).astype(np.float32)

        fused_a, _ = self.engine.fuse_article_vectors(t_a, v_a, has_image=True, strategy="weighted_concat")
        fused_b, _ = self.engine.fuse_article_vectors(t_b, v_b, has_image=True, strategy="weighted_concat")

        sim_t = MultimodalSimilarityEngine.compute_similarity(t_a, t_b)
        sim_v = MultimodalSimilarityEngine.compute_similarity(v_a, v_b)
        sim_fused = MultimodalSimilarityEngine.compute_similarity(fused_a, fused_b)

        expected_sim = self.alpha * sim_t + self.beta * sim_v
        self.assertAlmostEqual(sim_fused, expected_sim, places=5)

    def test_missing_modality_fallback_concat(self):
        """Test fallback when image is missing: returns unit norm with zero visual padding."""
        t_vec = np.random.randn(384).astype(np.float32)
        v_vec = np.zeros(2048, dtype=np.float32)

        fused, mode = self.engine.fuse_article_vectors(
            text_vec=t_vec,
            image_vec=v_vec,
            has_image=False,
            strategy="weighted_concat"
        )
        self.assertEqual(mode, "text_only")
        self.assertEqual(fused.shape, (2432,))
        # Visual portion should be strictly zero
        self.assertTrue(np.all(fused[384:] == 0.0))
        # Overall vector should still have unit norm
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)

    def test_missing_modality_fallback_projected(self):
        """Test projected fallback when image is missing: returns unit norm text vector."""
        t_vec = np.random.randn(384).astype(np.float32)

        fused, mode = self.engine.fuse_article_vectors(
            text_vec=t_vec,
            image_vec=None,
            has_image=False,
            strategy="projected"
        )
        self.assertEqual(mode, "text_only")
        self.assertEqual(fused.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)

    def test_orthogonal_projection_determinism(self):
        """Test that orthogonal projection is deterministic across identical calls."""
        proj_1 = self.engine.get_orthogonal_projection_matrix(2048, 384)
        proj_2 = self.engine.get_orthogonal_projection_matrix(2048, 384)
        np.testing.assert_array_equal(proj_1, proj_2)
        self.assertEqual(proj_1.shape, (2048, 384))

        # Check semi-orthogonality: Q^T * Q == I_384
        identity_approx = np.dot(proj_1.T, proj_1)
        np.testing.assert_allclose(identity_approx, np.eye(384), atol=1e-5)

    def test_indicbert_separate_fusion(self):
        """Test that IndicBERT (768-d) can be fused with ResNet (2048-d) producing 2816-d representation."""
        indic_vec = np.random.randn(768).astype(np.float32)
        v_vec = np.random.randn(2048).astype(np.float32)

        fused, mode = self.engine.fuse_article_vectors(
            text_vec=indic_vec,
            image_vec=v_vec,
            has_image=True,
            strategy="weighted_concat"
        )
        self.assertEqual(mode, "multimodal")
        self.assertEqual(fused.shape, (2816,))
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)

    def test_persisted_multimodal_embeddings(self):
        """Verify the integrity of persisted articles_multimodal_embeddings.npz."""
        self.assertTrue(self.npz_path.exists(), f"Missing file: {self.npz_path}")
        data = np.load(self.npz_path, allow_pickle=True)

        self.assertIn("fused_concat", data)
        self.assertIn("fused_projected", data)
        self.assertIn("fused_indicbert", data)
        self.assertIn("article_ids", data)
        self.assertIn("has_image", data)

        fused_concat = data["fused_concat"]
        fused_projected = data["fused_projected"]
        fused_indicbert = data["fused_indicbert"]
        article_ids = list(data["article_ids"])
        has_image = data["has_image"]

        self.assertEqual(len(article_ids), 16)
        self.assertEqual(fused_concat.shape, (16, 2432))
        self.assertEqual(fused_projected.shape, (16, 384))
        self.assertEqual(fused_indicbert.shape, (16, 2816))

        # Verify no NaN or Inf
        self.assertFalse(np.isnan(fused_concat).any())
        self.assertFalse(np.isinf(fused_concat).any())
        self.assertFalse(np.isnan(fused_projected).any())
        self.assertFalse(np.isnan(fused_indicbert).any())

        # Verify image availability count (13 available, 3 missing)
        self.assertEqual(int(np.sum(has_image)), 13)
        self.assertEqual(int(len(has_image) - np.sum(has_image)), 3)

    def test_persisted_multimodal_metadata(self):
        """Verify the integrity of articles_multimodal_metadata.json."""
        self.assertTrue(self.meta_path.exists(), f"Missing file: {self.meta_path}")
        with open(self.meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertEqual(len(metadata), 16)
        
        # Check specific articles with missing images
        missing_ids = ["art_en_002", "art_hi_001", "art_hi_005"]
        for aid in missing_ids:
            self.assertIn(aid, metadata)
            self.assertFalse(metadata[aid]["has_image"])
            self.assertEqual(metadata[aid]["fusion_mode"], "text_only")

        # Check multimodal article
        self.assertTrue(metadata["art_en_001"]["has_image"])
        self.assertEqual(metadata["art_en_001"]["fusion_mode"], "multimodal")
        self.assertEqual(metadata["art_en_001"]["dimensions"]["fused_concat"], 2432)

    def test_modality_comparison_engine(self):
        """Test MultimodalSimilarityEngine.compare_article_pair_modalities functionality."""
        t_a = np.array([1.0, 0.0], dtype=np.float32)
        t_b = np.array([1.0, 0.0], dtype=np.float32)
        v_a = np.array([0.0, 1.0], dtype=np.float32)
        v_b = np.array([0.0, 1.0], dtype=np.float32)

        fused_a, _ = self.engine.fuse_article_vectors(t_a, v_a, has_image=True, strategy="weighted_concat")
        fused_b, _ = self.engine.fuse_article_vectors(t_b, v_b, has_image=True, strategy="weighted_concat")

        comp = MultimodalSimilarityEngine.compare_article_pair_modalities(
            aid_a="test_1",
            aid_b="test_2",
            text_emb_a=t_a,
            text_emb_b=t_b,
            img_emb_a=v_a,
            img_emb_b=v_b,
            has_img_a=True,
            has_img_b=True,
            fused_emb_a=fused_a,
            fused_emb_b=fused_b,
            alpha=0.7,
            beta=0.3
        )
        self.assertEqual(comp["text_similarity"], 1.0)
        self.assertEqual(comp["image_similarity"], 1.0)
        self.assertEqual(comp["fused_similarity"], 1.0)
        self.assertEqual(comp["image_status"], "both_available")


if __name__ == "__main__":
    unittest.main()

