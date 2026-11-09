"""
Phase 3 Test Suite: Multilingual Text Representation & Embeddings.
Tests model loading, language-specific embedding generation (EN, HI, AS),
dimension validity, cosine similarity, batch processing, persistence, and edge-case handling.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR, DEFAULT_TEXT_MODEL
from src.embeddings.multilingual_encoder import MultilingualTextEncoder
from src.embeddings.embedding_generator import ArticleEmbeddingGenerator
from src.embeddings.similarity import EmbeddingSimilarityEngine

class TestMultilingualEmbeddings(unittest.TestCase):
    """Test suite for Phase 3 multilingual sentence representations."""

    @classmethod
    def setUpClass(cls):
        cls.encoder = MultilingualTextEncoder(model_name=DEFAULT_TEXT_MODEL)
        cls.generator = ArticleEmbeddingGenerator(encoder=cls.encoder)

    def test_model_loading_and_dimensions(self):
        """Verify model loads with correct embedding dimension (384 for MiniLM-L12)."""
        self.assertIsNotNone(self.encoder.model)
        self.assertEqual(self.encoder.embedding_dim, 384)

    def test_english_embedding_generation(self):
        """Verify English text generates non-zero, finite 384-dimensional embedding."""
        text = "Deep neural networks for multimodal and multilingual news recommendation."
        emb = self.encoder.encode(text)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (384,))
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertGreater(float(np.linalg.norm(emb)), 0.99)  # Normalized to unit norm

    def test_hindi_embedding_generation(self):
        """Verify Hindi text generates valid 384-dimensional embedding."""
        text = "भारतीय अंतरिक्ष अनुसंधान संगठन ने मानव अंतरिक्ष मिशन का सफल परीक्षण पूरा किया।"
        emb = self.encoder.encode(text)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (384,))
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertGreater(float(np.linalg.norm(emb)), 0.99)

    def test_assamese_embedding_generation(self):
        """Verify Assamese text generates valid 384-dimensional embedding."""
        text = "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যানত এশিঙীয়া গঁড়ৰ সংখ্যা বৃদ্ধি পাইছে।"
        emb = self.encoder.encode(text)
        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (384,))
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertGreater(float(np.linalg.norm(emb)), 0.99)

    def test_cosine_similarity_properties(self):
        """Verify cosine similarity properties: self-similarity is 1.0, dissimilar is lower."""
        emb1 = self.encoder.encode("Artificial intelligence and deep learning models")
        emb2 = self.encoder.encode("Machine learning and neural network architectures")
        emb3 = self.encoder.encode("A delicious recipe for homemade tomato soup")

        sim_self = EmbeddingSimilarityEngine.compute_similarity(emb1, emb1)
        sim_similar = EmbeddingSimilarityEngine.compute_similarity(emb1, emb2)
        sim_dissimilar = EmbeddingSimilarityEngine.compute_similarity(emb1, emb3)

        self.assertAlmostEqual(sim_self, 1.0, places=4)
        self.assertGreater(sim_similar, sim_dissimilar)
        self.assertGreater(sim_similar, 0.5)

    def test_batch_processing_consistency(self):
        """Verify batch encoding returns matrix matching individual vector encodings."""
        sentences = [
            "Global renewable energy capacity expanded rapidly.",
            "डिजिटल भुगतान और यूपीआई ने तोड़े रिकॉर्ड।",
            "অসমত তথ্য প্ৰযুক্তিৰ বিকাশ ঘটিছে।"
        ]
        batch_embs = self.encoder.encode(sentences, batch_size=3)
        self.assertEqual(batch_embs.shape, (3, 384))

        for i, s in enumerate(sentences):
            single_emb = self.encoder.encode(s)
            diff = np.max(np.abs(batch_embs[i] - single_emb))
            self.assertLess(float(diff), 1e-4)

    def test_empty_and_whitespace_text_handling(self):
        """Verify empty and whitespace strings are handled safely without crashing."""
        emb_empty = self.encoder.encode("")
        emb_spaces = self.encoder.encode("   \n\t  ")

        self.assertEqual(emb_empty.shape, (384,))
        self.assertEqual(emb_spaces.shape, (384,))
        self.assertTrue(np.all(np.isfinite(emb_empty)))
        self.assertTrue(np.all(np.isfinite(emb_spaces)))

    def test_embedding_persistence_and_retrieval(self):
        """Verify generate_and_save_embeddings saves and load_embeddings retrieves identical arrays."""
        embs, ids, stats = self.generator.generate_and_save_embeddings(show_progress=False)

        loaded_embs, loaded_ids, loaded_meta = self.generator.load_embeddings()

        self.assertEqual(len(ids), len(loaded_ids))
        self.assertEqual(embs.shape, loaded_embs.shape)
        self.assertTrue(np.allclose(embs, loaded_embs, atol=1e-5))

        # Check retrieval by article ID
        first_id = ids[0]
        single_vec = self.generator.get_article_embedding(first_id)
        self.assertIsNotNone(single_vec)
        self.assertTrue(np.allclose(single_vec, embs[0], atol=1e-5))

    def test_find_similar_articles_cross_lingual(self):
        """Verify find_similar_articles can filter by target language."""
        self.generator.load_embeddings()
        
        # Query an English AI article for Hindi similar articles
        hindi_matches = self.generator.find_similar_articles("art_en_001", top_k=2, target_language="hi")
        self.assertGreater(len(hindi_matches), 0)
        for match in hindi_matches:
            self.assertEqual(match["language"], "hi")
            self.assertIn("similarity_score", match)

if __name__ == "__main__":
    unittest.main()

