"""
Test suite for AI4Bharat IndicBERT Assamese & Indic Multilingual Encoder.
Tests model loading, attention-mask mean pooling, Assamese character coverage,
and separate embedding storage.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_EMBEDDINGS_DIR
from src.embeddings.indicbert_encoder import IndicBERTEncoder, INDICBERT_MODEL_NAME

class TestIndicBERTEncoder(unittest.TestCase):
    """Unit tests for IndicBERT Assamese representation module."""

    @classmethod
    def setUpClass(cls):
        cls.encoder = IndicBERTEncoder(model_name=INDICBERT_MODEL_NAME)

    def test_model_loading_and_hidden_dim(self):
        """Verify IndicBERT loads with 768-dimensional hidden states."""
        self.assertIsNotNone(self.encoder.model)
        self.assertEqual(self.encoder.embedding_dim, 768)

    def test_assamese_tokenization_and_embedding(self):
        """Verify Assamese Eastern Nagari text is tokenized and encoded properly."""
        as_text = "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যানত এশিঙীয়া গঁড়ৰ সংৰক্ষণ সুনিশ্চিত কৰা হৈছে।"
        emb = self.encoder.encode(as_text)

        self.assertIsInstance(emb, np.ndarray)
        self.assertEqual(emb.shape, (768,))
        self.assertTrue(np.all(np.isfinite(emb)))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_hindi_and_english_encoding(self):
        """Verify IndicBERT encodes Hindi and English without errors."""
        hi_text = "भारतीय अंतरिक्ष अनुसंधान संगठन का गगनयान मिशन।"
        en_text = "Space exploration and deep galaxy observation."

        emb_hi = self.encoder.encode(hi_text)
        emb_en = self.encoder.encode(en_text)

        self.assertEqual(emb_hi.shape, (768,))
        self.assertEqual(emb_en.shape, (768,))
        self.assertTrue(np.all(np.isfinite(emb_hi)))
        self.assertTrue(np.all(np.isfinite(emb_en)))

    def test_empty_string_handling(self):
        """Verify empty strings produce safe zero vectors without crashing."""
        emb_empty = self.encoder.encode("")
        self.assertEqual(emb_empty.shape, (768,))
        self.assertTrue(np.all(np.isfinite(emb_empty)))

    def test_separate_indicbert_storage_exists(self):
        """Verify IndicBERT output files were saved separately from MiniLM files."""
        indic_npz = DATA_EMBEDDINGS_DIR / "articles_indicbert_embeddings.npz"
        indic_json = DATA_EMBEDDINGS_DIR / "articles_indicbert_metadata.json"
        minilm_npz = DATA_EMBEDDINGS_DIR / "articles_text_embeddings.npz"

        self.assertTrue(indic_npz.exists(), "IndicBERT NPZ file must exist")
        self.assertTrue(indic_json.exists(), "IndicBERT metadata JSON must exist")
        self.assertTrue(minilm_npz.exists(), "Original MiniLM NPZ file must be preserved")

        data = np.load(indic_npz, allow_pickle=True)
        self.assertEqual(data["embeddings"].shape, (16, 768))

if __name__ == "__main__":
    unittest.main()

