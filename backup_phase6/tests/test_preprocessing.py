"""
Phase 2 Test Suite: Multilingual Dataset Collection and Preprocessing.
Tests script-preserving text cleaning, language detection, schema standardization,
deduplication, missing-value imputation, file export, and SQLite synchronization.
"""

import sys
import os
import unittest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_SAMPLE_DIR,
    DATA_PROCESSED_DIR,
    DATABASE_PATH,
    SUPPORTED_LANGUAGES,
)
from database.db_manager import DatabaseManager
from src.preprocessing.cleaner import MultilingualTextCleaner
from src.preprocessing.dataset_loader import NewsDatasetLoader, UNIFIED_ARTICLE_COLUMNS

class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for Phase 2 dataset preprocessing and text cleaner."""

    @classmethod
    def setUpClass(cls):
        cls.cleaner = MultilingualTextCleaner()
        cls.db = DatabaseManager(DATABASE_PATH)
        cls.loader = NewsDatasetLoader(db_manager=cls.db)

    def test_clean_text_removes_html_and_urls(self):
        """Verify HTML tags, URLs, and multiple spaces are removed."""
        raw = "<div>Breaking: <b>Major AI breakthrough</b> announced! Visit https://example.com/ai for details.</div>"
        cleaned = self.cleaner.clean_text(raw)
        self.assertNotIn("<div>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("breaking: major ai breakthrough announced! visit for details.", cleaned)

    def test_clean_text_preserves_hindi_script(self):
        """Verify Hindi Devanagari script is preserved intact."""
        hindi_raw = "इसरो ने <b>गगनयान</b> मिशन का सफल परीक्षण किया। https://isro.gov.in"
        cleaned = self.cleaner.clean_text(hindi_raw)
        self.assertIn("इसरो ने गगनयान मिशन का सफल परीक्षण किया।", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("https://", cleaned)

    def test_clean_text_preserves_assamese_script(self):
        """Verify Assamese Eastern Nagari script is preserved intact."""
        as_raw = "<p>কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যানত গঁড় সংৰক্ষণৰ সফল অভিযান।</p>"
        cleaned = self.cleaner.clean_text(as_raw)
        self.assertIn("কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যানত গঁড় সংৰক্ষণৰ সফল অভিযান।", cleaned)
        self.assertNotIn("<p>", cleaned)

    def test_language_detection(self):
        """Verify language script detection for English, Hindi, and Assamese."""
        en_text = "Generative AI and foundation models are transforming recommendation systems."
        hi_text = "भारतीय अंतरिक्ष अनुसंधान संगठन ने मानव अंतरिक्ष मिशन का सफल परीक्षण किया।"
        as_text = "কাজিৰঙা ৰাষ্ট্ৰীয় উদ্যানত এশিঙীয়া গঁড়ৰ সংখ্যা বৃদ্ধি পাইছে।"

        self.assertEqual(self.cleaner.detect_script_language(en_text), "en")
        self.assertEqual(self.cleaner.detect_script_language(hi_text), "hi")
        self.assertEqual(self.cleaner.detect_script_language(as_text), "as")

    def test_imputation_and_deduplication(self):
        """Verify missing abstracts are imputed from content and duplicates removed."""
        dirty_data = pd.DataFrame([
            {
                "article_id": "test_d1",
                "title": "Quantum Computing Scalability",
                "abstract": "",  # Missing abstract
                "content": "Full detailed article text explaining quantum error correction mechanisms in silicon qubits.",
                "language": "en",
                "category": "Technology"
            },
            {
                "article_id": "test_d2",
                "title": "Quantum Computing Scalability",  # Duplicate title & language
                "abstract": "Another copy of same article",
                "content": "Full content",
                "language": "en",
                "category": "Technology"
            },
            {
                "article_id": "test_d3",
                "title": "सौर ऊर्जा क्रांति",
                "abstract": "",
                "content": "भारत में स्वच्छ ऊर्जा उत्पादन ऐतिहासिक स्तर पर पहुंचा।",
                "language": "unknown",  # Missing language to be auto-detected
                "category": ""
            }
        ])

        processed = self.loader.preprocess_articles(dirty_data)

        # Should deduplicate test_d2
        self.assertEqual(len(processed), 2)
        
        # test_d1 abstract should be imputed from content
        row1 = processed[processed["article_id"] == "test_d1"].iloc[0]
        self.assertTrue(len(row1["abstract"]) > 0)
        self.assertIn("quantum error correction", row1["abstract"].lower())

        # test_d3 language should be detected as Hindi ('hi')
        row3 = processed[processed["article_id"] == "test_d3"].iloc[0]
        self.assertEqual(row3["language"], "hi")
        self.assertEqual(row3["category"], "general")

    def test_sample_news_and_export(self):
        """Verify sample multilingual news dataset loads, standardizes, and exports to data/processed/."""
        sample_path = DATA_SAMPLE_DIR / "sample_multilingual_news.csv"
        self.assertTrue(sample_path.exists(), "Sample multilingual news CSV must exist")

        df = pd.read_csv(sample_path, encoding="utf-8")
        processed_df = self.loader.preprocess_articles(df)

        # Check canonical columns
        for col in UNIFIED_ARTICLE_COLUMNS:
            self.assertIn(col, processed_df.columns)

        # Check languages present
        languages = set(processed_df["language"].unique())
        self.assertIn("en", languages)
        self.assertIn("hi", languages)
        self.assertIn("as", languages)

        # Test save
        csv_path, json_path = self.loader.save_processed(processed_df, file_prefix="test_multilingual_news")
        self.assertTrue(csv_path.exists())
        self.assertTrue(json_path.exists())

        # Test database sync
        synced = self.loader.sync_to_database(processed_df)
        self.assertGreater(synced, 0)

        # Query database to confirm articles exist
        db_arts = self.db.execute_query("SELECT COUNT(*) AS total FROM news_articles")
        self.assertGreaterEqual(db_arts[0]["total"], synced)

        # Test statistics generation
        stats = self.loader.generate_dataset_statistics(processed_df)
        self.assertIn("language_distribution", stats)
        self.assertEqual(stats["total_articles"], len(processed_df))
        self.assertIn("en", stats["language_distribution"])
        self.assertIn("hi", stats["language_distribution"])
        self.assertIn("as", stats["language_distribution"])

if __name__ == "__main__":
    unittest.main()

