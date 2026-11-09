"""
Phase 1 Setup Verification Test.
Verifies project directories, database initialization, schema integrity, and helper functions.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BASE_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EMBEDDINGS_DIR,
    DATA_SAMPLE_DIR,
    DATABASE_PATH,
    SUPPORTED_LANGUAGES,
)
from database.db_manager import DatabaseManager
from src.utils.helpers import setup_logger, cosine_similarity_matrix

class TestPhase1Setup(unittest.TestCase):
    """Test suite to ensure Phase 1 setup is complete and functional."""

    def test_directory_structure(self):
        """Check all essential directories exist."""
        for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR, DATA_SAMPLE_DIR]:
            self.assertTrue(directory.exists(), f"Directory missing: {directory}")

    def test_supported_languages(self):
        """Check that English, Hindi, and Assamese are configured."""
        self.assertIn("en", SUPPORTED_LANGUAGES)
        self.assertIn("hi", SUPPORTED_LANGUAGES)
        self.assertIn("as", SUPPORTED_LANGUAGES)

    def test_database_initialization(self):
        """Verify SQLite tables are initialized properly using schema.sql."""
        db = DatabaseManager(DATABASE_PATH)
        conn = db.get_connection()
        cursor = conn.cursor()

        # Query all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        expected_tables = [
            "users",
            "news_articles",
            "user_preferences",
            "reading_history",
            "user_interactions",
            "recommendation_logs"
        ]

        for table in expected_tables:
            self.assertIn(table, tables, f"Expected table '{table}' missing in database")

        conn.close()

    def test_user_operations_and_preferences(self):
        """Verify user insertion and preference handling."""
        db = DatabaseManager(DATABASE_PATH)
        test_uid = "test_user_001"
        test_uname = "phase1_test_user"
        
        # Clean up any existing test user
        db.execute_update("DELETE FROM users WHERE user_id = ?", (test_uid,))
        
        success = db.insert_user(test_uid, test_uname, "dummy_hash", "test@example.com")
        self.assertTrue(success)

        # Upsert preferences
        db.upsert_user_preferences(test_uid, ["en", "hi", "as"], ["technology", "science"])
        prefs = db.get_user_preferences(test_uid)
        
        self.assertIsNotNone(prefs)
        self.assertEqual(prefs["preferred_languages"], ["en", "hi", "as"])
        self.assertEqual(prefs["preferred_categories"], ["technology", "science"])

        # Clean up
        db.execute_update("DELETE FROM users WHERE user_id = ?", (test_uid,))

    def test_helpers_cosine_similarity(self):
        """Verify cosine similarity calculation with lists or numpy arrays."""
        vec1 = [[1.0, 0.0, 0.0]]
        vec2 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        sim = cosine_similarity_matrix(vec1, vec2)
        
        # sim should be 1.0 for first and 0.0 for second
        self.assertAlmostEqual(sim[0][0], 1.0, places=4)
        self.assertAlmostEqual(sim[0][1], 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
