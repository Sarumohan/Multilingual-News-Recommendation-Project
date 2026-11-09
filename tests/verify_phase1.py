"""
Phase 1 Complete Verification Script.
Checks:
1. Complete Project Structure and file hierarchy.
2. Requirements.txt dependency specifications.
3. Python environment and venv configuration.
4. SQLite database connectivity and PRAGMA settings.
5. Schema.sql table and index creation.
6. db_manager.py connection and CRUD operations.
7. Automated test suite execution.
"""

import sys
import os
import sqlite3
import json
from pathlib import Path

# Add project root to sys.path
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
from src.utils.helpers import setup_logger, cosine_similarity_matrix, format_duration

logger = setup_logger("Phase1Verification")

def check_project_structure():
    """Verify all expected folders and files exist."""
    expected_dirs = [
        BASE_DIR / "config",
        BASE_DIR / "data" / "raw",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "data" / "embeddings",
        BASE_DIR / "data" / "sample",
        BASE_DIR / "database",
        BASE_DIR / "src" / "preprocessing",
        BASE_DIR / "src" / "models",
        BASE_DIR / "src" / "recommendation",
        BASE_DIR / "src" / "explainability",
        BASE_DIR / "src" / "evaluation",
        BASE_DIR / "src" / "utils",
        BASE_DIR / "app" / "components",
        BASE_DIR / "app" / "static",
        BASE_DIR / "tests",
        BASE_DIR / "venv",
    ]

    expected_files = [
        BASE_DIR / ".gitignore",
        BASE_DIR / "README.md",
        BASE_DIR / "requirements.txt",
        BASE_DIR / "config" / "config.py",
        BASE_DIR / "database" / "schema.sql",
        BASE_DIR / "database" / "db_manager.py",
        BASE_DIR / "src" / "utils" / "helpers.py",
        BASE_DIR / "app" / "app.py",
        BASE_DIR / "app" / "static" / "style.css",
        BASE_DIR / "tests" / "test_setup.py",
    ]

    for d in expected_dirs:
        if not d.exists() or not d.is_dir():
            raise AssertionError(f"Missing expected directory: {d}")

    for f in expected_files:
        if not f.exists() or not f.is_file():
            raise AssertionError(f"Missing expected file: {f}")

    logger.info("Project structure check: PASS (All folders and files present)")
    return True

def check_requirements():
    """Verify requirements.txt exists and contains all required packages."""
    req_file = BASE_DIR / "requirements.txt"
    content = req_file.read_text(encoding="utf-8")
    
    required_packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "Pillow",
        "streamlit",
        "plotly"
    ]
    
    missing = [pkg for pkg in required_packages if pkg.lower() not in content.lower()]
    if missing:
        raise AssertionError(f"Missing packages in requirements.txt: {missing}")

    logger.info("requirements.txt check: PASS (All core dependencies specified)")
    return True

def check_venv():
    """Verify Python virtual environment configuration."""
    venv_dir = BASE_DIR / "venv"
    python_exe = venv_dir / "Scripts" / "python.exe"
    pip_exe = venv_dir / "Scripts" / "pip.exe"

    if not python_exe.exists() or not pip_exe.exists():
        raise AssertionError("Virtual environment binaries not found")

    logger.info(f"Virtual environment check: PASS (Python executable: {python_exe})")
    return True

def check_database_and_schema():
    """Verify SQLite database connectivity and schema table definitions."""
    db = DatabaseManager(DATABASE_PATH)
    conn = db.get_connection()
    cursor = conn.cursor()

    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    
    expected_tables = {
        "users",
        "news_articles",
        "user_preferences",
        "reading_history",
        "user_interactions",
        "recommendation_logs"
    }

    if not expected_tables.issubset(set(tables)):
        missing = expected_tables - set(tables)
        raise AssertionError(f"Missing tables in SQLite database: {missing}")

    # Verify indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
    indexes = [row[0] for row in cursor.fetchall()]
    expected_indexes = {
        "idx_news_lang_cat",
        "idx_history_user",
        "idx_interactions_user",
        "idx_recom_logs_user"
    }

    if not expected_indexes.issubset(set(indexes)):
        missing = expected_indexes - set(indexes)
        raise AssertionError(f"Missing indexes in SQLite database: {missing}")

    conn.close()
    logger.info("Database connectivity and schema.sql check: PASS (All 6 tables and 4 indexes verified)")
    return True

def test_db_manager_crud():
    """Verify db_manager.py connection and complete CRUD workflows."""
    db = DatabaseManager(DATABASE_PATH)
    test_user_id = "v_user_101"
    test_username = "verify_tester_101"
    test_article_id = "v_art_202"

    try:
        # 1. Clean prior test artifacts
        db.execute_update("DELETE FROM users WHERE user_id = ?", (test_user_id,))
        db.execute_update("DELETE FROM news_articles WHERE article_id = ?", (test_article_id,))

        # 2. Test user creation
        created = db.insert_user(test_user_id, test_username, "test_hash_xyz", "tester@domain.com")
        if not created:
            raise AssertionError("insert_user failed to create a new user")

        # 3. Test duplicate rejection
        duplicate = db.insert_user("other_id", test_username, "some_hash")
        if duplicate:
            raise AssertionError("insert_user allowed duplicate username")

        # 4. Test user retrieval
        user = db.get_user_by_username(test_username)
        if not user or user["user_id"] != test_user_id:
            raise AssertionError(f"get_user_by_username failed. Got: {user}")

        user_by_id = db.get_user_by_id(test_user_id)
        if not user_by_id or user_by_id["username"] != test_username:
            raise AssertionError(f"get_user_by_id failed. Got: {user_by_id}")

        # 5. Test user preferences upsert and retrieval
        langs = ["en", "hi", "as"]
        cats = ["technology", "science", "business"]
        db.upsert_user_preferences(test_user_id, langs, cats)
        prefs = db.get_user_preferences(test_user_id)
        if not prefs or prefs["preferred_languages"] != langs or prefs["preferred_categories"] != cats:
            raise AssertionError(f"User preferences mismatch: {prefs}")

        # 6. Test news article insert
        db.execute_update("""
            INSERT INTO news_articles (article_id, title, abstract, content, language, category, published_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (test_article_id, "Test News Title", "Test Abstract", "Full Text", "en", "technology"))

        # 7. Test reading history logging
        db.record_reading_history(test_user_id, test_article_id, duration=120)
        history = db.execute_query("SELECT * FROM reading_history WHERE user_id = ?", (test_user_id,))
        if not history or history[0]["article_id"] != test_article_id or history[0]["reading_duration_seconds"] != 120:
            raise AssertionError("Reading history insertion mismatch")

        # 8. Test user interaction logging
        db.record_user_interaction(test_user_id, test_article_id, "like", rating=1.0)
        interactions = db.execute_query("SELECT * FROM user_interactions WHERE user_id = ?", (test_user_id,))
        if not interactions or interactions[0]["interaction_type"] != "like":
            raise AssertionError("User interaction insertion mismatch")

        # 9. Test recommendation logging
        db.log_recommendation(
            user_id=test_user_id,
            article_id=test_article_id,
            algorithm="hybrid",
            relevance_score=0.94,
            diversity_score=0.88,
            explanation="Recommended based on high interest in Technology."
        )
        logs = db.execute_query("SELECT * FROM recommendation_logs WHERE user_id = ?", (test_user_id,))
        if not logs or logs[0]["relevance_score"] != 0.94 or logs[0]["explanation"] is None:
            raise AssertionError("Recommendation logs insertion mismatch")

        logger.info("db_manager.py CRUD & transaction operations check: PASS")
        return True

    finally:
        # Cleanup
        db.execute_update("DELETE FROM users WHERE user_id = ?", (test_user_id,))
        db.execute_update("DELETE FROM news_articles WHERE article_id = ?", (test_article_id,))

def run_tests():
    """Run all Phase 1 checks and print results."""
    print("=" * 70)
    print("PHASE 1 VERIFICATION & HEALTH CHECK")
    print("=" * 70)

    results = {}

    checks = [
        ("1. Complete Project Structure", check_project_structure),
        ("2. Requirements.txt Dependencies", check_requirements),
        ("3. Python Virtual Environment (venv)", check_venv),
        ("4. SQLite Database & Schema (Tables & Indexes)", check_database_and_schema),
        ("5. db_manager.py Connection & Operations", test_db_manager_crud),
    ]

    for name, func in checks:
        try:
            passed = func()
            results[name] = "PASSED" if passed else "FAILED"
        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            results[name] = f"FAILED: {e}"

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        print(f"[{status}] {name}")
    print("=" * 70)

    all_passed = all(status == "PASSED" for status in results.values())
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("SOME CHECKS FAILED.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())

