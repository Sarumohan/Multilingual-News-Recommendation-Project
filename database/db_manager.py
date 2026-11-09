"""
Database management module for News Recommendation Framework.
Provides connection pooling, schema initialization, and transactional queries.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from config.config import DATABASE_PATH, BASE_DIR

SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

class DatabaseManager:
    """Manages SQLite database connections and core data persistence operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a database connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def initialize_schema(self) -> None:
        """Initialize database tables and indexes from schema.sql."""
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        conn = self.get_connection()
        try:
            with conn:
                conn.executescript(schema_sql)
        finally:
            conn.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return rows."""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        finally:
            conn.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT, UPDATE, or DELETE query and return affected rows."""
        conn = self.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.rowcount
        finally:
            conn.close()

    def insert_user(self, user_id: str, username: str, password_hash: str, email: Optional[str] = None) -> bool:
        """Register a new user."""
        query = """
            INSERT INTO users (user_id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
        """
        try:
            self.execute_update(query, (user_id, username, email, password_hash))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by username."""
        query = "SELECT * FROM users WHERE username = ?"
        rows = self.execute_query(query, (username,))
        return dict(rows[0]) if rows else None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user record by user_id."""
        query = "SELECT * FROM users WHERE user_id = ?"
        rows = self.execute_query(query, (user_id,))
        return dict(rows[0]) if rows else None

    def upsert_user_preferences(self, user_id: str, languages: List[str], categories: List[str]) -> None:
        """Insert or update user reading preferences."""
        query = """
            INSERT INTO user_preferences (user_id, preferred_languages, preferred_categories, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                preferred_languages = excluded.preferred_languages,
                preferred_categories = excluded.preferred_categories,
                updated_at = CURRENT_TIMESTAMP
        """
        self.execute_update(query, (user_id, json.dumps(languages), json.dumps(categories)))

    def get_user_preferences(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user preferences."""
        query = "SELECT * FROM user_preferences WHERE user_id = ?"
        rows = self.execute_query(query, (user_id,))
        if not rows:
            return None
        res = dict(rows[0])
        res["preferred_languages"] = json.loads(res["preferred_languages"])
        res["preferred_categories"] = json.loads(res["preferred_categories"])
        return res

    def record_reading_history(self, user_id: str, article_id: str, duration: int = 0) -> None:
        """Log that a user read an article."""
        query = """
            INSERT INTO reading_history (user_id, article_id, reading_duration_seconds)
            VALUES (?, ?, ?)
        """
        self.execute_update(query, (user_id, article_id, duration))

    def record_user_interaction(self, user_id: str, article_id: str, interaction_type: str, rating: float = 1.0) -> None:
        """Log an explicit or implicit user interaction."""
        query = """
            INSERT INTO user_interactions (user_id, article_id, interaction_type, rating)
            VALUES (?, ?, ?, ?)
        """
        self.execute_update(query, (user_id, article_id, interaction_type, rating))

    def get_all_articles(self) -> List[Dict[str, Any]]:
        """Retrieve all news articles from database."""
        query = "SELECT * FROM news_articles ORDER BY published_at DESC"
        rows = self.execute_query(query)
        return [dict(r) for r in rows]

    def get_article_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single article by ID."""
        query = "SELECT * FROM news_articles WHERE article_id = ?"
        rows = self.execute_query(query, (article_id,))
        return dict(rows[0]) if rows else None

    def get_user_interactions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve interactions for a specific user, or all users if user_id is None."""
        if user_id:
            query = "SELECT * FROM user_interactions WHERE user_id = ? ORDER BY timestamp DESC"
            rows = self.execute_query(query, (user_id,))
        else:
            query = "SELECT * FROM user_interactions ORDER BY timestamp DESC"
            rows = self.execute_query(query)
        return [dict(r) for r in rows]

    def get_user_reading_history(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve reading history for a specific user, or all users if user_id is None."""
        if user_id:
            query = "SELECT * FROM reading_history WHERE user_id = ? ORDER BY read_timestamp DESC"
            rows = self.execute_query(query, (user_id,))
        else:
            query = "SELECT * FROM reading_history ORDER BY read_timestamp DESC"
            rows = self.execute_query(query)
        return [dict(r) for r in rows]

    def get_popular_articles(self, limit: int = 10, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve articles ranked by total interaction count for cold-start popularity fallback."""
        if language:
            query = """
                SELECT a.*, COUNT(i.interaction_id) as interaction_count
                FROM news_articles a
                LEFT JOIN user_interactions i ON a.article_id = i.article_id
                WHERE a.language = ?
                GROUP BY a.article_id
                ORDER BY interaction_count DESC, a.published_at DESC
                LIMIT ?
            """
            rows = self.execute_query(query, (language, limit))
        else:
            query = """
                SELECT a.*, COUNT(i.interaction_id) as interaction_count
                FROM news_articles a
                LEFT JOIN user_interactions i ON a.article_id = i.article_id
                GROUP BY a.article_id
                ORDER BY interaction_count DESC, a.published_at DESC
                LIMIT ?
            """
            rows = self.execute_query(query, (limit,))
        return [dict(r) for r in rows]

    def log_recommendation(
        self,
        user_id: str,
        article_id: str,
        algorithm: str,
        relevance_score: float,
        diversity_score: Optional[float] = None,
        explanation: Optional[str] = None
    ) -> None:
        """Log an article recommendation delivered to the user for explainability & auditing."""
        query = """
            INSERT INTO recommendation_logs (user_id, article_id, recommendation_algorithm, relevance_score, diversity_score, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        self.execute_update(query, (user_id, article_id, algorithm, relevance_score, diversity_score, explanation))

