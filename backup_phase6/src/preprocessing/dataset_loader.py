"""
Unified Multilingual News Dataset Loader and Ingestion Pipeline.
Handles format standardization, missing value imputation, deduplication,
and database synchronization.
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from config.config import (
    BASE_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    DATABASE_PATH,
    SUPPORTED_LANGUAGES,
)
from database.db_manager import DatabaseManager
from src.preprocessing.cleaner import MultilingualTextCleaner
from src.utils.helpers import setup_logger

logger = setup_logger("DatasetLoader")

# Canonical columns for unified articles dataframe
UNIFIED_ARTICLE_COLUMNS = [
    "article_id",
    "title",
    "abstract",
    "content",
    "language",
    "category",
    "subcategory",
    "source_name",
    "source_url",
    "image_url",
    "published_at",
    "cleaned_text",
]

class NewsDatasetLoader:
    """Loads, validates, cleans, and standardizes multilingual news datasets."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.cleaner = MultilingualTextCleaner()
        self.db = db_manager or DatabaseManager(DATABASE_PATH)

    def load_from_csv(self, file_path: Path) -> pd.DataFrame:
        """Load news articles from a standard CSV file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return self.preprocess_articles(df)

    def load_mind_news_tsv(self, tsv_path: Path, language: str = "en", source_name: str = "MSN") -> pd.DataFrame:
        """
        Load articles formatted according to the Microsoft News Dataset (MIND) schema.
        Columns: [news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities]
        """
        if not tsv_path.exists():
            raise FileNotFoundError(f"MIND news file not found: {tsv_path}")

        mind_cols = [
            "news_id",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"
        ]

        df = pd.read_csv(
            tsv_path,
            sep="\t",
            names=mind_cols,
            header=None,
            quoting=3,
            encoding="utf-8"
        )

        renamed = pd.DataFrame()
        renamed["article_id"] = df["news_id"]
        renamed["title"] = df["title"]
        renamed["abstract"] = df["abstract"]
        renamed["content"] = df["abstract"]  # MIND does not provide full body text
        renamed["language"] = language
        renamed["category"] = df["category"].str.lower()
        renamed["subcategory"] = df["subcategory"].str.lower()
        renamed["source_name"] = source_name
        renamed["source_url"] = df["url"]
        renamed["image_url"] = ""
        renamed["published_at"] = datetime.now(timezone.utc).isoformat()

        return self.preprocess_articles(renamed)

    def load_mind_behaviors_tsv(self, tsv_path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load behaviors / user interactions formatted according to MIND schema.
        Columns: [impression_id, user_id, time, history, impressions]
        """
        if not tsv_path.exists():
            raise FileNotFoundError(f"MIND behaviors file not found: {tsv_path}")

        behaviors_cols = ["impression_id", "user_id", "time", "history", "impressions"]
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            names=behaviors_cols,
            header=None,
            nrows=max_rows,
            encoding="utf-8"
        )

        return df

    def preprocess_articles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply strict standardization, deduplication, imputation, and cleaning.
        """
        df = df.copy()

        # 1. Map existing column names to canonical columns if needed
        col_mappings = {
            "id": "article_id",
            "news_id": "article_id",
            "body": "content",
            "text": "content",
            "summary": "abstract",
            "lang": "language",
            "image": "image_url",
            "source": "source_name",
            "url": "source_url",
            "timestamp": "published_at",
            "date": "published_at",
        }
        for old_col, new_col in col_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # 2. Ensure all canonical columns exist
        for col in UNIFIED_ARTICLE_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        # 3. Clean string columns and impute missing values
        df["article_id"] = df["article_id"].astype(str).str.strip()
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
        df["content"] = df["content"].fillna("").astype(str).str.strip()
        df["category"] = df["category"].fillna("general").astype(str).str.lower().str.strip()
        df.loc[df["category"] == "", "category"] = "general"
        df["subcategory"] = df["subcategory"].fillna("").astype(str).str.lower().str.strip()
        df["source_name"] = df["source_name"].fillna("NewsHub").astype(str).str.strip()
        df["source_url"] = df["source_url"].fillna("").astype(str).str.strip()
        df["image_url"] = df["image_url"].fillna("").astype(str).str.strip()

        # Discard records with completely empty titles
        df = df[df["title"] != ""].copy()

        # Impute missing abstracts from content
        mask_empty_abstract = (df["abstract"] == "") & (df["content"] != "")
        df.loc[mask_empty_abstract, "abstract"] = df.loc[mask_empty_abstract, "content"].str[:200]

        # Impute missing content from abstract or title
        mask_empty_content = df["content"] == ""
        df.loc[mask_empty_content, "content"] = df.loc[mask_empty_content, "abstract"]
        mask_still_empty_content = df["content"] == ""
        df.loc[mask_still_empty_content, "content"] = df.loc[mask_still_empty_content, "title"]

        # 4. Language validation and auto-detection
        def resolve_language(row):
            declared_lang = str(row["language"]).strip().lower()
            if declared_lang in SUPPORTED_LANGUAGES:
                return declared_lang
            # Auto-detect from title text script
            detected = self.cleaner.detect_script_language(row["title"] + " " + row["abstract"])
            return detected if detected in SUPPORTED_LANGUAGES else "en"

        df["language"] = df.apply(resolve_language, axis=1)

        # 5. Timestamp normalization
        def parse_timestamp(val):
            if not val or pd.isna(val) or str(val).strip() == "":
                return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            val_str = str(val).strip()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %I:%M:%S %p"):
                try:
                    return datetime.strptime(val_str, fmt).strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        df["published_at"] = df["published_at"].apply(parse_timestamp)

        # 6. Deduplication: By article_id first, then by (title, language)
        initial_count = len(df)
        df = df.drop_duplicates(subset=["article_id"], keep="first")
        df = df.drop_duplicates(subset=["title", "language"], keep="first")
        dedup_count = len(df)
        if initial_count > dedup_count:
            logger.info(f"Deduplication removed {initial_count - dedup_count} duplicate articles.")

        # 7. Generate clean representation for embeddings
        df["cleaned_text"] = df.apply(
            lambda r: self.cleaner.prepare_article_representation(r["title"], r["abstract"], r["content"]),
            axis=1
        )

        # Select only canonical columns in standard order
        return df[UNIFIED_ARTICLE_COLUMNS].reset_index(drop=True)

    def save_processed(self, df: pd.DataFrame, file_prefix: str = "multilingual_news") -> Tuple[Path, Path]:
        """Save cleaned dataframe to data/processed/ in CSV and JSON formats."""
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = DATA_PROCESSED_DIR / f"{file_prefix}_cleaned.csv"
        json_path = DATA_PROCESSED_DIR / f"{file_prefix}_cleaned.json"

        df.to_csv(csv_path, index=False, encoding="utf-8")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)

        logger.info(f"Saved processed dataset: {csv_path} ({len(df)} records)")
        return csv_path, json_path

    def sync_to_database(self, df: pd.DataFrame) -> int:
        """
        Insert or update cleaned news articles in SQLite news_articles table.
        Returns count of articles inserted.
        """
        inserted_count = 0
        conn = self.db.get_connection()
        try:
            with conn:
                for _, row in df.iterrows():
                    conn.execute("""
                        INSERT INTO news_articles (
                            article_id, title, abstract, content, language,
                            category, subcategory, source_name, source_url,
                            image_url, published_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(article_id) DO UPDATE SET
                            title = excluded.title,
                            abstract = excluded.abstract,
                            content = excluded.content,
                            language = excluded.language,
                            category = excluded.category,
                            subcategory = excluded.subcategory,
                            source_name = excluded.source_name,
                            source_url = excluded.source_url,
                            image_url = excluded.image_url,
                            published_at = excluded.published_at
                    """, (
                        row["article_id"],
                        row["title"],
                        row["abstract"],
                        row["content"],
                        row["language"],
                        row["category"],
                        row["subcategory"],
                        row["source_name"],
                        row["source_url"],
                        row["image_url"],
                        row["published_at"],
                    ))
                    inserted_count += 1
        finally:
            conn.close()

        logger.info(f"Synchronized {inserted_count} articles to database.")
        return inserted_count

    def generate_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical summary of dataset distribution and quality."""
        stats = {
            "total_articles": len(df),
            "language_distribution": df["language"].value_counts().to_dict(),
            "category_distribution": df["category"].value_counts().to_dict(),
            "sources_count": df["source_name"].nunique(),
            "avg_title_words": round(float(df["title"].str.split().str.len().mean()), 2) if len(df) > 0 else 0,
            "avg_abstract_words": round(float(df["abstract"].str.split().str.len().mean()), 2) if len(df) > 0 else 0,
            "articles_with_images": int((df["image_url"].str.strip() != "").sum()),
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        # Write summary JSON
        summary_path = DATA_PROCESSED_DIR / "dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset statistics generated at {summary_path}")
        return stats
