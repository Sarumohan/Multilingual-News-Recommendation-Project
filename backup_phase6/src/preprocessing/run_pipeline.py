"""
Phase 2 Data Preprocessing and Ingestion Pipeline Runner.
Loads raw/sample news datasets, cleans and standardizes them,
saves processed datasets in data/processed/, and populates SQLite.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_SAMPLE_DIR,
    DATABASE_PATH,
    SUPPORTED_LANGUAGES,
)
from database.db_manager import DatabaseManager
from src.preprocessing.cleaner import MultilingualTextCleaner
from src.preprocessing.dataset_loader import NewsDatasetLoader
from src.utils.helpers import setup_logger

logger = setup_logger("PreprocessingPipeline")

def run_preprocessing_pipeline():
    """Execute complete dataset cleaning, ingestion, and reporting pipeline."""
    logger.info("Initializing Preprocessing Pipeline...")
    db = DatabaseManager(DATABASE_PATH)
    loader = NewsDatasetLoader(db_manager=db)

    # 1. Determine files to process
    sample_news_path = DATA_SAMPLE_DIR / "sample_multilingual_news.csv"
    if not sample_news_path.exists():
        raise FileNotFoundError(f"Sample news dataset not found at {sample_news_path}")

    # Check if raw news data exists
    all_dfs = []
    logger.info(f"Loading primary dataset: {sample_news_path}")
    df_sample = pd.read_csv(sample_news_path, encoding="utf-8")
    all_dfs.append(df_sample)

    # Check for any extra CSVs in data/raw/
    for raw_csv in DATA_RAW_DIR.glob("*.csv"):
        try:
            logger.info(f"Discovered raw CSV: {raw_csv}")
            raw_df = pd.read_csv(raw_csv, encoding="utf-8")
            all_dfs.append(raw_df)
        except Exception as e:
            logger.warning(f"Could not load raw CSV {raw_csv}: {e}")

    # Combine datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total raw records gathered: {len(combined_df)}")

    # 2. Run Cleaning, Deduplication, and Representation Synthesis
    cleaned_df = loader.preprocess_articles(combined_df)
    logger.info(f"Cleaned and standardized records: {len(cleaned_df)}")

    # 3. Save processed datasets to data/processed/
    csv_out, json_out = loader.save_processed(cleaned_df, file_prefix="multilingual_news")

    # 4. Ingest Cleaned Articles into SQLite Database
    synced_articles = loader.sync_to_database(cleaned_df)

    # 5. Load and Ingest User Interactions and Preferences
    interactions_path = DATA_SAMPLE_DIR / "sample_user_interactions.csv"
    synced_interactions = 0
    synced_users = 0

    if interactions_path.exists():
        interactions_df = pd.read_csv(interactions_path, encoding="utf-8")
        logger.info(f"Loading user interactions: {len(interactions_df)} events")

        # Map unique users and create user profiles
        unique_users = interactions_df["user_id"].unique()
        for uid in unique_users:
            uname = f"user_{uid}"
            db.insert_user(uid, uname, "demo_pw_hash", f"{uname}@newsrecommender.org")
            synced_users += 1

            # Determine preferred languages and categories from interaction history
            user_articles = interactions_df[interactions_df["user_id"] == uid]["article_id"]
            user_arts_df = cleaned_df[cleaned_df["article_id"].isin(user_articles)]
            
            top_langs = user_arts_df["language"].value_counts().index.tolist()
            if not top_langs:
                top_langs = ["en"]
            top_cats = user_arts_df["category"].value_counts().index.tolist()
            if not top_cats:
                top_cats = ["technology", "general"]

            db.upsert_user_preferences(uid, top_langs, top_cats)

        # Ingest reading history and explicit interactions
        for _, row in interactions_df.iterrows():
            uid = str(row["user_id"])
            aid = str(row["article_id"])
            itype = str(row["interaction_type"])
            rating = float(row.get("rating", 1.0))
            duration = int(row.get("duration_seconds", 60))

            db.record_reading_history(uid, aid, duration=duration)
            db.record_user_interaction(uid, aid, itype, rating=rating)
            synced_interactions += 1

    # 6. Generate Dataset Statistics Report
    stats = loader.generate_dataset_statistics(cleaned_df)
    stats["database_synced_articles"] = synced_articles
    stats["database_synced_users"] = synced_users
    stats["database_synced_interactions"] = synced_interactions

    # Update summary file
    summary_path = DATA_PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DATASET PREPROCESSING & INGESTION COMPLETED")
    print("=" * 70)
    print(f"Total Processed Articles: {stats['total_articles']}")
    print(f"Language Distribution:    {stats['language_distribution']}")
    print(f"Category Distribution:    {stats['category_distribution']}")
    print(f"Distinct News Sources:    {stats['sources_count']}")
    print(f"Articles with Images:     {stats['articles_with_images']}")
    print(f"Average Title Words:      {stats['avg_title_words']}")
    print(f"Average Abstract Words:   {stats['avg_abstract_words']}")
    print(f"Saved Processed CSV:      {csv_out}")
    print(f"Saved Processed JSON:     {json_out}")
    print(f"Saved Summary Stats:      {summary_path}")
    print("=" * 70)

    return stats

if __name__ == "__main__":
    run_preprocessing_pipeline()

