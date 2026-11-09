"""
Inspection script for Phase 2 Verification and Reporting.
Outputs exact counts, schemas, sample records, and database stats.
"""

import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_PROCESSED_DIR, DATA_SAMPLE_DIR, DATABASE_PATH

def generate_report():
    print("=" * 70)
    print("PHASE 2 COMPREHENSIVE STATUS REPORT")
    print("=" * 70)

    # 1. Processed CSV
    csv_file = DATA_PROCESSED_DIR / "multilingual_news_cleaned.csv"
    if not csv_file.exists():
        print("ERROR: Processed CSV file does not exist!")
        return

    df = pd.read_csv(csv_file, encoding="utf-8")
    print(f"\n1. ACTUAL PROCESSED ARTICLES: {len(df)}")
    
    # Language counts
    lang_counts = df["language"].value_counts().to_dict()
    print(f"2. LANGUAGE BREAKDOWN: {lang_counts}")
    print(f"   - English  (en): {lang_counts.get('en', 0)}")
    print(f"   - Hindi    (hi): {lang_counts.get('hi', 0)}")
    print(f"   - Assamese (as): {lang_counts.get('as', 0)}")

    # Sources
    sources = df["source_name"].value_counts().to_dict()
    print(f"\n3. NEWS SOURCES ({len(sources)} distinct):")
    for s, count in sources.items():
        print(f"   - {s}: {count} articles")

    # Categories
    cats = df["category"].value_counts().to_dict()
    print(f"\n4. CATEGORY BREAKDOWN: {cats}")

    # Columns
    print(f"\n5. COLUMNS IN DATASET ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        missing = df[col].isna().sum()
        empty = (df[col].astype(str).str.strip() == "").sum()
        print(f"   {i:2d}. {col:15s} (missing: {missing}, empty: {empty})")

    # Image availability
    has_image = (df["image_url"].astype(str).str.strip() != "").sum()
    print(f"\n6. IMAGE AVAILABILITY: {has_image} / {len(df)} articles ({has_image / len(df) * 100:.1f}%)")

    # User Interactions in CSV & DB
    interactions_csv = DATA_SAMPLE_DIR / "sample_user_interactions.csv"
    if interactions_csv.exists():
        idf = pd.read_csv(interactions_csv)
        print(f"\n7. USER INTERACTIONS DATASET:")
        print(f"   - Total Interaction Events: {len(idf)}")
        print(f"   - Unique Users: {idf['user_id'].nunique()}")
        print(f"   - Interaction Types: {idf['interaction_type'].value_counts().to_dict()}")

    # Database Verification
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) AS c FROM news_articles")
    db_arts = c.fetchone()["c"]

    c.execute("SELECT language, COUNT(*) AS c FROM news_articles GROUP BY language")
    db_langs = {r["language"]: r["c"] for r in c.fetchall()}

    c.execute("SELECT COUNT(*) AS c FROM users")
    db_users = c.fetchone()["c"]

    c.execute("SELECT COUNT(*) AS c FROM user_interactions")
    db_interactions = c.fetchone()["c"]

    c.execute("SELECT COUNT(*) AS c FROM reading_history")
    db_history = c.fetchone()["c"]

    conn.close()

    print(f"\n8. SQLITE DATABASE VERIFICATION:")
    print(f"   - news_articles in DB: {db_arts}")
    print(f"   - languages in DB:     {db_langs}")
    print(f"   - registered users:    {db_users}")
    print(f"   - user_interactions:   {db_interactions}")
    print(f"   - reading_history logs:{db_history}")

    # Sample records per language
    print("\n9. SAMPLE RECORDS (One per language):")
    for lang in ["en", "hi", "as"]:
        sample = df[df["language"] == lang].iloc[0]
        print(f"\n   [{lang.upper()}] ID: {sample['article_id']} | Category: {sample['category']} | Source: {sample['source_name']}")
        print(f"   Title:    {sample['title']}")
        print(f"   Abstract: {sample['abstract']}")
        print(f"   Image:    {sample['image_url']}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    generate_report()

