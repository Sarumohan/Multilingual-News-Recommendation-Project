"""
Phase 4 Image Feature Extraction and Visual Similarity Runner.
Executes the image downloading, ResNet-50 feature extraction,
persistence, and visual similarity analysis across multilingual news articles.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR
from src.vision.image_feature_pipeline import ImageFeaturePipeline
from src.vision.similarity import ImageSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("VisionRunner")

def run_vision_pipeline():
    print("=" * 75)
    print("PHASE 4: MULTIMODAL IMAGE FEATURE EXTRACTION & EVALUATION")
    print("=" * 75)

    pipeline = ImageFeaturePipeline()
    stats = pipeline.run_pipeline(auto_download=False)  # Images are already in data/images/

    print(f"\n1. IMAGE FEATURE EXTRACTION SUMMARY:")
    print(f"   - Total Articles:       {stats['total_articles']}")
    print(f"   - Images Available:     {stats['images_available']}")
    print(f"   - Images Failed/Missing:{stats['images_failed']}")
    print(f"   - Feature Dimension:    {stats['embedding_dim']}")
    print(f"   - Model Architecture:   {stats['model_name']}")
    print(f"   - Execution Device:     {stats['device']}")
    print(f"   - Extraction Time:      {stats['elapsed_seconds']}s")
    print(f"   - Embeddings Storage:   {stats['npz_path']}")
    print(f"   - Metadata Storage:     {stats['metadata_path']}")

    # 2. Multimodal Data Mapping Breakdown by Language
    print("\n" + "-" * 75)
    print("2. MULTIMODAL ARTICLE MAPPING BY LANGUAGE:")
    print("-" * 75)
    
    meta = pipeline.metadata
    for lang in ["en", "hi", "as"]:
        lang_items = [v for v in meta.values() if v["language"] == lang]
        avail = sum(1 for v in lang_items if v["is_available"])
        print(f"\n[{lang.upper()}] Total: {len(lang_items)} | Images Available: {avail} | Missing: {len(lang_items) - avail}")
        for item in lang_items:
            status_symbol = "✓" if item["is_available"] else "✗"
            title_trunc = item["title"][:40]
            print(f"   [{status_symbol}] {item['article_id']} ({item['category']:10s}): {title_trunc}... -> Status: {item['status']}")

    # 3. Visual Similarity Analysis among Available Images
    print("\n" + "-" * 75)
    print("3. VISUAL SIMILARITY EVALUATION (Available Images):")
    print("-" * 75)

    valid_ids = [aid for aid, v in meta.items() if v["is_available"]]
    valid_vecs = np.array([pipeline.get_article_image_embedding(aid) for aid in valid_ids])

    sim_mat = ImageSimilarityEngine.compute_similarity_matrix(valid_vecs)

    pairs = []
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            pairs.append((valid_ids[i], valid_ids[j], float(sim_mat[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 Visually Most Similar Image Pairs:")
    for a1, a2, score in pairs[:5]:
        m1 = meta[a1]
        m2 = meta[a2]
        print(f"   • Score: {score:.4f}")
        print(f"     A: [{m1['language'].upper()}] ({m1['category']}) {m1['title'][:45]}")
        print(f"     B: [{m2['language'].upper()}] ({m2['category']}) {m2['title'][:45]}")

    print("\nLeast Similar Image Pairs (Visual Contrast):")
    for a1, a2, score in pairs[-3:]:
        m1 = meta[a1]
        m2 = meta[a2]
        print(f"   • Score: {score:.4f}")
        print(f"     A: [{m1['language'].upper()}] ({m1['category']}) {m1['title'][:45]}")
        print(f"     B: [{m2['language'].upper()}] ({m2['category']}) {m2['title'][:45]}")

    print("\n" + "=" * 75)
    print("PHASE 4 IMAGE FEATURE PIPELINE COMPLETE")
    print("=" * 75)

    return stats

if __name__ == "__main__":
    run_vision_pipeline()

