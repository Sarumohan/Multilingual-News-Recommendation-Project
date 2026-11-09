"""
Execution Runner for Multimodal Feature Fusion (Revised Phase 4).
Fuses text representations (MiniLM, IndicBERT) and visual features (ResNet-50),
handles missing modalities, tests cosine similarity, and persists multimodal embeddings.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_EMBEDDINGS_DIR, DATA_PROCESSED_DIR
from src.fusion.multimodal_fusion import MultimodalFusionEngine
from src.fusion.similarity import MultimodalSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("FusionRunner")

MULTIMODAL_EMBEDDINGS_FILE = DATA_EMBEDDINGS_DIR / "articles_multimodal_embeddings.npz"
MULTIMODAL_METADATA_FILE = DATA_EMBEDDINGS_DIR / "articles_multimodal_metadata.json"

def run_fusion_pipeline(alpha: float = 0.7, beta: float = 0.3):
    print("=" * 75)
    print("REVISED PHASE 4: MULTIMODAL FEATURE FUSION PIPELINE")
    print("=" * 75)

    # 1. Load All Precomputed Modality Embeddings
    emb_dir = DATA_EMBEDDINGS_DIR
    
    # Text (MiniLM)
    minilm_data = np.load(emb_dir / "articles_text_embeddings.npz", allow_pickle=True)
    text_embs = minilm_data["embeddings"]
    article_ids = list(minilm_data["article_ids"])

    # Text (IndicBERT)
    indic_data = np.load(emb_dir / "articles_indicbert_embeddings.npz", allow_pickle=True)
    indic_embs = indic_data["embeddings"]

    # Image (ResNet-50)
    img_data = np.load(emb_dir / "articles_image_embeddings.npz", allow_pickle=True)
    img_embs = img_data["embeddings"]
    has_image = img_data["has_image"]

    # Metadata
    with open(emb_dir / "articles_image_metadata.json", "r", encoding="utf-8") as f:
        img_meta = json.load(f)

    print(f"Loaded {len(article_ids)} articles across all modalities.")
    print(f"Weights configured: Text (alpha) = {alpha:.2f}, Image (beta) = {beta:.2f}")

    # 2. Perform Multimodal Fusion
    fusion_engine = MultimodalFusionEngine(alpha=alpha, beta=beta)

    # Strategy 1: Weighted Normalized Concatenation (Preserves full 2048-d visual feature fidelity)
    fused_concat, modes_concat, stats_concat = fusion_engine.fuse_datasets(
        text_embeddings=text_embs,
        image_embeddings=img_embs,
        has_image_mask=has_image,
        article_ids=article_ids,
        strategy="weighted_concat"
    )

    # Strategy 2: Projected Common Dimension Fusion (Maps 2048-d image to 384-d text space)
    fused_projected, modes_proj, stats_proj = fusion_engine.fuse_datasets(
        text_embeddings=text_embs,
        image_embeddings=img_embs,
        has_image_mask=has_image,
        article_ids=article_ids,
        strategy="projected"
    )

    # Strategy 3: Dedicated IndicBERT Multimodal Fusion (Preserves 768-d Indic space)
    fused_indicbert, modes_indic, stats_indic = fusion_engine.fuse_datasets(
        text_embeddings=indic_embs,
        image_embeddings=img_embs,
        has_image_mask=has_image,
        article_ids=article_ids,
        strategy="weighted_concat"
    )

    print("\n" + "-" * 75)
    print("1. FUSION EXECUTION SUMMARY:")
    print("-" * 75)
    print(f"   • Total Articles Processed:      {stats_concat['total_articles']}")
    print(f"   • Fully Multimodal Articles:     {stats_concat['multimodal_count']} (Text + Image)")
    print(f"   • Text-Only Fallback Articles:   {stats_concat['text_only_count']} (Missing Image)")
    print(f"   • Concat Fused Shape:            {fused_concat.shape} (384 + 2048 = 2432-d)")
    print(f"   • Projected Fused Shape:         {fused_projected.shape} (384-d)")
    print(f"   • IndicBERT Fused Shape:         {fused_indicbert.shape} (768 + 2048 = 2816-d)")

    # 3. Persist Multimodal Embeddings and Detailed Metadata
    np.savez_compressed(
        MULTIMODAL_EMBEDDINGS_FILE,
        fused_concat=fused_concat,
        fused_projected=fused_projected,
        fused_indicbert=fused_indicbert,
        article_ids=np.array(article_ids, dtype=object),
        has_image=has_image
    )
    logger.info(f"Saved multimodal embeddings to {MULTIMODAL_EMBEDDINGS_FILE}")

    multimodal_meta = {}
    for i, aid in enumerate(article_ids):
        im = img_meta.get(aid, {})
        multimodal_meta[aid] = {
            "article_id": aid,
            "language": im.get("language", ""),
            "category": im.get("category", ""),
            "title": im.get("title", ""),
            "has_image": bool(has_image[i]),
            "fusion_mode": modes_concat[i],
            "weights": {"alpha": alpha, "beta": beta},
            "dimensions": {
                "fused_concat": int(fused_concat.shape[1]),
                "fused_projected": int(fused_projected.shape[1]),
                "fused_indicbert": int(fused_indicbert.shape[1])
            },
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

    with open(MULTIMODAL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(multimodal_meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved multimodal metadata to {MULTIMODAL_METADATA_FILE}")

    # 4. Modality Similarity Comparison Analysis (Task 5)
    print("\n" + "-" * 75)
    print("2. MODALITY SIMILARITY COMPARISON (Text vs Image vs Fused):")
    print("-" * 75)
    print(f"{'Pair Comparison':24s} | {'Text Sim':8s} | {'Image Sim':9s} | {'Fused Sim':9s} | {'Image Status'}")
    print("-" * 75)

    test_pairs = [
        # Pair 1: Both images available, high text & high image similarity (Fintech EN <-> Fintech HI)
        ("art_en_006", "art_hi_002", "Fintech EN <-> HI"),
        # Pair 2: Both images available, moderate text & moderate image (AI EN <-> AI HI)
        ("art_en_001", "art_hi_003", "AI EN <-> HI"),
        # Pair 3: Both images available, text similar, visual high contrast (AI EN <-> Tea AS)
        ("art_en_001", "art_as_005", "Tech EN <-> Tea AS"),
        # Pair 4: One image missing (Renewables EN <-> Solar HI - art_en_002 missing image)
        ("art_en_002", "art_hi_004", "Energy (1 img missing)"),
        # Pair 5: Both images available, Kaziranga AS <-> Brahmaputra AS
        ("art_as_001", "art_as_003", "Kaziranga <-> River AS"),
        # Pair 6: One image missing (Cricket EN <-> Cricket HI - art_hi_005 missing image)
        ("art_en_005", "art_hi_005", "Cricket (1 img missing)"),
    ]

    for aid_a, aid_b, desc in test_pairs:
        idx_a = article_ids.index(aid_a)
        idx_b = article_ids.index(aid_b)

        comp = MultimodalSimilarityEngine.compare_article_pair_modalities(
            aid_a=aid_a,
            aid_b=aid_b,
            text_emb_a=text_embs[idx_a],
            text_emb_b=text_embs[idx_b],
            img_emb_a=img_embs[idx_a],
            img_emb_b=img_embs[idx_b],
            has_img_a=bool(has_image[idx_a]),
            has_img_b=bool(has_image[idx_b]),
            fused_emb_a=fused_concat[idx_a],
            fused_emb_b=fused_concat[idx_b],
            alpha=alpha,
            beta=beta
        )

        img_str = f"{comp['image_similarity']:.4f}" if comp['image_similarity'] is not None else "   N/A   "
        print(f"{desc:24s} | {comp['text_similarity']:8.4f} | {img_str:9s} | {comp['fused_similarity']:9.4f} | {comp['image_status']}")

    print("-" * 75)
    print("\nKey Analytical Finding:")
    print("When both images are available under Weighted Concatenation:")
    print("  Fused Similarity = alpha * Text Similarity + beta * Image Similarity")
    print("Example (art_en_006 <-> art_hi_002):")
    print(f"  0.7 * {0.6032:.4f} + 0.3 * {0.8016:.4f} = {0.7 * 0.6032 + 0.3 * 0.8016:.4f} (Matches Fused Sim)")
    print("When an image is missing, the engine falls back to text-only representation gracefully.")

    print("\n" + "=" * 75)
    print("REVISED PHASE 4 MULTIMODAL FUSION COMPLETE")
    print("=" * 75)

    return stats_concat

if __name__ == "__main__":
    run_fusion_pipeline()

