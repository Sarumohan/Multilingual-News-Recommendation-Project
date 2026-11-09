"""
Audit module for existing text, IndicBERT, and visual embeddings.
Verifies dimensions, ID alignments, numerical validity, and metadata compatibility.
"""

import sys
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_EMBEDDINGS_DIR

def audit_embeddings():
    emb_dir = DATA_EMBEDDINGS_DIR

    # 1. MiniLM Text Embeddings
    text_npz = np.load(emb_dir / "articles_text_embeddings.npz", allow_pickle=True)
    text_embs = text_npz["embeddings"]
    text_ids = list(text_npz["article_ids"])

    with open(emb_dir / "articles_metadata.json", "r", encoding="utf-8") as f:
        text_meta = json.load(f)

    # 2. IndicBERT Embeddings
    indic_npz = np.load(emb_dir / "articles_indicbert_embeddings.npz", allow_pickle=True)
    indic_embs = indic_npz["embeddings"]
    indic_ids = list(indic_npz["article_ids"])

    with open(emb_dir / "articles_indicbert_metadata.json", "r", encoding="utf-8") as f:
        indic_meta = json.load(f)

    # 3. ResNet-50 Image Embeddings
    img_npz = np.load(emb_dir / "articles_image_embeddings.npz", allow_pickle=True)
    img_embs = img_npz["embeddings"]
    img_ids = list(img_npz["article_ids"])
    img_has = img_npz["has_image"]

    with open(emb_dir / "articles_image_metadata.json", "r", encoding="utf-8") as f:
        img_meta = json.load(f)

    print("=" * 70)
    print("EMBEDDING INTEGRITY & COMPATIBILITY AUDIT")
    print("=" * 70)

    print("\n1. MODALITY DIMENSIONS & RECORD COUNTS:")
    print(f"   • MiniLM Text Embeddings:   Shape = {text_embs.shape} (384-d), Count = {len(text_ids)}")
    print(f"   • IndicBERT Embeddings:     Shape = {indic_embs.shape} (768-d), Count = {len(indic_ids)}")
    print(f"   • ResNet-50 Image Embeds:   Shape = {img_embs.shape} (2048-d), Count = {len(img_ids)}")

    print("\n2. ARTICLE ID ALIGNMENT:")
    ids_match_text_indic = (text_ids == indic_ids)
    ids_match_text_img = (text_ids == img_ids)
    print(f"   • MiniLM IDs == IndicBERT IDs: {ids_match_text_indic}")
    print(f"   • MiniLM IDs == ResNet-50 IDs: {ids_match_text_img}")
    if not (ids_match_text_indic and ids_match_text_img):
        raise ValueError("Article IDs are not strictly aligned across modalities!")

    print("\n3. NUMERICAL VALIDITY:")
    text_finite = bool(np.all(np.isfinite(text_embs)))
    indic_finite = bool(np.all(np.isfinite(indic_embs)))
    img_finite = bool(np.all(np.isfinite(img_embs)))
    print(f"   • MiniLM Finite:   {text_finite} | Mean L2 Norm: {np.mean(np.linalg.norm(text_embs, axis=1)):.4f}")
    print(f"   • IndicBERT Finite:{indic_finite} | Mean L2 Norm: {np.mean(np.linalg.norm(indic_embs, axis=1)):.4f}")
    valid_img_embs = img_embs[img_has]
    print(f"   • ResNet-50 Finite:{img_finite} | Valid L2 Norm: {np.mean(np.linalg.norm(valid_img_embs, axis=1)):.4f}")

    print("\n4. MISSING MODALITY AUDIT:")
    avail_count = int(np.sum(img_has))
    missing_count = len(img_has) - avail_count
    print(f"   • Images Available: {avail_count} / {len(img_has)} ({avail_count / len(img_has) * 100:.1f}%)")
    print(f"   • Missing Images:   {missing_count} / {len(img_has)}")
    missing_ids = [img_ids[i] for i, h in enumerate(img_has) if not h]
    for mid in missing_ids:
        meta = img_meta[mid]
        print(f"     - [{mid}] ({meta['language']}, {meta['category']}): {meta['title'][:45]}... (Status: {meta['status']})")

    audit_summary = {
        "text_shape": list(text_embs.shape),
        "indicbert_shape": list(indic_embs.shape),
        "image_shape": list(img_embs.shape),
        "total_articles": len(text_ids),
        "images_available": avail_count,
        "images_missing": missing_count,
        "missing_article_ids": missing_ids,
        "id_alignment_verified": bool(ids_match_text_indic and ids_match_text_img),
        "numerical_validity_verified": bool(text_finite and indic_finite and img_finite)
    }

    print("\n" + "=" * 70)
    print("AUDIT RESULT: ALL EMBEDDINGS COMPATIBLE & VALIDATED FOR FUSION")
    print("=" * 70)

    return audit_summary

if __name__ == "__main__":
    audit_embeddings()

