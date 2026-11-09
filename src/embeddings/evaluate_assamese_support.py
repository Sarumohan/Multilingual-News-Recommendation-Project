"""
Comparative Evaluation and Benchmark Runner for Assamese Multilingual Support.
Compares paraphrase-multilingual-MiniLM-L12-v2 vs ai4bharat/IndicBERTv2-SS
across manually reviewed equivalent and unrelated sentence pairs in Assamese, Hindi, and English.
Saves separate IndicBERT embeddings to data/embeddings/articles_indicbert_embeddings.npz.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR
from src.embeddings.multilingual_encoder import MultilingualTextEncoder
from src.embeddings.indicbert_encoder import IndicBERTEncoder
from src.embeddings.similarity import EmbeddingSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("AssameseEvaluation")

EVAL_PAIRS_FILE = PROJECT_ROOT / "data" / "evaluation" / "multilingual_eval_pairs.json"
INDIC_EMBEDDINGS_FILE = DATA_EMBEDDINGS_DIR / "articles_indicbert_embeddings.npz"
INDIC_METADATA_FILE = DATA_EMBEDDINGS_DIR / "articles_indicbert_metadata.json"
REPORT_OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation" / "comparative_evaluation_report.json"

def run_comparative_evaluation():
    print("=" * 75)
    print("PHASE 3 ENHANCEMENT: ASSAMESE & INDIC MULTILINGUAL EVALUATION")
    print("=" * 75)

    # 1. Load Evaluation Pairs
    if not EVAL_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {EVAL_PAIRS_FILE}")

    with open(EVAL_PAIRS_FILE, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
    print(f"Loaded {len(eval_pairs)} manually reviewed evaluation pairs across AS, HI, EN.")

    # 2. Initialize Models
    print("\nInitializing Model A: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2...")
    encoder_minilm = MultilingualTextEncoder()

    print("Initializing Model B: ai4bharat/IndicBERTv2-SS (with attention-mask mean pooling)...")
    encoder_indicbert = IndicBERTEncoder()

    # 3. Evaluate Both Models on the Evaluation Dataset
    results = []

    print("\n" + "-" * 75)
    print(f"{'Pair ID':16s} | {'Type':5s} | {'Equiv?':6s} | {'MiniLM (384d)':13s} | {'IndicBERT (768d)':16s} | {'Topic'}")
    print("-" * 75)

    for pair in eval_pairs:
        t1, t2 = pair["text1"], pair["text2"]
        is_eq = pair["is_equivalent"]

        # MiniLM
        emb1_m = encoder_minilm.encode(t1)
        emb2_m = encoder_minilm.encode(t2)
        sim_m = float(np.dot(emb1_m, emb2_m))

        # IndicBERT
        emb1_i = encoder_indicbert.encode(t1)
        emb2_i = encoder_indicbert.encode(t2)
        sim_i = float(np.dot(emb1_i, emb2_i))

        results.append({
            "id": pair["id"],
            "pair_type": pair["pair_type"],
            "topic": pair["topic"],
            "is_equivalent": is_eq,
            "minilm_sim": round(sim_m, 4),
            "indicbert_sim": round(sim_i, 4)
        })

        eq_str = "YES" if is_eq else "NO"
        print(f"{pair['id']:16s} | {pair['pair_type']:5s} | {eq_str:6s} | {sim_m:13.4f} | {sim_i:16.4f} | {pair['topic']}")

    print("-" * 75)

    # 4. Grouped Statistics
    df_res = pd.DataFrame(results)

    stats_summary = {}
    print("\n" + "=" * 75)
    print("METRIC SUMMARY BY PAIR TYPE & EQUIVALENCE DISCRIMINATION:")
    print("=" * 75)

    for ptype in ["AS-AS", "AS-HI", "AS-EN", "HI-EN"]:
        sub = df_res[df_res["pair_type"] == ptype]
        eq_m = sub[sub["is_equivalent"]]["minilm_sim"].mean()
        uneq_m = sub[~sub["is_equivalent"]]["minilm_sim"].mean()
        margin_m = eq_m - uneq_m

        eq_i = sub[sub["is_equivalent"]]["indicbert_sim"].mean()
        uneq_i = sub[~sub["is_equivalent"]]["indicbert_sim"].mean()
        margin_i = eq_i - uneq_i

        stats_summary[ptype] = {
            "minilm": {
                "mean_equivalent": round(float(eq_m), 4) if pd.notna(eq_m) else None,
                "mean_unrelated": round(float(uneq_m), 4) if pd.notna(uneq_m) else None,
                "discrimination_margin": round(float(margin_m), 4) if pd.notna(margin_m) else None
            },
            "indicbert": {
                "mean_equivalent": round(float(eq_i), 4) if pd.notna(eq_i) else None,
                "mean_unrelated": round(float(uneq_i), 4) if pd.notna(uneq_i) else None,
                "discrimination_margin": round(float(margin_i), 4) if pd.notna(margin_i) else None
            }
        }

        print(f"\n[{ptype}]")
        print(f"  MiniLM:    Equivalent={eq_m:.4f} | Unrelated={uneq_m:.4f} | Margin (Delta) = {margin_m:+.4f}")
        print(f"  IndicBERT: Equivalent={eq_i:.4f} | Unrelated={uneq_i:.4f} | Margin (Delta) = {margin_i:+.4f}")

    # 5. Process and Persist Articles using IndicBERT Separately
    print("\n" + "=" * 75)
    print("GENERATING SEPARATE INDICBERT ARTICLE EMBEDDINGS")
    print("=" * 75)

    news_csv = DATA_PROCESSED_DIR / "multilingual_news_cleaned.csv"
    news_df = pd.read_csv(news_csv)

    texts = [str(r.get("cleaned_text", r["title"] + " " + r["abstract"])) for _, r in news_df.iterrows()]
    article_ids = news_df["article_id"].tolist()

    t0 = time.time()
    indic_embs = encoder_indicbert.encode(texts, batch_size=8, normalize_embeddings=True)
    t_elapsed = time.time() - t0

    # Save to separate files
    np.savez_compressed(
        INDIC_EMBEDDINGS_FILE,
        embeddings=indic_embs,
        article_ids=np.array(article_ids, dtype=object)
    )

    indic_metadata = {}
    for idx, row in news_df.iterrows():
        aid = str(row["article_id"])
        indic_metadata[aid] = {
            "article_id": aid,
            "title": str(row["title"]),
            "language": str(row["language"]),
            "category": str(row["category"]),
            "model_name": encoder_indicbert.model_name,
            "embedding_dim": int(indic_embs.shape[1]),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

    with open(INDIC_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(indic_metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(article_ids)} IndicBERT embeddings ({indic_embs.shape}) to:")
    print(f"  • {INDIC_EMBEDDINGS_FILE}")
    print(f"  • {INDIC_METADATA_FILE}")
    print(f"Encoding elapsed time: {t_elapsed:.2f}s ({len(texts)/t_elapsed:.1f} articles/s)")

    # 6. Save Comprehensive Report
    report_data = {
        "evaluation_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_pairs_evaluated": len(eval_pairs),
        "models_compared": {
            "model_a": {
                "name": encoder_minilm.model_name,
                "embedding_dim": encoder_minilm.embedding_dim,
                "assamese_native_corpus": False
            },
            "model_b": {
                "name": encoder_indicbert.model_name,
                "embedding_dim": encoder_indicbert.embedding_dim,
                "assamese_native_corpus": True
            }
        },
        "pair_results": results,
        "summary_by_pair_type": stats_summary
    }

    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\nComparative evaluation report saved to {REPORT_OUTPUT_FILE}")
    print("=" * 75)

    return report_data

if __name__ == "__main__":
    run_comparative_evaluation()

