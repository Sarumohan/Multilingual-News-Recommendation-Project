"""
Phase 3 Embedding Generation and Cross-Lingual Semantic Evaluation Runner.
Executes batch encoding across English, Hindi, and Assamese news articles,
persists embeddings in data/embeddings/, and computes intra- and cross-lingual similarity matrices.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_PROCESSED_DIR, DATA_EMBEDDINGS_DIR
from src.embeddings.embedding_generator import ArticleEmbeddingGenerator
from src.embeddings.similarity import EmbeddingSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("EmbeddingRunner")

def run_embedding_pipeline():
    print("=" * 70)
    print("PHASE 3: MULTILINGUAL TEXT EMBEDDING GENERATION & EVALUATION")
    print("=" * 70)

    # 1. Initialize Generator and Encode Processed Articles
    generator = ArticleEmbeddingGenerator()
    embeddings, article_ids, stats = generator.generate_and_save_embeddings(show_progress=True)

    print(f"\n1. EMBEDDINGS GENERATED & PERSISTED:")
    print(f"   - Total Articles Encoded: {stats['total_articles']}")
    print(f"   - Embedding Dimensions:   {stats['embedding_dim']}")
    print(f"   - Model Architecture:     {stats['model_name']}")
    print(f"   - Execution Device:       {stats['device']}")
    print(f"   - Encoding Elapsed Time:  {stats['elapsed_seconds']}s")
    print(f"   - Storage NPZ File:       {DATA_EMBEDDINGS_DIR / 'articles_text_embeddings.npz'}")
    print(f"   - Metadata JSON File:     {DATA_EMBEDDINGS_DIR / 'articles_metadata.json'}")

    # 2. Group Articles by Language
    en_ids = [aid for aid in article_ids if generator.metadata[aid]["language"] == "en"]
    hi_ids = [aid for aid in article_ids if generator.metadata[aid]["language"] == "hi"]
    as_ids = [aid for aid in article_ids if generator.metadata[aid]["language"] == "as"]

    def get_matrix_and_top_pair(id_list_a, id_list_b, is_intra=False):
        vecs_a = np.array([generator.get_article_embedding(aid) for aid in id_list_a])
        vecs_b = np.array([generator.get_article_embedding(aid) for aid in id_list_b])
        sim_mat = EmbeddingSimilarityEngine.compute_similarity_matrix(vecs_a, vecs_b)

        pairs = []
        for i, aid_a in enumerate(id_list_a):
            for j, aid_b in enumerate(id_list_b):
                if is_intra and i >= j:
                    continue
                pairs.append((aid_a, aid_b, float(sim_mat[i, j])))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return sim_mat, pairs

    # 3. Intra-lingual Similarities
    print("\n" + "-" * 70)
    print("2. INTRA-LINGUAL SEMANTIC SIMILARITY")
    print("-" * 70)

    for lang_code, id_list in [("English (EN)", en_ids), ("Hindi (HI)", hi_ids), ("Assamese (AS)", as_ids)]:
        _, top_pairs = get_matrix_and_top_pair(id_list, id_list, is_intra=True)
        print(f"\nTop Similar Pairs within {lang_code}:")
        for aid_a, aid_b, score in top_pairs[:2]:
            t_a = generator.metadata[aid_a]["title"][:45]
            t_b = generator.metadata[aid_b]["title"][:45]
            cat_a = generator.metadata[aid_a]["category"]
            cat_b = generator.metadata[aid_b]["category"]
            print(f"   • [{score:.4f}] '{t_a}...' ({cat_a}) <==> '{t_b}...' ({cat_b})")

    # 4. Cross-lingual Similarities
    print("\n" + "-" * 70)
    print("3. CROSS-LINGUAL SEMANTIC SIMILARITY")
    print("-" * 70)

    cross_pairs = [
        ("English <==> Hindi", en_ids, hi_ids),
        ("English <==> Assamese", en_ids, as_ids),
        ("Hindi <==> Assamese", hi_ids, as_ids),
    ]

    for title, id_a, id_b in cross_pairs:
        _, top_cross = get_matrix_and_top_pair(id_a, id_b, is_intra=False)
        print(f"\nTop Cross-Lingual Semantic Matches ({title}):")
        for aid_a, aid_b, score in top_cross[:3]:
            meta_a = generator.metadata[aid_a]
            meta_b = generator.metadata[aid_b]
            print(f"   • Score: {score:.4f}")
            print(f"     [{meta_a['language'].upper()}] ({meta_a['category']}) {meta_a['title']}")
            print(f"     [{meta_b['language'].upper()}] ({meta_b['category']}) {meta_b['title']}")

    print("\n" + "=" * 70)
    print("PHASE 3 EMBEDDINGS AND SIMILARITY EVALUATION COMPLETE")
    print("=" * 70)

    return stats

if __name__ == "__main__":
    run_embedding_pipeline()

