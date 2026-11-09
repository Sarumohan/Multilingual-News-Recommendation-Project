"""
Article Embedding Generation, Persistence, and Semantic Retrieval Module.
Processes cleaned news articles, generates dense multilingual representations,
and persists embeddings with metadata in data/embeddings/.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

from config.config import (
    DATA_PROCESSED_DIR,
    DATA_EMBEDDINGS_DIR,
    DEFAULT_TEXT_MODEL,
    TEXT_EMBEDDING_DIM,
)
from src.embeddings.multilingual_encoder import MultilingualTextEncoder
from src.embeddings.similarity import EmbeddingSimilarityEngine
from src.utils.helpers import setup_logger

logger = setup_logger("EmbeddingGenerator")

EMBEDDINGS_FILE = DATA_EMBEDDINGS_DIR / "articles_text_embeddings.npz"
METADATA_FILE = DATA_EMBEDDINGS_DIR / "articles_metadata.json"

class ArticleEmbeddingGenerator:
    """Manages generation, storage, and retrieval of multilingual news embeddings."""

    def __init__(
        self,
        encoder: Optional[MultilingualTextEncoder] = None,
        model_name: str = DEFAULT_TEXT_MODEL
    ):
        self.encoder = encoder or MultilingualTextEncoder(model_name=model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.article_ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def generate_and_save_embeddings(
        self,
        articles_csv: Optional[Path] = None,
        batch_size: int = 16,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Process articles from CSV, generate embeddings, and save to data/embeddings/.
        """
        csv_path = articles_csv or (DATA_PROCESSED_DIR / "multilingual_news_cleaned.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed news CSV not found at {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"Loaded {len(df)} articles from {csv_path} for embedding generation.")

        # Prepare composite text for embedding
        texts = []
        for _, row in df.iterrows():
            if "cleaned_text" in row and pd.notna(row["cleaned_text"]) and str(row["cleaned_text"]).strip():
                texts.append(str(row["cleaned_text"]).strip())
            else:
                title = str(row.get("title", "")).strip()
                abstract = str(row.get("abstract", "")).strip()
                texts.append(f"{title} {abstract}".strip())

        # Generate embeddings
        start_time = time.time()
        logger.info(f"Encoding {len(texts)} articles using {self.encoder.model_name}...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        elapsed_sec = time.time() - start_time
        logger.info(f"Encoding completed in {elapsed_sec:.2f}s ({len(texts)/max(elapsed_sec, 0.001):.1f} articles/s).")

        article_ids = df["article_id"].tolist()

        # Build metadata dictionary
        metadata_dict = {}
        for idx, row in df.iterrows():
            aid = str(row["article_id"])
            metadata_dict[aid] = {
                "article_id": aid,
                "title": str(row["title"]),
                "language": str(row["language"]),
                "category": str(row["category"]),
                "subcategory": str(row.get("subcategory", "")),
                "source_name": str(row.get("source_name", "")),
                "published_at": str(row.get("published_at", "")),
                "model_name": self.encoder.model_name,
                "embedding_dim": int(embeddings.shape[1]),
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

        # Persist embeddings and metadata
        self.save_embeddings(embeddings, article_ids, metadata_dict)

        self.embeddings = embeddings
        self.article_ids = article_ids
        self.metadata = metadata_dict

        stats = {
            "total_articles": len(article_ids),
            "embedding_dim": int(embeddings.shape[1]),
            "elapsed_seconds": round(elapsed_sec, 2),
            "model_name": self.encoder.model_name,
            "device": self.encoder.device
        }

        return embeddings, article_ids, stats

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        article_ids: List[str],
        metadata: Dict[str, Dict[str, Any]]
    ) -> None:
        """Save embeddings as compressed NPZ and metadata as JSON."""
        DATA_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        # Save compressed NPZ
        np.savez_compressed(
            EMBEDDINGS_FILE,
            embeddings=embeddings,
            article_ids=np.array(article_ids, dtype=object)
        )
        logger.info(f"Saved embeddings to {EMBEDDINGS_FILE} ({embeddings.shape})")

        # Save metadata JSON
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved embedding metadata to {METADATA_FILE}")

    def load_embeddings(self) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, Any]]]:
        """Load persisted embeddings and metadata from disk."""
        if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Persisted embeddings or metadata file not found.")

        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        embeddings = data["embeddings"]
        article_ids = list(data["article_ids"])

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.embeddings = embeddings
        self.article_ids = article_ids
        self.metadata = metadata

        logger.info(f"Loaded {len(article_ids)} embeddings from {EMBEDDINGS_FILE}")
        return embeddings, article_ids, metadata

    def get_article_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding vector for a specific article ID."""
        if self.embeddings is None or not self.article_ids:
            self.load_embeddings()

        try:
            idx = self.article_ids.index(article_id)
            return self.embeddings[idx]
        except ValueError:
            return None

    def find_similar_articles(
        self,
        article_id: str,
        top_k: int = 5,
        target_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find top-K most semantically similar articles to a given article.
        Optionally filter by target language to perform cross-lingual retrieval.
        """
        if self.embeddings is None:
            self.load_embeddings()

        query_vec = self.get_article_embedding(article_id)
        if query_vec is None:
            raise KeyError(f"Article ID '{article_id}' not found in embeddings.")

        # Candidate pool
        if target_language:
            candidate_indices = [
                i for i, aid in enumerate(self.article_ids)
                if self.metadata.get(aid, {}).get("language") == target_language and aid != article_id
            ]
        else:
            candidate_indices = [
                i for i, aid in enumerate(self.article_ids)
                if aid != article_id
            ]

        if not candidate_indices:
            return []

        candidate_embeddings = self.embeddings[candidate_indices]
        candidate_ids = [self.article_ids[i] for i in candidate_indices]

        top_results = EmbeddingSimilarityEngine.find_top_k(
            query_embedding=query_vec,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            top_k=top_k
        )

        results = []
        for aid, score in top_results:
            meta = self.metadata.get(aid, {})
            results.append({
                "article_id": aid,
                "title": meta.get("title", ""),
                "language": meta.get("language", ""),
                "category": meta.get("category", ""),
                "source_name": meta.get("source_name", ""),
                "similarity_score": round(score, 4)
            })

        return results

