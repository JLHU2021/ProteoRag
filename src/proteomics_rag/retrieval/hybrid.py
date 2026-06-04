"""Hybrid retrieval layer — combines dense (FAISS), sparse (BM25), and reranking.

Implements Reciprocal Rank Fusion (RRF) to merge dense and sparse results,
followed by a cross-encoder reranker for final precision.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from proteomics_rag.config import BM25_INDEX_PATH, RERANKER_MODEL, TOP_K, RERANK_TOP_K
from proteomics_rag.embedding.embedder import Embedder, load_faiss_index

logger = logging.getLogger(__name__)

# RRF constant — empirically good default, no tuning needed
RRF_K = 60


class BM25Retriever:
    """Sparse keyword-based retrieval using Okapi BM25."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.corpus: list[str] = []
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[dict], save_path: Path = BM25_INDEX_PATH):
        """Build BM25 index from chunks."""
        logger.info(f"Building BM25 index over {len(chunks)} chunks")
        tokenized = [doc["text"].lower().split() for doc in chunks]
        self.corpus = [doc["text"] for doc in chunks]
        self.chunk_ids = [doc["chunk_id"] for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "corpus": self.corpus, "chunk_ids": self.chunk_ids},
                f,
            )
        logger.info(f"BM25 index saved to {save_path}")

    def load(self, save_path: Path = BM25_INDEX_PATH):
        """Load a pre-built BM25 index."""
        if not save_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.corpus = data["corpus"]
        self.chunk_ids = data["chunk_ids"]
        logger.info(f"BM25 index loaded ({len(self.corpus)} docs)")

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search and return (chunk_id, score) pairs."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded or built")
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]


class Reranker:
    """Cross-encoder reranker for re-ranking top candidates."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        logger.info(f"Loading reranker: {model_name}")
        print(f"Loading reranker model: {model_name} (this may take a moment on first run)...")
        self.model = CrossEncoder(model_name, local_files_only=True)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, str]],  # (chunk_id, text, source)
        top_k: int = RERANK_TOP_K,
    ) -> list[tuple[str, str, str, float]]:
        """Rerank candidates by cross-encoder score.

        Returns sorted list of (chunk_id, text, source, score).
        """
        if not candidates:
            return []
        pairs = [(query, text) for _, text, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(
                [c[0] for c in candidates],
                [c[1] for c in candidates],
                [c[2] for c in candidates],
                scores,
            ),
            key=lambda x: x[3],
            reverse=True,
        )
        return ranked[:top_k]


class HybridRetriever:
    """Hybrid retrieval combining dense (FAISS), sparse (BM25), and reranking.

    Pipeline:
    1. Dense search via FAISS (PubMedBERT embeddings)
    2. Sparse search via BM25 (keyword matching)
    3. Merge with Reciprocal Rank Fusion (RRF)
    4. Rerank top candidates with cross-encoder
    """

    def __init__(self):
        self.faiss_index = None
        self.faiss_metadata = None
        self.embedder: Embedder | None = None
        self.bm25 = BM25Retriever()
        self.reranker: Reranker | None = None

    def load(
        self,
        chunk_texts: list[str] | None = None,
    ):
        """Load all retrieval components."""
        self.faiss_index, self.faiss_metadata = load_faiss_index()
        self.embedder = Embedder()

        # Try to load BM25; if unavailable, fall back to dense-only
        try:
            self.bm25.load()
            logger.info("BM25 index loaded — hybrid retrieval enabled")
        except FileNotFoundError:
            logger.warning("BM25 index not found — dense-only retrieval")

        self.reranker = Reranker()

    def _rrf_merge(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Merge dense and sparse results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        for rank, (chunk_id, _) in enumerate(sparse_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Execute hybrid search and return ranked chunk dicts."""
        if self.faiss_index is None or self.embedder is None:
            raise RuntimeError("Retriever not loaded. Call .load() first.")

        # Dense search
        query_vec = self.embedder.encode_query(query).reshape(1, -1)
        dense_scores, dense_ids = self.faiss_index.search(query_vec, top_k * 2)
        dense_results = [
            (self.faiss_metadata[i]["chunk_id"], float(dense_scores[0][j]))
            for j, i in enumerate(dense_ids[0])
            if i != -1
        ]

        # Sparse search
        sparse_results = []
        if self.bm25.bm25 is not None:
            sparse_results = self.bm25.search(query, top_k * 2)

        # RRF merge
        merged = self._rrf_merge(dense_results, sparse_results, top_k * 2)

        # Gather chunk texts for reranking
        candidates = []
        chunk_id_to_text = self._build_id_to_text_map()
        for chunk_id, _ in merged:
            if chunk_id in chunk_id_to_text:
                text, source = chunk_id_to_text[chunk_id]
                candidates.append((chunk_id, text, source))

        # Rerank
        if self.reranker is not None and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k)
            return [
                {
                    "chunk_id": c[0],
                    "text": c[1],
                    "source": c[2],
                    "score": float(c[3]),
                }
                for c in reranked
            ]

        # Fallback without reranker
        return [
            {"chunk_id": c[0], "text": chunk_id_to_text.get(c[0], ("", ""))[0], "score": c[1]}
            for c in merged[:top_k]
        ]

    def _build_id_to_text_map(self) -> dict[str, tuple[str, str]]:
        """Build a mapping from chunk_id to (text, source)."""
        import json
        from proteomics_rag.config import DATA_INDEX

        chunk_store_path = DATA_INDEX / "chunk_store.json"
        if not chunk_store_path.exists():
            logger.warning(f"Chunk store not found at {chunk_store_path}")
            return {}

        with open(chunk_store_path) as f:
            chunks = json.load(f)

        return {c["chunk_id"]: (c["text"], c["source"]) for c in chunks}


if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.load()
    results = retriever.search("S-nitrosylation of PRMT5")
    for r in results[:5]:
        print(f"[{r['score']:.4f}] {r['chunk_id']} — {r['text'][:80]}...")
