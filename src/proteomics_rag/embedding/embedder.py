"""Embedding layer — PubMedBERT domain-specific vectorization.

Uses sentence-transformers to load PubMedBERT and encode document chunks
into 768-dimensional vectors for FAISS indexing.
"""

import logging
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from proteomics_rag.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
)

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps PubMedBERT (or any sentence-transformer) for encoding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = EMBEDDING_DIM

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts into normalized vectors."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]


def build_faiss_index(
    chunks: list[dict],
    embedder: Embedder,
    index_path: Path = FAISS_INDEX_PATH,
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build a FAISS index from document chunks.

    Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
    making inner product equivalent to cosine similarity.

    Args:
        chunks: List of chunk dicts with 'text' key.
        embedder: Initialized Embedder instance.
        index_path: Where to save the FAISS index file.

    Returns:
        (faiss_index, metadata_array) — metadata_array maps FAISS IDs
        back to chunk metadata.
    """
    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks with {embedder.model}")
    vectors = embedder.encode(texts)

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # Save metadata (chunk_id, source, etc.) as parallel array
    metadata = np.array(
        [(c["chunk_id"], c["source"]) for c in chunks],
        dtype=[("chunk_id", "U64"), ("source", "U128")],
    )
    meta_path = index_path.with_suffix(".meta.npy")
    np.save(str(meta_path), metadata)

    logger.info(f"FAISS index saved: {index_path} ({index.ntotal} vectors, dim={dim})")
    return index, metadata


def load_faiss_index(
    index_path: Path = FAISS_INDEX_PATH,
) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Load a pre-built FAISS index and its metadata."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run `python scripts/02_build_index.py` first."
        )
    index = faiss.read_index(str(index_path))
    meta_path = index_path.with_suffix(".meta.npy")
    metadata = np.load(str(meta_path), allow_pickle=True)
    return index, metadata


if __name__ == "__main__":
    # Quick test
    embedder = Embedder()
    test = embedder.encode_query("S-nitrosylation of GAPDH at Cys152")
    print(f"Query embedding shape: {test.shape}, norm: {np.linalg.norm(test):.4f}")
