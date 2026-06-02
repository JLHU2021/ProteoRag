"""Configuration management for ProteoRAG.

Loads settings from .env file with sensible defaults for all components:
- Embedding model (PubMedBERT)
- FAISS index paths
- DuckDB metadata store
- Reranker model
- LLM provider & API keys
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env if it exists
load_dotenv(PROJECT_ROOT / ".env")

# ── LLM ──────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Embedding ────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── FAISS ────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = Path(
    os.getenv("FAISS_INDEX_PATH", "data/index/faiss_index.faiss")
)
TOP_K = int(os.getenv("TOP_K", "10"))

# ── BM25 ─────────────────────────────────────────────────────────────
BM25_INDEX_PATH = Path(
    os.getenv("BM25_INDEX_PATH", "data/index/bm25_index.pkl")
)

# ── Reranker ─────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# ── DuckDB ───────────────────────────────────────────────────────────
DUCKDB_PATH = Path(
    os.getenv("DUCKDB_PATH", "data/index/metadata.duckdb")
)

# ── Streamlit ────────────────────────────────────────────────────────
STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))

# ── Data paths ───────────────────────────────────────────────────────
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INDEX = PROJECT_ROOT / "data" / "index"
