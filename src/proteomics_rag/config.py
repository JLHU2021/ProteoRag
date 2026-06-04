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

# Load .env if it exists (override system env vars)
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Force offline mode for Hugging Face — models are cached locally,
# and Python 3.13 httpx client lifecycle can cause "client closed" errors.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# ── LLM ──────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "dashscope")  # Default to DashScope (Qwen)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

# LLM Model (DashScope defaults)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")  # e.g., qwen-turbo, qwen-plus, qwen-max

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
