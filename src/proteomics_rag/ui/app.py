"""Streamlit UI for ProteoRAG.

Provides a clean web interface where users can:
- Ask proteomics/PTM questions in natural language
- See AI answers with source citations
- Click PMID links to original papers
- View retrieval metadata (route, scores, sources)
"""

import logging
from pathlib import Path

import streamlit as st

from proteomics_rag.chain.pipeline import ProteoRAGChain
from proteomics_rag.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProteoRAG",
    page_icon="🧬",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.title("🧬 ProteoRAG")
st.sidebar.markdown(
    "Domain-adapted RAG for proteomics literature\n"
    "and post-translational modifications (PTMs)."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported PTMs:**")
st.sidebar.markdown("- S-nitrosylation (SNO)")
st.sidebar.markdown("- Arginine methylation")
st.sidebar.markdown("- Cysteine oxidation")

# ── Initialize chain ─────────────────────────────────────────────────
@st.cache_resource
def load_chain() -> ProteoRAGChain:
    """Load the RAG chain once and cache it."""
    chain = ProteoRAGChain()
    with st.spinner("Loading models (first run downloads ~2GB)..."):
        chain.load()
    return chain


# Check if index exists before loading
index_path = PROJECT_ROOT / "data" / "index" / "faiss_index.faiss"
if not index_path.exists():
    st.warning(
        "⚠️ FAISS index not found. "
        "Run `python scripts/02_build_index.py` to build the index first."
    )
    st.stop()

chain = load_chain()

# ── Main content ─────────────────────────────────────────────────────
st.title("🧬 ProteoRAG — Proteomics Literature Expert")
st.caption(
    "Ask questions about proteomics, post-translational modifications, "
    "and protein function. Answers are grounded in retrieved literature."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['source']}** (id: `{src['chunk_id']}`, "
                        f"score: {src['score']:.4f})"
                    )

# Query input
if question := st.chat_input("Ask a proteomics question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching proteomics literature..."):
            result = chain.query(question)

        # Show answer
        st.markdown(result["answer"])

        # Show route badge
        if result["route"] == "sql":
            st.info("📊 Answered via structured SQL query")
        else:
            st.success(f"🔍 Answered via RAG ({len(result['sources'])} sources)")

        # Show sources
        if result["sources"]:
            with st.expander("📚 Sources", expanded=True):
                for src in result["sources"]:
                    st.markdown(
                        f"- **{src['source']}** (id: `{src['chunk_id']}`, "
                        f"score: {src['score']:.4f})"
                    )

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with LangChain + FAISS + PubMedBERT + DuckDB + Streamlit | "
    "[GitHub](https://github.com/hujiliang/proteomics-rag)"
)
