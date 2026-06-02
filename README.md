# proteomics-rag

A domain-specialized retrieval-augmented generation (RAG) system for proteomics literature, with a focus on post-translational modifications (PTMs) including S-nitrosylation and arginine methylation.

**Status**: 🚧 Under active development.

## Why this exists

Proteomics research produces a mix of unstructured literature (PDFs, PubMed abstracts, methods manuals) and structured data (MaxQuant output tables, PTM site lists, fold-change matrices). Existing RAG systems either lack domain-specific retrieval quality or cannot hybrid-search across text and tabular data. This project bridges that gap.

## Architecture

```
[PubMed / PDFs / CSVs]
        │
        ▼
   DuckDB ─── metadata store (PMID, journal, year, PTM tag, author)
        │
        ▼
PubMedBERT ── domain-specific embedding (768-dim)
        │
        ▼
   FAISS ──── vector index + similarity search (top-k chunks)
        │
        ▼
Cross-encoder reranker (MS-MARCO fine-tune)
        │
        ▼
 LangChain ── orchestration: retriever → prompt → LLM → answer
        │
        ▼
 Streamlit ── Web UI with citations and clickable PMID links
```

**Key design**: FAISS handles "what is similar" (semantic), DuckDB handles "what is it" (metadata filtering). Hybrid retrieval combines dense (PubMedBERT) + sparse (BM25) with Reciprocal Rank Fusion (RRF), followed by a cross-encoder reranker. A SQL router dispatches structured queries ("how many sites with FC > 2?") to DuckDB instead of RAG.

## Quickstart

```bash
# 1. Install dependencies
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Build the index (downloads ~2GB PubMedBERT + reranker on first run)
python scripts/02_build_index.py

# 3. Launch the UI
streamlit run app.py
```

## Design decisions

| Decision | Why |
|----------|-----|
| PubMedBERT over generic embeddings | Domain terms (S-nitrosylation, PRMT5, ADMA) are single tokens, not split |
| FAISS over cloud vector DBs | Fully local, free, sub-ms retrieval for <1M vectors |
| DuckDB over SQLite | Column-store OLAP, zero-config, Arrow interop |
| RRF (k=60) for hybrid | No tuning needed, robust across query types |
| src layout over flat | Forces proper package installation, catches import bugs early |
| LangChain core only | Avoids heavy Agent/Chain abstractions; uses LCEL + Retriever interface |

## Evaluation results

| Configuration | Recall@5 | MRR | Faithfulness |
|--------------|----------|-----|--------------|
| Baseline (BGE + dense only) | — | — | — |
| + PubMedBERT embedding | — | — | — |
| + BM25 hybrid (RRF) | — | — | — |
| + Cross-encoder rerank | — | — | — |
| Full pipeline + SQL router | — | — | — |

*(Run `python scripts/03_run_eval.py` to populate with real data)*

## Limitations & future work

- **Coverage**: Currently limited to PubMed abstracts; full-text PDF parsing is WIP
- **Language**: English-only; Chinese PTM literature not yet indexed
- **Tables**: Row-level serialization works for simple CSVs; complex supplementary tables need better parsing
- **Self-feedback**: OpenScholar-style iterative refinement is planned for v2.0

## License

MIT — see [LICENSE](LICENSE) for details.
