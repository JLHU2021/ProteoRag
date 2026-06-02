"""Data ingestion layer for ProteoRAG.

Handles loading and chunking of:
- PubMed abstracts (XML/MEDLINE format)
- Scientific PDFs (via pypdf)
- Structured PTM tables (CSV/Excel from MaxQuant)
"""

import json
import re
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd
from tqdm import tqdm

from proteomics_rag.config import DATA_RAW, DUCKDB_PATH, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunk:
    """A single text chunk with its metadata."""

    __slots__ = ("text", "source", "chunk_id", "metadata")

    def __init__(
        self,
        text: str,
        source: str,
        chunk_id: str,
        metadata: dict | None = None,
    ):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            **self.metadata,
        }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks respecting paragraph boundaries.

    Uses a sliding window approach but tries to split on sentence/paragraph
    boundaries to avoid cutting biological terms mid-word.
    """
    # Split on paragraph boundaries first
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < chunk_size:
            current += "\n" + para if current else para
        else:
            if current:
                chunks.append(current)
            # If single paragraph exceeds chunk_size, split on sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) < chunk_size:
                        current += " " + sent if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap by merging boundary chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            boundary = " ".join(chunks[i - 1].split()[-overlap // 2:] + chunks[i].split()[:overlap // 2])
            overlapped.append(chunks[i])
        return overlapped

    return chunks


def load_csv_table(path: Path) -> Iterator[DocumentChunk]:
    """Convert a PTM table (CSV/Excel) into natural-language chunks.

    Uses row-level serialization: each row becomes a descriptive sentence
    that LLMs can understand easily.
    E.g. "Protein GAPDH has SNO site at Cys152 with fold change 2.3 and p-value 0.01."
    """
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    source = path.name

    for idx, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}={val}")
        text = f"{source} row {idx}: " + ", ".join(parts)
        yield DocumentChunk(
            text=text,
            source=source,
            chunk_id=f"{source}_row_{idx}",
            metadata={"type": "table_row", "row_index": idx},
        )


def load_pdf(path: Path) -> Iterator[DocumentChunk]:
    """Extract and chunk text from a scientific PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    reader = PdfReader(str(path))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    chunks = chunk_text(full_text)
    source = path.name
    for i, chunk in enumerate(chunks):
        yield DocumentChunk(
            text=chunk,
            source=source,
            chunk_id=f"{source}_chunk_{i}",
            metadata={"type": "pdf", "chunk_index": i},
        )


def init_duckdb(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB metadata database with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            pmid VARCHAR,
            title VARCHAR,
            authors VARCHAR,
            journal VARCHAR,
            year INTEGER,
            ptm_type VARCHAR,
            abstract TEXT,
            file_path VARCHAR
        )
    """)
    return conn


def ingest_papers(paper_dir: Path = DATA_RAW) -> int:
    """Ingest all papers and tables from the raw data directory.

    Returns total number of chunks ingested.
    """
    conn = init_duckdb()
    chunk_count = 0

    paper_dir.mkdir(parents=True, exist_ok=True)
    files = list(paper_dir.iterdir())

    if not files:
        print(f"No files found in {paper_dir}. Add PDFs, CSVs, or XMLs first.")
        return 0

    for f in tqdm(files, desc="Ingesting"):
        if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
            for chunk in load_csv_table(f):
                chunk_count += 1
        elif f.suffix.lower() == ".pdf":
            for chunk in load_pdf(f):
                chunk_count += 1

    conn.close()
    print(f"Ingested {chunk_count} chunks from {len(files)} files.")
    return chunk_count


if __name__ == "__main__":
    count = ingest_papers()
    print(f"Done: {count} chunks ready for embedding.")
