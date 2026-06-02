"""Tests for the data ingestion layer."""

import pytest
from proteomics_rag.data.ingestor import chunk_text, DocumentChunk


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("") == []

    def test_single_paragraph(self):
        result = chunk_text("Hello world")
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_multiple_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = chunk_text(text)
        assert len(result) >= 1

    def test_chunk_size_respected(self):
        text = "A " * 300  # ~600 chars
        result = chunk_text(text, chunk_size=200)
        for chunk in result:
            assert len(chunk) <= 250  # Allow some tolerance


class TestDocumentChunk:
    def test_to_dict(self):
        chunk = DocumentChunk(
            text="test",
            source="file.pdf",
            chunk_id="abc123",
            metadata={"page": 1},
        )
        d = chunk.to_dict()
        assert d["text"] == "test"
        assert d["source"] == "file.pdf"
        assert d["chunk_id"] == "abc123"
        assert d["page"] == 1
