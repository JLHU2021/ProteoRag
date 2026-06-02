"""Shared pytest fixtures for ProteoRAG tests."""

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "index").mkdir()
    return tmp_path


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample PTM table CSV."""
    csv_path = tmp_path / "sample_ptm.csv"
    csv_path.write_text(
        "Protein,Site,FoldChange,PValue\n"
        "GAPDH,Cys152,2.3,0.001\n"
        "STAT3,Tyr705,1.8,0.01\n"
        "p53,Ser15,3.1,0.0001\n"
    )
    return csv_path
