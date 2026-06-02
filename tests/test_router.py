"""Tests for the SQL router."""

import pytest
from proteomics_rag.retrieval.router import is_sql_query, SQLRouter


class TestIsSqlQuery:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("How many SNO sites have fold change greater than 2?", True),
            ("List proteins with p-value below 0.01", True),
            ("Count the total number of significant sites", True),
            ("What is S-nitrosylation?", False),
            ("How does PRMT5 regulate STAT3?", False),
            ("Explain the mechanism of cysteine oxidation", False),
        ],
    )
    def test_classification(self, query, expected):
        assert is_sql_query(query) == expected


class TestSQLRouter:
    def test_execute_no_db(self):
        """Router should gracefully handle missing DB."""
        router = SQLRouter(db_path="/tmp/nonexistent.duckdb")
        result = router.execute("How many papers?")
        # Should return SQL route but with empty results since no data
        assert result["route"] in ("sql", "rag")

    def test_nl_to_sql_fold_change(self):
        router = SQLRouter()
        sql = router._nl_to_sql("How many sites have fold change greater than 2?")
        assert sql is not None
        assert "fold_change" in sql.lower()
        assert "2" in sql

    def test_nl_to_sql_irrelevant(self):
        router = SQLRouter()
        sql = router._nl_to_sql("What is the meaning of life?")
        assert sql is None
