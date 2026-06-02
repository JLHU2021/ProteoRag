"""SQL router — dispatches structured queries to DuckDB instead of RAG.

Detects when a user question is answerable via SQL (e.g., "how many sites
with fold change > 2?") and routes it to DuckDB for precise execution.

Uses a lightweight LLM-based classifier with rule-based fallback.
"""

import logging
import re

import duckdb

from proteomics_rag.config import DUCKDB_PATH

logger = logging.getLogger(__name__)

# Keywords that suggest a structured/tabular query
SQL_KEYWORDS = [
    r"how many",
    r"count",
    r"fold change",
    r"fold-change",
    r"greater than",
    r"less than",
    r"above",
    r"below",
    r"list.*protein",
    r"list.*site",
    r"p-value",
    r"p value",
    r"fdr",
    r"significant",
    r"number of",
    r"total.*site",
    r"total.*protein",
]

_SQL_RE = re.compile("|".join(SQL_KEYWORDS), re.IGNORECASE)


def is_sql_query(query: str) -> bool:
    """Heuristic check: does the query look like it needs SQL?"""
    return bool(_SQL_RE.search(query))


class SQLRouter:
    """Routes structured questions to DuckDB for exact answers."""

    def __init__(self, db_path: Path = DUCKDB_PATH):
        self.db_path = db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            if self.db_path.exists():
                self._conn = duckdb.connect(str(self.db_path))
            else:
                self._conn = duckdb.connect()
        return self._conn

    def execute(self, query: str) -> dict:
        """Route a structured query to DuckDB.

        Returns:
            {"route": "sql", "answer": str, "results": list}
        """
        # Parse natural language into SQL
        sql = self._nl_to_sql(query)
        if sql is None:
            return {"route": "rag", "answer": "", "results": []}

        try:
            rows = self.conn.execute(sql).fetchall()
            columns = [desc[0] for desc in self.conn.description]
            results = [dict(zip(columns, row)) for row in rows]
            answer = self._format_results(query, results)
            return {"route": "sql", "answer": answer, "results": results}
        except Exception as e:
            logger.warning(f"SQL execution failed: {e}")
            return {"route": "rag", "answer": "", "results": []}

    def _nl_to_sql(self, query: str) -> str | None:
        """Convert natural language query to SQL.

        This is a simplified rule-based translator. For a production system,
        you'd use an LLM or Text2SQL model.
        """
        query_lower = query.lower()

        # "How many SNO sites have fold change greater than X?"
        m = re.search(
            r"how many.*(?:site|protein).*(?:greater than|above|>\s*)(\d+\.?\d*)",
            query_lower,
        )
        if m:
            threshold = m.group(1)
            return f"SELECT COUNT(*) FROM ptm_sites WHERE fold_change > {threshold}"

        # "List proteins with fold change above X"
        m = re.search(
            r"(?:list|show).*protein.*(?:greater than|above|>\s*)(\d+\.?\d*)",
            query_lower,
        )
        if m:
            threshold = m.group(1)
            return f"SELECT DISTINCT protein_name FROM ptm_sites WHERE fold_change > {threshold}"

        # "What papers about S-nitrosylation?"
        m = re.search(r"(?:paper|article|literature).*?(s-nitrosylation|methylation|oxidation)", query_lower)
        if m:
            ptm = m.group(1)
            return f"SELECT pmid, title, year FROM papers WHERE ptm_type LIKE '%{ptm}%'"

        # "How many papers?"
        if "how many" in query_lower and ("paper" in query_lower or "article" in query_lower):
            return "SELECT COUNT(*) FROM papers"

        return None

    def _format_results(self, query: str, results: list[dict]) -> str:
        """Format SQL results into a natural language answer."""
        if not results:
            return "No matching records found."

        # Single count result
        if len(results) == 1 and len(results[0]) == 1:
            val = list(results[0].values())[0]
            return f"{val}"

        # Multiple rows — format as table
        lines = [f"Found {len(results)} results:"]
        for row in results[:20]:  # Limit display
            parts = [f"{k}={v}" for k, v in row.items()]
            lines.append(" | ".join(parts))
        if len(results) > 20:
            lines.append(f"... and {len(results) - 20} more")
        return "\n".join(lines)

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
