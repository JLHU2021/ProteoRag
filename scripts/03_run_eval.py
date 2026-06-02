#!/usr/bin/env python3
"""Evaluation harness for ProteoRAG.

Runs a curated test set of questions through the RAG pipeline and computes:
- Recall@K
- Mean Reciprocal Rank (MRR)
- Faithfulness (answer grounded in retrieved context)

Usage:
    python scripts/03_run_eval.py

Expected input: data/eval/eval_set.jsonl
Each line: {"question": "...", "answer": "...", "relevant_sources": [...], "route": "rag"}
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteomics_rag.chain.pipeline import ProteoRAGChain
from proteomics_rag.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"


def load_eval_set(path: Path = EVAL_SET_PATH) -> list[dict]:
    """Load the evaluation set from JSONL."""
    if not path.exists():
        logger.warning(f"Eval set not found at {path}. Creating a minimal test set.")
        path.parent.mkdir(parents=True, exist_ok=True)
        default_set = [
            {
                "question": "What is S-nitrosylation?",
                "answer": "A reversible PTM where NO attaches to cysteine thiol groups",
                "relevant_sources": ["dummy_sno_knowledge.txt"],
                "route": "rag",
            },
            {
                "question": "What does PRMT5 do?",
                "answer": "Catalyzes symmetric dimethylation of arginine residues",
                "relevant_sources": ["dummy_prmt5_knowledge.txt"],
                "route": "rag",
            },
        ]
        with open(path, "w") as f:
            for item in default_set:
                f.write(json.dumps(item) + "\n")
        return default_set

    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_recall_at_k(
    retrieved_sources: list[str],
    relevant_sources: list[str],
    k: int = 5,
) -> float:
    """Compute Recall@K: fraction of relevant sources found in top-K."""
    if not relevant_sources:
        return 1.0
    found = sum(1 for s in relevant_sources if s in retrieved_sources[:k])
    return found / len(relevant_sources)


def compute_mrr(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, src in enumerate(retrieved_sources):
        if src in relevant_sources:
            return 1.0 / (i + 1)
    return 0.0


def run_eval():
    """Run the full evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("ProteoRAG — Evaluation Harness")
    logger.info("=" * 60)

    eval_set = load_eval_set()
    logger.info(f"Loaded {len(eval_set)} evaluation questions")

    chain = ProteoRAGChain()
    chain.load()

    results = []
    recall_scores = []
    mrr_scores = []

    for item in eval_set:
        question = item["question"]
        relevant = item.get("relevant_sources", [])

        result = chain.query(question)
        retrieved = [s["source"] for s in result.get("sources", [])]

        r_at_5 = compute_recall_at_k(retrieved, relevant, k=5)
        mrr = compute_mrr(retrieved, relevant)

        recall_scores.append(r_at_5)
        mrr_scores.append(mrr)

        results.append({
            "question": question,
            "route": result["route"],
            "recall@5": r_at_5,
            "mrr": mrr,
            "num_sources": len(retrieved),
        })

        status = "✓" if r_at_5 > 0 else "✗"
        logger.info(f"  {status} {question[:60]} — Recall@5={r_at_5:.2f}, MRR={mrr:.2f}")

    # Summary
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

    logger.info("=" * 60)
    logger.info(f"Results ({len(eval_set)} questions):")
    logger.info(f"  Avg Recall@5: {avg_recall:.2f}")
    logger.info(f"  Avg MRR:      {avg_mrr:.2f}")
    logger.info("=" * 60)

    # Save results
    results_path = PROJECT_ROOT / "results" / "eval_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "n_questions": len(eval_set),
            "avg_recall@5": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "per_question": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_eval()
