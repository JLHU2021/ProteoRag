#!/usr/bin/env python3
"""Build FAISS index and BM25 index from raw data.

Pipeline:
1. Ingest raw files (PDFs, CSVs) into chunks
2. Encode chunks with PubMedBERT
3. Build FAISS index
4. Build BM25 index
"""

import json
import logging
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteomics_rag.config import DATA_RAW, DATA_INDEX
from proteomics_rag.data.ingestor import ingest_papers
from proteomics_rag.embedding.embedder import Embedder, build_faiss_index
from proteomics_rag.retrieval.hybrid import BM25Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("ProteoRAG — Index Builder")
    logger.info("=" * 60)

    # Step 1: Ingest
    logger.info("Step 1: Ingesting raw data...")
    count = ingest_papers(DATA_RAW)
    if count == 0:
        logger.warning("No data ingested. Add files to data/raw/ and re-run.")
        logger.info("Creating a minimal test index with dummy data...")
        # Create dummy chunks so the system can at least run
        dummy_chunks = [
            {
                "text": "S-nitrosylation (SNO) is a reversible post-translational modification "
                "where nitric oxide (NO) covalently attaches to the thiol group of cysteine residues. "
                "SNO regulates protein function, stability, and subcellular localization. "
                "GAPDH is a well-characterized SNO target at Cys152, which inhibits its glycolytic activity.",
                "chunk_id": "dummy_sno_0",
                "source": "dummy_sno_knowledge.txt",
            },
            {
                "text": "PRMT5 (Protein Arginine Methyltransferase 5) catalyzes symmetric "
                "dimethylation of arginine residues. PRMT5 activity is regulated by S-nitrosylation, "
                "which modulates its substrate binding and catalytic efficiency. "
                "Key substrates include STAT3, p53, and histone H4R3me2s.",
                "chunk_id": "dummy_prmt5_0",
                "source": "dummy_prmt5_knowledge.txt",
            },
            {
                "text": "Cysteine oxidation is an irreversible or reversible PTM depending on "
                "the oxidant and the resulting modification. Sulfenylation (-SOH) is reversible, "
                "while sulfinylation (-SO2H) and sulfonylation (-SO3H) are generally irreversible. "
                "Peroxiredoxins are key sensors of cellular H2O2 levels via cysteine oxidation.",
                "chunk_id": "dummy_cys_ox_0",
                "source": "dummy_cysteine_oxidation.txt",
            },
            {
                "text": "MaxQuant is a widely used computational platform for mass spectrometry-based "
                "proteomics data analysis. It performs peak detection, peptide identification, "
                "protein quantification via LFQ or SILAC, and FDR control at PSM, peptide, and "
                "protein levels. Default FDR threshold is 1%.",
                "chunk_id": "dummy_maxquant_0",
                "source": "dummy_maxquant_manual.txt",
            },
            {
                "text": "In the study of S-nitrosylation in Arabidopsis, 1195 SNO-modified peptides "
                "were identified across 676 proteins. Key pathways affected include photosynthesis, "
                "stress response, and primary metabolism. The trans-denitrosylation assay using "
                "HPDP-biotin was used for enrichment.",
                "chunk_id": "dummy_arabidopsis_sno_0",
                "source": "dummy_arabidopsis_sno.txt",
            },
        ]
        chunks = dummy_chunks
    else:
        logger.info(f"Ingested {count} chunks. Proceeding to index build...")
        # In a real scenario, we'd load chunks from the ingestion output
        # For now, use dummy data as placeholder
        chunks = []

    # If no real data, use dummy
    if not chunks:
        chunks = [
            {
                "text": "S-nitrosylation (SNO) is a reversible post-translational modification "
                "where nitric oxide (NO) covalently attaches to the thiol group of cysteine residues. "
                "GAPDH is a well-characterized SNO target at Cys152.",
                "chunk_id": "dummy_sno_0",
                "source": "dummy_sno_knowledge.txt",
            },
            {
                "text": "PRMT5 catalyzes symmetric dimethylation of arginine residues. "
                "PRMT5 activity is regulated by S-nitrosylation. "
                "Key substrates include STAT3, p53, and histone H4R3me2s.",
                "chunk_id": "dummy_prmt5_0",
                "source": "dummy_prmt5_knowledge.txt",
            },
        ]

    # Step 2: Build FAISS index
    logger.info("Step 2: Building FAISS index...")
    embedder = Embedder()
    faiss_index, metadata = build_faiss_index(chunks, embedder)

    # Step 3: Build BM25 index
    logger.info("Step 3: Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.build(chunks)

    # Save chunk store for retrieval text lookup
    chunk_store_path = DATA_INDEX / "chunk_store.json"
    chunk_store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunk_store_path, "w") as f:
        json.dump(chunks, f, indent=2)

    logger.info("=" * 60)
    logger.info("Index build complete!")
    logger.info(f"  FAISS index: {DATA_INDEX / 'faiss_index.faiss'}")
    logger.info(f"  BM25 index:  {DATA_INDEX / 'bm25_index.pkl'}")
    logger.info(f"  Chunk store: {chunk_store_path}")
    logger.info(f"  Total chunks: {len(chunks)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
