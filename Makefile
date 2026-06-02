.PHONY: help install build-index test eval dev clean

help:
	@echo "ProteoRAG — Makefile targets:"
	@echo "  install       Install dependencies (uv + pip)"
	@echo "  build-index   Build FAISS + BM25 indexes"
	@echo "  test          Run pytest"
	@echo "  eval          Run evaluation harness"
	@echo "  dev           Launch Streamlit dev server"
	@echo "  clean         Remove generated indexes and caches"

install:
	uv venv --python 3.11 && . .venv/bin/activate && uv pip install -r requirements.txt

build-index:
	PYTHONPATH=src python scripts/02_build_index.py

test:
	PYTHONPATH=src pytest tests/ -v

eval:
	PYTHONPATH=src python scripts/03_run_eval.py

dev:
	streamlit run app.py --server.port 8501

clean:
	rm -rf data/index/*.faiss data/index/*.pkl data/index/*.duckdb*
	rm -rf data/index/*.npy data/index/chunk_store.json
	rm -rf __pycache__ src/**/__pycache__ .pytest_cache .ruff_cache
	rm -rf .venv/
