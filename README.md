# Codebase RAG Evaluation Harness

A small, Python-based harness for benchmarking codebase RAG systems against ground-truth datasets on public repositories.

## Features
- Modular `RAGSystem` interface with ingestion and query hooks
- Dataset-driven evaluation (YAML) with repo pinning to commits
- IR metrics: Precision@K, Recall@K, MRR, NDCG with overlap-based chunk matching
- CLI for running benchmarks, listing datasets, and validation
- Example adapter (`examples/simple_adapter.py`) for quick integration

## Quickstart
```bash
pip install -e .
rag-eval run --dataset datasets/sample-benchmark.yaml --adapter examples.simple_adapter:SimpleGrepRAG
```

## Datasets
Datasets live under `datasets/` and pin to a specific repo/commit for reproducibility. See `datasets/sample-benchmark.yaml` for the schema.

## Development
```bash
pip install -e .[dev]
pytest
```

