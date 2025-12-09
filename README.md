# Codebase RAG Evaluation Harness

A small, Python-based harness for benchmarking codebase RAG systems against ground-truth datasets on public repositories.

Requires Python 3.10 or newer.

## Features
- Modular `RAGSystem` interface with ingestion and query hooks
- Dataset-driven evaluation (YAML) with repo pinning to commits
- IR metrics: Precision@K, Recall@K, MRR, NDCG with overlap-based chunk matching
- CLI for running benchmarks, listing datasets, and validation
- RAG System adapters (`adapters/simple_adapter.py`) for quick integration

## Quickstart
```bash
pip install -e .
rag-eval run --dataset datasets/sample-benchmark.yaml --adapter adapters.simple_adapter:SimpleGrepRAG
```

## Datasets
Datasets live under `datasets/` and pin to a specific repo/commit for reproducibility. See `datasets/sample-benchmark.yaml` for the schema.

## Benchmarking code-rag

To benchmark the [code-rag](https://github.com/noahfren/code-rag) vector search system:

### 1. Clone code-rag
```bash
git clone https://github.com/noahfren/code-rag.git
```

### 2. Install code-rag

```bash
pip install -e /path/to/code-rag
```

### 3. Run the benchmark

```bash
rag-eval run \
    --dataset datasets/sample-benchmark.yaml \
    --adapter examples.code_rag_adapter:CodeRAGAdapter \
    --fmt md \
    --output code-rag-report.md
```

> **Note:** Requires `OPENAI_API_KEY` to be set for embedding generation.

## Development
```bash
pip install -e .[dev]
pytest
```

