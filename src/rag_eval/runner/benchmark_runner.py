from pathlib import Path

from rag_eval.interfaces import RAGSystem
from rag_eval.metrics.core import compute_metrics
from rag_eval.models.dataset import Dataset, GroundTruthChunk, Query
from rag_eval.models.results import EvaluationReport, QueryResult
from rag_eval.runner.dataset_loader import load_dataset, prepare_repo


class BenchmarkRunner:
    """Coordinates ingestion, querying, and scoring for a dataset."""

    def __init__(self, rag_system: RAGSystem, cache_dir: str | Path = ".rag_eval_cache") -> None:
        self.rag_system = rag_system
        self.cache_dir = Path(cache_dir)

    def run(
        self,
        dataset_path: str | Path,
        top_k: int | None = None,
        overlap_threshold: float = 0.5,
    ) -> EvaluationReport:
        dataset = load_dataset(dataset_path)
        repo_path = prepare_repo(dataset.repo, self.cache_dir)

        try:
            self.rag_system.clear()
        except NotImplementedError:
            pass

        self.rag_system.ingest(str(repo_path))

        k = top_k or dataset.top_k
        query_results = [self._run_single(q, k, overlap_threshold) for q in dataset.queries]
        aggregate = self._aggregate(query_results)
        return EvaluationReport(dataset=dataset, aggregate_metrics=aggregate, query_results=query_results)

    def _run_single(
        self, query: Query, top_k: int, overlap_threshold: float
    ) -> QueryResult:
        retrieved = self.rag_system.query(query.text, top_k=top_k)
        metrics = compute_metrics(retrieved, query.ground_truth, top_k, overlap_threshold)
        return QueryResult(query=query, retrieved=retrieved, metrics=metrics)

    def _aggregate(self, query_results: list[QueryResult]) -> dict:
        if not query_results:
            return {}

        keys = query_results[0].metrics.keys()
        sums = {k: 0.0 for k in keys}
        for result in query_results:
            for key, value in result.metrics.items():
                sums[key] += value
        return {key: sums[key] / len(query_results) for key in keys}

