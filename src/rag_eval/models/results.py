from dataclasses import dataclass

from .chunk import CodeChunk
from .dataset import Dataset, Query


@dataclass
class QueryResult:
    query: Query
    retrieved: list[CodeChunk]
    metrics: dict[str, float]


@dataclass
class EvaluationReport:
    dataset: Dataset
    aggregate_metrics: dict[str, float]
    query_results: list[QueryResult]

