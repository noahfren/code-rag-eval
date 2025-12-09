from dataclasses import dataclass
from typing import Dict, List

from .chunk import CodeChunk
from .dataset import Dataset, Query


@dataclass
class QueryResult:
    query: Query
    retrieved: List[CodeChunk]
    metrics: Dict[str, float]


@dataclass
class EvaluationReport:
    dataset: Dataset
    aggregate_metrics: Dict[str, float]
    query_results: List[QueryResult]

