from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GroundTruthChunk:
    file_path: str
    start_line: int
    end_line: int


@dataclass
class Query:
    id: str
    text: str
    ground_truth: List[GroundTruthChunk] = field(default_factory=list)


@dataclass
class RepoSpec:
    url: str
    commit: Optional[str] = None


@dataclass
class Dataset:
    name: str
    repo: RepoSpec
    queries: List[Query]
    top_k: int = 10

