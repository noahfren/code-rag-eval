from dataclasses import dataclass, field


@dataclass
class GroundTruthChunk:
    file_path: str
    start_line: int
    end_line: int


@dataclass
class Query:
    id: str
    text: str
    ground_truth: list[GroundTruthChunk] = field(default_factory=list)


@dataclass
class RepoSpec:
    url: str
    commit: str | None = None


@dataclass
class Dataset:
    name: str
    repo: RepoSpec
    queries: list[Query]
    top_k: int = 10

