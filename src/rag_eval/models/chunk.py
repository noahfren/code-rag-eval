from dataclasses import dataclass
from typing import Optional


@dataclass
class CodeChunk:
    """Represents a code snippet returned by a RAG system."""

    file_path: str
    start_line: int
    end_line: int
    content: str = ""
    score: Optional[float] = None

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")

