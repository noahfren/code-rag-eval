from abc import ABC, abstractmethod
from typing import List

from rag_eval.models.chunk import CodeChunk


class RAGSystem(ABC):
    """Interface that target RAG systems must implement."""

    @abstractmethod
    def ingest(self, repo_path: str) -> None:
        """Ingest a code repository for indexing."""
        raise NotImplementedError

    @abstractmethod
    def query(self, query: str, top_k: int = 10) -> List[CodeChunk]:
        """Query the RAG system and return ranked code chunks."""
        raise NotImplementedError

    def clear(self) -> None:
        """Optional: clear any existing index."""
        # Implementations can override; noop by default
        return None

