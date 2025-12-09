"""Adapter for the code-rag system to work with rag-eval benchmarks.

Usage:
    python -m rag_eval run \
        --dataset datasets/sample-benchmark.yaml \
        --adapter examples.code_rag_adapter:CodeRAGAdapter

Requires code-rag to be installed:
    pip install -e /path/to/code-rag
"""

from code_rag.chunker import walk_codebase
from code_rag.embedder import Embedder
from code_rag.store import VectorStore

from rag_eval.interfaces import RAGSystem
from rag_eval.models import CodeChunk


class CodeRAGAdapter(RAGSystem):
    """Adapter that wraps code-rag's vector search for evaluation."""

    def __init__(
        self,
        collection_name: str = "rag_eval",
        data_dir: str = "./data/chroma_eval",
    ) -> None:
        """
        Initialize the adapter.

        Args:
            collection_name: ChromaDB collection name for this evaluation run.
            data_dir: Directory for ChromaDB storage.
        """
        self.collection_name = collection_name
        self.data_dir = data_dir
        self.embedder = Embedder()
        self.store = VectorStore(
            collection_name=collection_name,
            data_dir=data_dir,
        )

    def ingest(self, repo_path: str) -> None:
        """Ingest a repository by chunking, embedding, and storing."""
        # Clear any existing data for a clean evaluation
        self.store.clear()

        # Walk and chunk the codebase
        chunks = walk_codebase(repo_path)
        if not chunks:
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)

        # Store in vector database
        self.store.add_chunks(chunks, embeddings)

    def query(self, query: str, top_k: int = 10) -> list[CodeChunk]:
        """Query the vector store and return ranked code chunks."""
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search for similar chunks
        results = self.store.search(query_embedding, top_k=top_k)

        # Convert SearchResult objects to CodeChunk objects
        return [
            CodeChunk(
                file_path=result.relative_path,
                start_line=result.start_line,
                end_line=result.end_line,
                content=result.content,
                # Convert distance to similarity score (lower distance = higher similarity)
                score=1.0 - result.distance,
            )
            for result in results
        ]

    def clear(self) -> None:
        """Clear the vector store."""
        self.store.clear()
