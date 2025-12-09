from __future__ import annotations

from pathlib import Path
from typing import List

from rag_eval.interfaces import RAGSystem
from rag_eval.models import CodeChunk


class SimpleGrepRAG(RAGSystem):
    """A naive keyword search adapter for demonstration purposes."""

    def __init__(self) -> None:
        self.repo_path: Path | None = None
        self._files: list[tuple[Path, list[str]]] = []

    def ingest(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path)
        self._files.clear()

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            self._files.append((path, text.splitlines()))

    def query(self, query: str, top_k: int = 10) -> List[CodeChunk]:
        if self.repo_path is None:
            raise RuntimeError("ingest() must be called before query().")

        terms = [t.lower() for t in query.split() if t.strip()]
        if not terms:
            return []

        scored: list[tuple[float, CodeChunk]] = []
        for path, lines in self._files:
            lowered = [line.lower() for line in lines]
            score = sum(line.count(term) for term in terms for line in lowered)
            if score <= 0:
                continue

            start_idx = next(
                (idx for idx, line in enumerate(lowered) if any(term in line for term in terms)),
                0,
            )
            end_idx = min(start_idx + 20, len(lines))
            chunk = CodeChunk(
                file_path=str(path.relative_to(self.repo_path)),
                start_line=start_idx + 1,
                end_line=end_idx,
                content="\n".join(lines[start_idx:end_idx]),
                score=float(score),
            )
            scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

