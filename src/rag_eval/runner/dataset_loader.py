from __future__ import annotations

from pathlib import Path
from typing import Any

import git
import yaml

from rag_eval.models.dataset import Dataset, GroundTruthChunk, Query, RepoSpec


def _require(data: dict, key: str, context: str) -> Any:
    if key not in data:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return data[key]


def load_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = yaml.safe_load(dataset_path.read_text())
    name = _require(data, "name", "dataset")
    repo_data = _require(data, "repo", "dataset")
    repo = RepoSpec(url=_require(repo_data, "url", "repo"), commit=repo_data.get("commit"))

    queries_data = _require(data, "queries", "dataset")
    queries = []
    for q in queries_data:
        query_id = _require(q, "id", "query")
        text = _require(q, "text", f"query {query_id}")
        gt_chunks = [
            GroundTruthChunk(
                file_path=_require(gt, "file_path", f"query {query_id} ground_truth"),
                start_line=int(_require(gt, "start_line", f"query {query_id} ground_truth")),
                end_line=int(_require(gt, "end_line", f"query {query_id} ground_truth")),
            )
            for gt in q.get("ground_truth", [])
        ]
        queries.append(Query(id=query_id, text=text, ground_truth=gt_chunks))

    return Dataset(
        name=name,
        repo=repo,
        queries=queries,
        top_k=int(data.get("top_k", 10)),
    )


def prepare_repo(repo: RepoSpec, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_name = Path(repo.url.rstrip("/")).stem
    target_dir = cache_dir / repo_name

    if target_dir.exists():
        repo_obj = git.Repo(target_dir)
        repo_obj.remotes.origin.fetch()
    else:
        target_dir = Path(git.Repo.clone_from(repo.url, target_dir).working_tree_dir)
        repo_obj = git.Repo(target_dir)

    if repo.commit:
        repo_obj.git.checkout(repo.commit)
    else:
        repo_obj.remotes.origin.pull()

    return target_dir

