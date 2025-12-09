import json
from typing import Any

from rag_eval.models.results import EvaluationReport, QueryResult


def _chunk_to_dict(chunk) -> dict[str, Any]:
    return {
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "score": chunk.score,
        "content": chunk.content,
    }


def _query_result_to_dict(result: QueryResult) -> dict[str, Any]:
    return {
        "id": result.query.id,
        "text": result.query.text,
        "ground_truth": [
            {
                "file_path": gt.file_path,
                "start_line": gt.start_line,
                "end_line": gt.end_line,
            }
            for gt in result.query.ground_truth
        ],
        "retrieved": [_chunk_to_dict(chunk) for chunk in result.retrieved],
        "metrics": result.metrics,
    }


def render_json(report: EvaluationReport) -> str:
    payload = {
        "dataset": {
            "name": report.dataset.name,
            "repo": {
                "url": report.dataset.repo.url,
                "commit": report.dataset.repo.commit,
            },
            "top_k": report.dataset.top_k,
        },
        "aggregate_metrics": report.aggregate_metrics,
        "queries": [_query_result_to_dict(r) for r in report.query_results],
    }
    return json.dumps(payload, indent=2)

