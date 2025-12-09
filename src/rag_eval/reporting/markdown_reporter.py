from __future__ import annotations

from rag_eval.models.results import EvaluationReport, QueryResult


def _format_metrics(metrics: dict) -> str:
    lines = ["| metric | value |", "| --- | --- |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines)


def _format_ground_truth(result: QueryResult) -> str:
    if not result.query.ground_truth:
        return "_None provided_"
    items = [
        f"- `{gt.file_path}:{gt.start_line}-{gt.end_line}`"
        for gt in result.query.ground_truth
    ]
    return "\n".join(items)


def _format_retrieved(result: QueryResult, limit: int = 5) -> str:
    if not result.retrieved:
        return "_No results returned_"
    items = []
    for chunk in result.retrieved[:limit]:
        items.append(
            f"- `{chunk.file_path}:{chunk.start_line}-{chunk.end_line}`"
            + (f" (score={chunk.score:.3f})" if chunk.score is not None else "")
        )
    if len(result.retrieved) > limit:
        items.append(f"- ... {len(result.retrieved) - limit} more")
    return "\n".join(items)


def render_markdown(report: EvaluationReport) -> str:
    lines = [f"# RAG Evaluation: {report.dataset.name}", ""]

    lines.append("## Aggregate Metrics")
    lines.append(_format_metrics(report.aggregate_metrics) or "_No queries evaluated_")
    lines.append("")

    lines.append("## Per-Query Results")
    for result in report.query_results:
        lines.append(f"### {result.query.id} — {result.query.text}")
        lines.append(_format_metrics(result.metrics))
        lines.append("")
        lines.append("Ground truth:")
        lines.append(_format_ground_truth(result))
        lines.append("")
        lines.append("Top retrieved:")
        lines.append(_format_retrieved(result))
        lines.append("")

    return "\n".join(lines).strip() + "\n"

