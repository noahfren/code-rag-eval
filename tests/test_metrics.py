from rag_eval.metrics.core import compute_metrics, precision_at_k, recall_at_k
from rag_eval.models import CodeChunk, GroundTruthChunk


def test_chunk_matching_and_metrics():
    ground_truth = [
        GroundTruthChunk(file_path="a.py", start_line=1, end_line=10),
        GroundTruthChunk(file_path="b.py", start_line=5, end_line=15),
    ]
    retrieved = [
        CodeChunk(file_path="a.py", start_line=1, end_line=5),
        CodeChunk(file_path="b.py", start_line=7, end_line=12),
        CodeChunk(file_path="c.py", start_line=1, end_line=3),
    ]

    assert precision_at_k(retrieved, ground_truth, k=2) == 1.0
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.5

    metrics = compute_metrics(retrieved, ground_truth, k=3)
    assert metrics["precision@k"] == 2 / 3
    assert metrics["recall@k"] == 1.0

