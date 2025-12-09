import math
from typing import Sequence

from rag_eval.models.chunk import CodeChunk
from rag_eval.models.dataset import GroundTruthChunk


def _line_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def chunk_overlap(gt: GroundTruthChunk, retrieved: CodeChunk) -> int:
    if gt.file_path != retrieved.file_path:
        return 0
    return _line_overlap(gt.start_line, gt.end_line, retrieved.start_line, retrieved.end_line)


def chunk_matches(
    gt: GroundTruthChunk, retrieved: CodeChunk, overlap_threshold: float = 0.5
) -> bool:
    overlap = chunk_overlap(gt, retrieved)
    if overlap == 0:
        return False
    gt_len = gt.end_line - gt.start_line + 1
    retrieved_len = retrieved.end_line - retrieved.start_line + 1
    shorter = min(gt_len, retrieved_len)
    return (overlap / shorter) >= overlap_threshold


def _first_match_index(
    chunk: CodeChunk, ground_truth: Sequence[GroundTruthChunk], overlap_threshold: float
) -> int | None:
    for idx, gt in enumerate(ground_truth):
        if chunk_matches(gt, chunk, overlap_threshold):
            return idx
    return None


def precision_at_k(
    retrieved: Sequence[CodeChunk],
    ground_truth: Sequence[GroundTruthChunk],
    k: int = 10,
    overlap_threshold: float = 0.5,
) -> float:
    if not retrieved:
        return 0.0
    considered = list(retrieved[:k])
    hits = sum(
        1
        for chunk in considered
        if _first_match_index(chunk, ground_truth, overlap_threshold) is not None
    )
    return hits / len(considered)


def recall_at_k(
    retrieved: Sequence[CodeChunk],
    ground_truth: Sequence[GroundTruthChunk],
    k: int = 10,
    overlap_threshold: float = 0.5,
) -> float:
    if not ground_truth:
        return 1.0
    matched_gt = set()
    for chunk in retrieved[:k]:
        match_idx = _first_match_index(chunk, ground_truth, overlap_threshold)
        if match_idx is not None:
            matched_gt.add(match_idx)
    return len(matched_gt) / len(ground_truth)


def mrr(
    retrieved: Sequence[CodeChunk],
    ground_truth: Sequence[GroundTruthChunk],
    k: int = 10,
    overlap_threshold: float = 0.5,
) -> float:
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if _first_match_index(chunk, ground_truth, overlap_threshold) is not None:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[CodeChunk],
    ground_truth: Sequence[GroundTruthChunk],
    k: int = 10,
    overlap_threshold: float = 0.5,
) -> float:
    relevance = [
        1 if _first_match_index(chunk, ground_truth, overlap_threshold) is not None else 0
        for chunk in retrieved[:k]
    ]
    if not relevance:
        return 0.0
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevance))
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_metrics(
    retrieved: Sequence[CodeChunk],
    ground_truth: Sequence[GroundTruthChunk],
    k: int,
    overlap_threshold: float = 0.5,
) -> dict:
    return {
        "precision@k": precision_at_k(retrieved, ground_truth, k, overlap_threshold),
        "recall@k": recall_at_k(retrieved, ground_truth, k, overlap_threshold),
        "mrr": mrr(retrieved, ground_truth, k, overlap_threshold),
        "ndcg@k": ndcg_at_k(retrieved, ground_truth, k, overlap_threshold),
    }

