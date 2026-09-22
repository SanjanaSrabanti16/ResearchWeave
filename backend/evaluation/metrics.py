import math
from collections.abc import Sequence


def precision_at_k(relevances: Sequence[int], k: int, threshold: int = 2) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(value >= threshold for value in relevances[:k]) / k


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(relevances[:k], 1))


def ndcg_at_k(
    relevances: Sequence[int], k: int, ideal_relevances: Sequence[int] | None = None
) -> float:
    ideal_source = relevances if ideal_relevances is None else ideal_relevances
    ideal = dcg_at_k(sorted(ideal_source, reverse=True), k)
    return dcg_at_k(relevances, k) / ideal if ideal else 0.0


def reciprocal_rank(relevances: Sequence[int], threshold: int = 2) -> float:
    for rank, value in enumerate(relevances, 1):
        if value >= threshold:
            return 1.0 / rank
    return 0.0


def top_k_overlap(left: Sequence[str], right: Sequence[str], k: int = 20) -> dict[str, float | int]:
    if k <= 0:
        raise ValueError("k must be positive")
    left_set = set(left[:k])
    right_set = set(right[:k])
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "intersection_count": intersection,
        "overlap_fraction": intersection / k,
        "jaccard": intersection / union if union else 1.0,
    }
