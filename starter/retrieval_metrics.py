"""Retrieval evaluation metrics module (Phase 3: Task 3.4).

Provides standard IR and RAG retrieval evaluation metrics:
- MRR (Mean Reciprocal Rank)
- Hit@k (e.g. Hit@1, Hit@3, Hit@5)
- Recall@k
- Context Evidence Rate
"""
from typing import Any, Sequence


def calculate_mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.
    
    When averaged across all queries, this yields Mean Reciprocal Rank (MRR).
    Returns 1.0 / rank of the first relevant chunk found (1-based), or 0.0 if none found.
    """
    if not retrieved_ids or not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def calculate_hit_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> int:
    """Check if at least one gold chunk exists in the top k results (Hit@k).
    
    Returns 1 if hit, 0 otherwise.
    """
    if not retrieved_ids or not gold_ids or k <= 0:
        return 0
    top_k_ids = retrieved_ids[:k]
    gold_set = set(gold_ids)
    return 1 if any(cid in gold_set for cid in top_k_ids) else 0


def calculate_recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    """Calculate Recall@k: fraction of unique gold chunks retrieved in the top k.
    
    Returns a float in [0.0, 1.0]. If gold_ids is empty, returns 0.0 (or 1.0 if unanswerable with no chunks).
    """
    if not gold_ids:
        return 0.0
    if not retrieved_ids or k <= 0:
        return 0.0
    top_k_ids = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    hits = len(gold_set & top_k_ids)
    return hits / len(gold_set)


def calculate_context_evidence_rate(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k: int | None = None) -> float:
    """Calculate Context Evidence Rate: proportion of retrieved chunks that are true gold evidence.
    
    Also known as Evidence Precision / Context Density.
    Evaluates chunk signal-to-noise ratio in the provided context window.
    """
    selected_ids = retrieved_ids[:k] if (k is not None and k > 0) else list(retrieved_ids)
    if not selected_ids:
        return 0.0
    gold_set = set(gold_ids)
    evidence_count = sum(1 for cid in selected_ids if cid in gold_set)
    return evidence_count / len(selected_ids)


def evaluate_retrieval(retrieved_ids: Sequence[str], gold_ids: Sequence[str], k_values: Sequence[int] = (1, 3, 5)) -> dict[str, Any]:
    """Compute all standard retrieval metrics for a single query."""
    metrics: dict[str, Any] = {
        "mrr": calculate_mrr(retrieved_ids, gold_ids),
        "context_evidence_rate": calculate_context_evidence_rate(retrieved_ids, gold_ids),
    }
    for k in k_values:
        metrics[f"hit@{k}"] = calculate_hit_at_k(retrieved_ids, gold_ids, k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(retrieved_ids, gold_ids, k)
    return metrics
