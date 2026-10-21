import math
import re


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    hits = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return hits / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    if not relevant_ids:
        return None

    retrieved = retrieved_ids[:k]

    hits = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved = retrieved_ids[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)

    if ideal_hits == 0:
        return None

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / ideal_dcg


def token_f1(reference, prediction):
    """Lexical overlap score, not semantic correctness."""

    tokenize = lambda text: re.findall(
        r"\b\w+\b", text.lower()
    )

    reference_tokens = tokenize(reference)
    prediction_tokens = tokenize(prediction)

    if not reference_tokens or not prediction_tokens:
        return 0.0

    from collections import Counter

    ref_counts = Counter(reference_tokens)
    pred_counts = Counter(prediction_tokens)

    common = sum((ref_counts & pred_counts).values())

    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)