"""
evaluation.py
=============
Evaluation metrics module — Faithfulness, Relevancy, Confidence, BLEU, ROUGE.
"""

import re
import numpy as np


def keyword_overlap(query: str, chunks: list) -> list:
    """Count keyword overlap between query and each chunk."""
    query_words = set(re.findall(r'\w+', query.lower()))
    overlaps = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        overlaps.append(len(query_words.intersection(chunk_words)))
    return overlaps


def faithfulness_score(answer: str, context: str) -> float:
    """
    Faithfulness — how much of the answer is supported by the context.
    Measures what fraction of answer words appear in context.
    """
    answer_words = set(re.findall(r'\w+', answer.lower()))
    context_words = set(re.findall(r'\w+', context.lower()))

    if not answer_words:
        return 0.0

    supported = answer_words.intersection(context_words)
    score = len(supported) / len(answer_words)
    return round(min(score, 1.0), 3)


def relevancy_score(query: str, retrieved_chunks: list) -> float:
    """
    Relevancy — how relevant are retrieved chunks to the query.
    Measures what fraction of query words appear in retrieved context.
    """
    query_words = set(re.findall(r'\w+', query.lower()))
    context = " ".join(retrieved_chunks)
    context_words = set(re.findall(r'\w+', context.lower()))

    if not query_words:
        return 0.0

    covered = query_words.intersection(context_words)
    score = len(covered) / len(query_words)
    return round(min(score, 1.0), 3)


def confidence_level(scores: list, overlaps: list, query: str) -> tuple:
    """
    Determine confidence level based on similarity scores and keyword overlap.

    Returns:
        (level, max_score) where level is 'low', 'medium', or 'high'
    """
    max_score = max(scores) if scores else 0
    max_overlap = max(overlaps) if overlaps else 0
    query_len = len(query.split())

    # STRICT LOW — very low score AND very low overlap
    if max_score < 0.35 and max_overlap == 0:
        return "low", max_score

    # SHORT QUERY FIX — "What is OS", "What is RAM" etc
    if query_len <= 4:
        if max_score >= 0.35:
            return "high", max_score
        else:
            return "medium", max_score

    # MEDIUM
    if max_score < 0.45 or max_overlap <= 1:
        return "medium", max_score

    return "high", max_score


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Simple BLEU-1 score between reference and hypothesis."""
    ref_words = set(reference.lower().split())
    hyp_words = hypothesis.lower().split()

    if not hyp_words:
        return 0.0

    matches = sum(1 for w in hyp_words if w in ref_words)
    return round(matches / len(hyp_words), 4)


def compute_rouge1(reference: str, hypothesis: str) -> float:
    """Simple ROUGE-1 F1 score."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    if not ref_words or not hyp_words:
        return 0.0

    overlap = ref_words.intersection(hyp_words)
    precision = len(overlap) / len(hyp_words)
    recall = len(overlap) / len(ref_words)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def full_evaluation(query: str, answer: str, context: str,
                    retrieved_chunks: list, scores: list) -> dict:
    """
    Run full evaluation and return all metrics.

    Returns:
        Dict with all metric scores
    """
    overlaps = keyword_overlap(query, retrieved_chunks)
    confidence, max_score = confidence_level(scores, overlaps, query)

    return {
        "confidence_level": confidence,
        "similarity_score": round(max_score, 4),
        "faithfulness": faithfulness_score(answer, context),
        "relevancy": relevancy_score(query, retrieved_chunks),
        "keyword_overlap": max(overlaps) if overlaps else 0,
        "avg_similarity": round(np.mean(scores), 4) if scores else 0,
    }