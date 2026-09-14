from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from .validation import require, require_finite


def stable_rank_indices(scores: Sequence[float], representative_ids: Sequence[int]) -> np.ndarray:
    s = np.asarray(scores, dtype=float); ids = np.asarray(representative_ids, dtype=int)
    require(len(s) == len(ids), "RANK_LENGTH", "scores and IDs must have equal length")
    require_finite(s, "RANK_FINITE")
    return np.lexsort((ids, -s))


def ranked_ids(scores: Sequence[float], representative_ids: Sequence[int]) -> list[int]:
    order = stable_rank_indices(scores, representative_ids); ids = np.asarray(representative_ids, dtype=int)
    return [int(x) for x in ids[order]]


def top_k_ids(scores: Sequence[float], representative_ids: Sequence[int], k: int) -> list[int]:
    require(1 <= k <= len(scores), "TOPK_RANGE", "k outside ranking size")
    return ranked_ids(scores, representative_ids)[:k]


def ndcg_at_k(reference_relevance: Sequence[float], predicted_scores: Sequence[float], k: int) -> float:
    y_true = np.asarray(reference_relevance, dtype=float); y_score = np.asarray(predicted_scores, dtype=float)
    require(len(y_true) == len(y_score), "NDCG_LENGTH", "reference and predicted arrays must align")
    require(1 <= k <= len(y_true), "NDCG_K", "Invalid NDCG cutoff")
    require_finite(y_true, "NDCG_TRUE_FINITE"); require_finite(y_score, "NDCG_SCORE_FINITE")
    return float(ndcg_score(y_true.reshape(1,-1), y_score.reshape(1,-1), k=k))


def spearman_score(reference_scores: Sequence[float], predicted_scores: Sequence[float]) -> float:
    return float(spearmanr(reference_scores, predicted_scores).statistic)


def pairwise_order_accuracy(reference_scores: Sequence[float], predicted_scores: Sequence[float], representative_ids: Sequence[int]) -> float:
    ref=np.asarray(reference_scores,float); pred=np.asarray(predicted_scores,float); ids=np.asarray(representative_ids,int)
    require(len(ref)==len(pred)==len(ids), "PAIRWISE_LENGTH", "inputs must align")
    correct=total=0
    for i in range(len(ref)):
        for j in range(i+1,len(ref)):
            if np.isclose(ref[i],ref[j],atol=1e-12,rtol=0): continue
            ref_before=ref[i]>ref[j]
            pred_before=(ids[i]<ids[j]) if np.isclose(pred[i],pred[j],atol=1e-12,rtol=0) else (pred[i]>pred[j])
            correct += int(ref_before==pred_before); total += 1
    return float(correct/total) if total else 1.0


def membership_change_count(scores_a: Sequence[float], scores_b: Sequence[float], representative_ids: Sequence[int], k: int) -> int:
    a=set(top_k_ids(scores_a,representative_ids,k)); b=set(top_k_ids(scores_b,representative_ids,k))
    return len(a.symmetric_difference(b))//2
