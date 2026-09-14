from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .ranking import ndcg_at_k, pairwise_order_accuracy, ranked_ids, spearman_score
from .validation import require, require_finite


@dataclass(frozen=True)
class FeedbackAssignment:
    scores: np.ndarray
    reviewed_ids: tuple[int, ...]
    management_order: tuple[int, ...]
    no_op: bool


def validate_management_order(management_order: Sequence[int], frozen_top17: Sequence[int]) -> tuple[int, ...]:
    order = tuple(int(x) for x in management_order)
    frozen = tuple(int(x) for x in frozen_top17)
    require(len(order) == 17, "MGMT_ORDER_LENGTH", "Management order must contain exactly 17 candidates")
    require(len(set(order)) == 17, "MGMT_ORDER_UNIQUE", "Management order IDs must be unique")
    require(set(order) == set(frozen), "MGMT_ORDER_SET", "Management order must equal the frozen baseline top17 set")
    return order


def restricted_management_order(management_top17_order: Sequence[int], reviewed_ids: Sequence[int]) -> tuple[int, ...]:
    reviewed = {int(x) for x in reviewed_ids}
    restricted = tuple(int(x) for x in management_top17_order if int(x) in reviewed)
    require(len(restricted) == len(reviewed), "MGMT_RESTRICT_LENGTH", "Restricted order does not cover reviewed set")
    require(set(restricted) == reviewed, "MGMT_RESTRICT_SET", "Restricted management order set mismatch")
    return restricted


def relative_order_is_locked(scores: Sequence[float], ids: Sequence[int], locked_order: Sequence[int]) -> bool:
    locked = [int(x) for x in locked_order]
    if len(locked) <= 1:
        return True
    rank = ranked_ids(scores, ids)
    return [x for x in rank if x in set(locked)] == locked


def assign_reviewed_score_multiset(
    current_scores: Sequence[float],
    representative_ids: Sequence[int],
    reviewed_ids: Sequence[int],
    management_order: Sequence[int],
    locked_relative_order: Sequence[int] = (),
) -> FeedbackAssignment:
    """Reassign only the reviewed candidates' current score multiset to management order.

    Candidates outside the reviewed set are unchanged. Previously reviewed candidates may
    receive different numeric scores at a deeper feedback depth, but their locked relative
    management order must remain valid.
    """
    scores = np.asarray(current_scores, dtype=float)
    ids = np.asarray(representative_ids, dtype=int)
    require(len(scores) == len(ids), "FEEDBACK_LENGTH", "Scores and IDs must align")
    require(len(np.unique(ids)) == len(ids), "FEEDBACK_ID_UNIQUE", "IDs must be unique")
    require_finite(scores, "FEEDBACK_SCORES_FINITE")

    reviewed = tuple(int(x) for x in reviewed_ids)
    order = tuple(int(x) for x in management_order)
    require(len(set(reviewed)) == len(reviewed), "REVIEWED_UNIQUE", "Reviewed IDs must be unique")
    require(set(order) == set(reviewed) and len(order) == len(reviewed), "FEEDBACK_ORDER_SET", "Management order must equal reviewed set")
    id_to_idx = {int(cid): i for i, cid in enumerate(ids)}
    require(all(cid in id_to_idx for cid in reviewed), "FEEDBACK_REVIEWED_MEMBER", "Reviewed ID missing from candidate universe")

    before = scores.copy()
    reviewed_values = sorted((float(scores[id_to_idx[cid]]) for cid in reviewed), reverse=True)
    after = scores.copy()
    for cid, value in zip(order, reviewed_values):
        after[id_to_idx[cid]] = value

    outside_mask = np.array([int(cid) not in set(reviewed) for cid in ids], dtype=bool)
    require(np.array_equal(after[outside_mask], before[outside_mask]), "FEEDBACK_OUTSIDE_UNCHANGED", "Outside reviewed set changed")
    require(np.allclose(np.sort(after), np.sort(before), atol=1e-12, rtol=0.0), "FEEDBACK_MULTISET", "Global score multiset changed")
    if locked_relative_order:
        require(relative_order_is_locked(after, ids, locked_relative_order), "FEEDBACK_LOCK_ORDER", "Previously locked relative order was violated")
    no_op = bool(np.allclose(after, before, atol=1e-12, rtol=0.0))
    return FeedbackAssignment(after, reviewed, order, no_op)


def build_depth_assignment(
    base_scores: Sequence[float],
    representative_ids: Sequence[int],
    frozen_top17: Sequence[int],
    management_top17_order: Sequence[int],
    reviewed_k: int,
    previous_locked_order: Sequence[int] = (),
) -> FeedbackAssignment:
    require(reviewed_k in {4, 7, 17}, "FEEDBACK_K", "reviewed_k must be one of 4,7,17")
    top17 = tuple(int(x) for x in frozen_top17)
    require(len(top17) == 17, "FROZEN_TOP17_LENGTH", "Frozen top17 must contain 17 IDs")
    management = validate_management_order(management_top17_order, top17)
    reviewed = top17[:reviewed_k]
    restricted = restricted_management_order(management, reviewed)
    return assign_reviewed_score_multiset(
        base_scores,
        representative_ids,
        reviewed,
        restricted,
        locked_relative_order=previous_locked_order,
    )


def feedback_metrics(
    target_scores: Sequence[float],
    predicted_scores: Sequence[float],
    representative_ids: Sequence[int],
    reviewed_ids: Sequence[int],
    ndcg_k: int,
) -> dict[str, float]:
    target = np.asarray(target_scores, dtype=float)
    pred = np.asarray(predicted_scores, dtype=float)
    ids = np.asarray(representative_ids, dtype=int)
    require(len(target) == len(pred) == len(ids), "FEEDBACK_METRIC_LENGTH", "Inputs must align")
    require_finite(target, "FEEDBACK_TARGET_FINITE")
    require_finite(pred, "FEEDBACK_PRED_FINITE")

    reviewed = {int(x) for x in reviewed_ids}
    target_top = set(ranked_ids(target, ids)[:len(reviewed)])
    pred_top = set(ranked_ids(pred, ids)[:len(reviewed)])
    agreement = len(target_top & pred_top) / len(reviewed) if reviewed else 1.0
    mse = float(np.mean((target - pred) ** 2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "ndcg": ndcg_at_k(target, pred, ndcg_k),
        "spearman": spearman_score(target, pred),
        "pairwise_accuracy": pairwise_order_accuracy(target, pred, ids),
        "reviewed_set_agreement": float(agreement),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(target - pred))),
        "r2": 0.0 if denominator == 0 else float(1.0 - np.sum((target - pred) ** 2) / denominator),
    }


def best_stage(trajectory: pd.DataFrame, metric_column: str = "ndcg") -> int:
    require("stage" in trajectory.columns and metric_column in trajectory.columns, "BEST_STAGE_COLUMNS", "trajectory missing required columns")
    max_value = float(trajectory[metric_column].max())
    tied = trajectory[np.isclose(trajectory[metric_column].astype(float), max_value, atol=1e-12, rtol=0.0)]
    require(len(tied) > 0, "BEST_STAGE_EMPTY", "No best stage found")
    return int(tied["stage"].min())


def effort_summary(
    review_percentage: float,
    reviewed_candidates: int,
    total_actions: int,
    effective_actions: int,
    ndcg_stage0: float,
    ndcg_final: float,
) -> dict[str, float | int | None]:
    require(total_actions >= effective_actions >= 0, "EFFORT_ACTION_COUNTS", "Invalid action counts")
    gain = float(ndcg_final - ndcg_stage0)
    return {
        "review_percentage": float(review_percentage),
        "reviewed_candidates": int(reviewed_candidates),
        "total_actions": int(total_actions),
        "effective_actions": int(effective_actions),
        "no_op_actions": int(total_actions - effective_actions),
        "ndcg_gain_per_effective_action": None if effective_actions == 0 else float(gain / effective_actions),
    }
