import numpy as np
import pandas as pd
import pytest

from src.feedback import (
    assign_reviewed_score_multiset,
    best_stage,
    build_depth_assignment,
    effort_summary,
    restricted_management_order,
    validate_management_order,
)


def test_management_order_validation_and_restriction():
    frozen = list(range(1, 18))
    management = list(reversed(frozen))
    assert validate_management_order(management, frozen) == tuple(management)
    assert restricted_management_order(management, frozen[:4]) == (4, 3, 2, 1)
    with pytest.raises(AssertionError):
        validate_management_order(management[:-1], frozen)


def test_score_multiset_and_outside_are_preserved():
    ids = np.arange(1, 9)
    scores = np.array([.8, .7, .6, .5, .4, .3, .2, .1])
    out = assign_reviewed_score_multiset(scores, ids, [1, 2, 3, 4], [4, 3, 2, 1])
    assert np.allclose(np.sort(out.scores), np.sort(scores))
    assert np.allclose(out.scores[4:], scores[4:])
    assert out.scores[3] > out.scores[2] > out.scores[1] > out.scores[0]
    assert not out.no_op


def test_depth_assignment_uses_frozen_topk_sets():
    ids = np.arange(1, 21)
    scores = np.linspace(1.0, 0.05, 20)
    top17 = list(range(1, 18))
    management = [4, 3, 2, 1, 7, 6, 5, 10, 9, 8, 13, 12, 11, 17, 16, 15, 14]
    out = build_depth_assignment(scores, ids, top17, management, 4)
    assert set(out.reviewed_ids) == set(top17[:4])
    assert out.management_order == (4, 3, 2, 1)


def test_best_stage_earliest_tie_and_effort():
    trajectory = pd.DataFrame({"stage": [0, 1, 2, 3], "ndcg": [.5, .6, .6, .55]})
    assert best_stage(trajectory) == 1
    effort = effort_summary(.2, 7, total_actions=5, effective_actions=3, ndcg_stage0=.5, ndcg_final=.65)
    assert effort["no_op_actions"] == 2
    assert np.isclose(effort["ndcg_gain_per_effective_action"], .05)
