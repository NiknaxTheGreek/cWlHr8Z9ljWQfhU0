import pytest
from src.ranking import membership_change_count, ndcg_at_k, pairwise_order_accuracy, ranked_ids


def test_stable_tie_break_uses_ascending_id():
    assert ranked_ids([0.5,0.5,0.7],[9,2,5])==[5,2,9]


def test_ndcg_perfect_is_one():
    y=[3.0,2.0,1.0]; assert ndcg_at_k(y,y,3)==pytest.approx(1.0)


def test_pairwise_accuracy_perfect():
    ids=[1,2,3]; ref=[0.9,0.5,0.1]; assert pairwise_order_accuracy(ref,ref,ids)==pytest.approx(1.0)


def test_membership_change_count():
    ids=[1,2,3,4]; assert membership_change_count([4,3,2,1],[4,1,3,2],ids,2)==1
