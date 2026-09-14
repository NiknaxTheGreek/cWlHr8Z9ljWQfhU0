import numpy as np
import pytest
from src.embeddings import EXPECTED_DIM, mean_embedding, preprocess_tokens


def test_preprocess_shared_rules():
    tokens=preprocess_tokens("2019 Aspiring HR Professional at Acme (408) 709-2621")
    assert "2019" not in tokens and "at" not in tokens and "aspiring" in tokens and "hr" in tokens


def test_mean_embedding_coverage():
    vectors={"human":np.ones(EXPECTED_DIM,dtype=np.float32),"resources":np.full(EXPECTED_DIM,3,dtype=np.float32)}
    vec,diag=mean_embedding(["human","resources","unknown"],vectors)
    assert vec.shape==(EXPECTED_DIM,) and np.allclose(vec,2.0)
    assert diag["recognized_count"]==2 and diag["coverage_ratio"]==pytest.approx(2/3)


def test_mean_embedding_zero_coverage_fails():
    with pytest.raises(AssertionError): mean_embedding(["missing"],{})
