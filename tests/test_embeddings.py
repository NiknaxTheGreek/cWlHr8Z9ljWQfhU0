from pathlib import Path

import numpy as np
import pytest

from src.embeddings import EXPECTED_DIM, mean_embedding, preprocess_tokens


def test_preprocess_shared_rules():
    tokens = preprocess_tokens("2019 Aspiring HR Professional at Acme (408) 709-2621")
    assert "2019" not in tokens
    assert "at" not in tokens
    assert "aspiring" in tokens
    assert "hr" in tokens


def test_mean_embedding_coverage():
    vectors = {"human": np.ones(EXPECTED_DIM, dtype=np.float32), "resources": np.full(EXPECTED_DIM, 3, dtype=np.float32)}
    vec, diag = mean_embedding(["human", "resources", "unknown"], vectors)
    assert vec.shape == (EXPECTED_DIM,)
    assert np.allclose(vec, 2.0)
    assert diag["recognized_count"] == 2
    assert diag["coverage_ratio"] == pytest.approx(2/3)


def test_mean_embedding_zero_coverage_fails():
    with pytest.raises(AssertionError):
        mean_embedding(["missing"], {})

from src.embeddings import cosine_matrix, embedding_matrix, semantic_scores, save_compact_cache_from_vectors, load_compact_cache, required_tokens


def test_embedding_matrix_and_semantic_scores():
    vectors = {
        "aspiring": np.ones(EXPECTED_DIM, dtype=np.float32),
        "human": np.full(EXPECTED_DIM, 2, dtype=np.float32),
        "resources": np.full(EXPECTED_DIM, 3, dtype=np.float32),
        "manager": np.linspace(0.1, 1.0, EXPECTED_DIM, dtype=np.float32),
    }
    X, diag = embedding_matrix(["Aspiring Human Resources Manager"], vectors)
    Q, _ = embedding_matrix(["aspiring human resources", "human resources"], vectors)
    scaled, w = semantic_scores(X, Q)
    assert X.shape == (1, EXPECTED_DIM)
    assert Q.shape == (2, EXPECTED_DIM)
    assert scaled.shape == (1, 2)
    assert w.shape == (1,)
    assert 0 <= w[0] <= 1
    assert diag[0]["recognized_count"] == 4


def test_recovered_cache_roundtrip(tmp_path):
    req = ["human", "resources", "missing"]
    vectors = {
        "human": np.ones(EXPECTED_DIM, dtype=np.float32),
        "resources": np.full(EXPECTED_DIM, 2, dtype=np.float32),
    }
    npz = tmp_path / "cache.npz"
    meta = tmp_path / "cache.json"
    md = save_compact_cache_from_vectors(npz, meta, vectors, req, {"method": "unit-test"})
    loaded, loaded_md = load_compact_cache(npz, meta, expected_token_set=req)
    assert set(loaded) == {"human", "resources"}
    assert md["oov_required_tokens"] == ["missing"]
    assert loaded_md["cache_npz_sha256"] == md["cache_npz_sha256"]


def test_cosine_matrix_requires_matching_dimensions():
    with pytest.raises(AssertionError):
        cosine_matrix(np.ones((2, 3)), np.ones((2, 4)))
