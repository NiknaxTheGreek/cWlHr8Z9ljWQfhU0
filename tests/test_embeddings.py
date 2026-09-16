from pathlib import Path

import numpy as np
import pytest

from src.embeddings import (
    EXPECTED_DIM,
    cosine_matrix,
    embedding_matrix,
    extract_required_vectors_from_bin,
    load_model40_metadata,
    mean_embedding,
    model40_coverage,
    preprocess_tokens,
    semantic_scores,
)


def test_preprocess_shared_rules():
    tokens = preprocess_tokens("2019 Aspiring HR Professional at Acme (408) 709-2621")
    assert "2019" not in tokens
    assert "at" not in tokens
    assert "aspiring" in tokens
    assert "hr" in tokens


def test_mean_embedding_coverage():
    vectors = {
        "human": np.ones(EXPECTED_DIM, dtype=np.float32),
        "resources": np.full(EXPECTED_DIM, 3, dtype=np.float32),
    }
    vec, diag = mean_embedding(["human", "resources", "unknown"], vectors)
    assert vec.shape == (EXPECTED_DIM,)
    assert np.allclose(vec, 2.0)
    assert diag["recognized_count"] == 2
    assert diag["coverage_ratio"] == pytest.approx(2 / 3)


def test_mean_embedding_zero_coverage_fails():
    with pytest.raises(AssertionError):
        mean_embedding(["missing"], {})


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


def test_binary_word2vec_reader_streams_only_required_tokens(tmp_path):
    model = tmp_path / "model.bin"
    rows = [
        ("human", np.arange(EXPECTED_DIM, dtype=np.float32)),
        ("resources", np.arange(EXPECTED_DIM, dtype=np.float32) + 100),
        ("irrelevant", np.arange(EXPECTED_DIM, dtype=np.float32) + 200),
    ]
    with model.open("wb") as f:
        f.write(f"{len(rows)} {EXPECTED_DIM}\n".encode("utf-8"))
        for word, vector in rows:
            f.write(word.encode("utf-8") + b" ")
            f.write(vector.astype("<f4").tobytes())
            f.write(b"\n")

    vectors = extract_required_vectors_from_bin(
        model,
        ["human", "resources", "missing"],
        expected_vocab=len(rows),
        expected_dim=EXPECTED_DIM,
    )

    assert set(vectors) == {"human", "resources"}
    assert np.allclose(vectors["human"], rows[0][1])
    assert np.allclose(vectors["resources"], rows[1][1])
    coverage = model40_coverage(["human", "resources", "missing"], vectors)
    assert coverage["found_token_count"] == 2
    assert coverage["oov_required_tokens"] == ["missing"]


def test_model40_metadata_validation(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(
        """{
          "id": 40,
          "vocabulary size": 4027169,
          "dimensions": 100,
          "contents": [{"filename": "model.bin", "format": "data"}]
        }""",
        encoding="utf-8",
    )
    loaded = load_model40_metadata(meta)
    assert loaded["id"] == 40
    assert loaded["dimensions"] == 100


def test_cosine_matrix_requires_matching_dimensions():
    with pytest.raises(AssertionError):
        cosine_matrix(np.ones((2, 3)), np.ones((2, 4)))
