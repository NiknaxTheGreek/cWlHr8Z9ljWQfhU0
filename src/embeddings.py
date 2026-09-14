from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .io_utils import atomic_write_json, canonical_json_hash, sha256_file
from .validation import require, require_finite

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
EXPECTED_MODEL40_VOCAB = 4_027_169
EXPECTED_DIM = 100


def preprocess_tokens(text: str) -> list[str]:
    raw = PHONE_RE.sub(" ", str(text))
    raw = YEAR_RE.sub(" ", raw)
    tokens = TOKEN_RE.findall(raw)
    out = []
    for token in tokens:
        if token.casefold() == "at":
            continue
        if len(token) == 1 and token.isupper():
            continue
        norm = token.casefold().replace("’", "'")
        if norm.endswith("'s") and len(norm) > 2:
            norm = norm[:-2]
        if norm:
            out.append(norm)
    return out


def required_tokens(texts: Iterable[str]) -> list[str]:
    return sorted({t for text in texts for t in preprocess_tokens(text)})


def preprocessing_hash() -> str:
    payload = {
        "token_regex": TOKEN_RE.pattern,
        "phone_regex": PHONE_RE.pattern,
        "year_regex": YEAR_RE.pattern,
        "drop_at": True,
        "drop_upper_single_char": True,
        "casefold": True,
        "possessive_strip": True,
    }
    return canonical_json_hash(payload)


def _resolve_member(zf: zipfile.ZipFile) -> str:
    candidates = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith((".txt", ".vec"))]
    require(bool(candidates), "MODEL40_MEMBER", "No text/vector member found in Model 40 archive")
    return max(candidates, key=lambda n: zf.getinfo(n).file_size)


def extract_required_vectors_from_zip(zip_path: Path | str, tokens: Iterable[str], expected_vocab: int = EXPECTED_MODEL40_VOCAB,
                                      expected_dim: int = EXPECTED_DIM) -> dict[str, np.ndarray]:
    wanted = set(tokens)
    found: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as zf:
        member = _resolve_member(zf)
        with zf.open(member, "r") as raw:
            header = raw.readline().decode("utf-8", errors="strict").strip().split()
            require(len(header) >= 2, "MODEL40_HEADER", "Invalid embedding header")
            vocab, dim = int(header[0]), int(header[1])
            require(vocab == expected_vocab, "MODEL40_VOCAB", f"Expected vocab {expected_vocab}, got {vocab}")
            require(dim == expected_dim, "MODEL40_DIM", f"Expected dim {expected_dim}, got {dim}")
            for line_b in raw:
                if len(found) == len(wanted):
                    break
                line = line_b.decode("utf-8", errors="ignore").rstrip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != dim + 1:
                    continue
                word = parts[0].casefold()
                if word in wanted and word not in found:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                    require_finite(vec, "MODEL40_VECTOR_FINITE")
                    found[word] = vec
    return found


def save_compact_cache(npz_path: Path | str, metadata_path: Path | str, vectors: Mapping[str, np.ndarray], source_zip: Path | str,
                       source_url: str, token_set: Iterable[str]) -> dict:
    tokens = sorted(vectors)
    matrix = np.vstack([np.asarray(vectors[t], dtype=np.float32) for t in tokens]) if tokens else np.empty((0, EXPECTED_DIM), np.float32)
    require(matrix.ndim == 2 and matrix.shape[1] == EXPECTED_DIM, "CACHE_DIM", "Cache vectors must be 100D")
    npz = Path(npz_path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, tokens=np.asarray(tokens), vectors=matrix)
    metadata = {
        "model": "NLPL model 40 English CoNLL17 Word2Vec",
        "expected_vocab": EXPECTED_MODEL40_VOCAB,
        "dimension": EXPECTED_DIM,
        "source_url": source_url,
        "source_zip_sha256": sha256_file(source_zip),
        "preprocessing_hash": preprocessing_hash(),
        "required_token_set_hash": canonical_json_hash(sorted(set(token_set))),
        "cached_token_count": len(tokens),
        "cache_npz_sha256": sha256_file(npz),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def load_compact_cache(npz_path: Path | str, metadata_path: Path | str, expected_token_set: Iterable[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    npz_path, metadata_path = Path(npz_path), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["dimension"] == EXPECTED_DIM, "CACHE_METADATA_DIM", "Unexpected cache dimension")
    require(metadata["preprocessing_hash"] == preprocessing_hash(), "CACHE_PREPROCESSING_HASH", "Preprocessing hash mismatch")
    require(metadata["cache_npz_sha256"] == sha256_file(npz_path), "CACHE_FILE_HASH", "Cache file hash mismatch")
    if expected_token_set is not None:
        expected_hash = canonical_json_hash(sorted(set(expected_token_set)))
        require(metadata["required_token_set_hash"] == expected_hash, "CACHE_TOKEN_SET_HASH", "Required-token set hash mismatch")
    data = np.load(npz_path, allow_pickle=False)
    tokens = [str(x) for x in data["tokens"].tolist()]
    matrix = np.asarray(data["vectors"], dtype=np.float32)
    require(matrix.shape == (len(tokens), EXPECTED_DIM), "CACHE_SHAPE", "Compact cache shape mismatch")
    require_finite(matrix, "CACHE_FINITE")
    return {t: matrix[i] for i, t in enumerate(tokens)}, metadata


def mean_embedding(tokens: Iterable[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    toks = list(tokens)
    recognized = [t for t in toks if t in vectors]
    require(bool(recognized), "EMBEDDING_ZERO_COVERAGE", "No recognized tokens for text")
    mat = np.vstack([vectors[t] for t in recognized])
    vec = mat.mean(axis=0)
    require(vec.shape == (EXPECTED_DIM,), "MEAN_EMBEDDING_DIM", "Mean embedding must be 100D")
    require_finite(vec, "MEAN_EMBEDDING_FINITE")
    coverage = len(recognized) / len(toks) if toks else 0.0
    return vec, {"token_count": len(toks), "recognized_count": len(recognized), "coverage_ratio": coverage, "oov": [t for t in toks if t not in vectors]}


def save_compact_cache_from_vectors(npz_path: Path | str, metadata_path: Path | str,
                                    vectors: Mapping[str, np.ndarray], token_set: Iterable[str],
                                    provenance: Mapping[str, object]) -> dict:
    """Persist a validated compact cache from an already-validated vector mapping.

    This supports recovery/promotion when the full multi-GB source archive is intentionally
    not committed, while keeping the provenance chain explicit and hash-verifiable.
    """
    required = sorted(set(token_set))
    tokens = sorted(t for t in vectors if t in set(required))
    matrix = np.vstack([np.asarray(vectors[t], dtype=np.float32) for t in tokens]) if tokens else np.empty((0, EXPECTED_DIM), np.float32)
    require(matrix.shape == (len(tokens), EXPECTED_DIM), "CACHE_DIM", "Cache vectors must be 100D")
    require_finite(matrix, "CACHE_FINITE")
    npz = Path(npz_path)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, tokens=np.asarray(tokens), vectors=matrix)
    missing = sorted(set(required) - set(tokens))
    metadata = {
        "model": "NLPL model 40 English CoNLL17 Word2Vec",
        "model_id": 40,
        "repository_release": 20,
        "model_url": "https://vectors.nlpl.eu/repository/20/40.zip",
        "expected_vocab": EXPECTED_MODEL40_VOCAB,
        "dimension": EXPECTED_DIM,
        "preprocessing_hash": preprocessing_hash(),
        "required_token_set_hash": canonical_json_hash(required),
        "required_token_count": len(required),
        "cached_token_count": len(tokens),
        "oov_required_tokens": missing,
        "cache_npz_sha256": sha256_file(npz),
        "provenance": dict(provenance),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def embedding_matrix(texts: Iterable[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    rows, diagnostics = [], []
    for text in texts:
        toks = preprocess_tokens(text)
        vec, diag = mean_embedding(toks, vectors)
        rows.append(vec)
        diagnostics.append({"text": str(text), **diag})
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    require(matrix.ndim == 2 and matrix.shape[1] == EXPECTED_DIM, "EMBEDDING_MATRIX_SHAPE", "Embedding matrix must have 100 columns")
    require_finite(matrix, "EMBEDDING_MATRIX_FINITE")
    return matrix, diagnostics


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    require(a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1], "COSINE_SHAPE", "Cosine inputs must be 2D with matching dimensions")
    an = np.linalg.norm(a, axis=1, keepdims=True)
    bn = np.linalg.norm(b, axis=1, keepdims=True)
    require(np.all(an > 0) and np.all(bn > 0), "COSINE_NONZERO", "Cosine inputs must have non-zero norm")
    out = (a / an) @ (b / bn).T
    out = np.clip(out, -1.0, 1.0)
    require_finite(out, "COSINE_FINITE")
    return out


def semantic_scores(candidate_embeddings: np.ndarray, query_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos = cosine_matrix(candidate_embeddings, query_embeddings)
    require(cos.shape[1] == 2, "QUERY_COUNT", "Exactly two query embeddings are required")
    scaled = (cos + 1.0) / 2.0
    w = scaled.mean(axis=1)
    require(np.all((scaled >= 0.0) & (scaled <= 1.0)), "SEMANTIC_RANGE", "Rescaled query scores must lie in [0,1]")
    require(np.all((w >= 0.0) & (w <= 1.0)), "W_RANGE", "W must lie in [0,1]")
    return scaled, w
