from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .io_utils import canonical_json_hash
from .validation import require, require_finite

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

EXPECTED_MODEL40_ID = 40
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


def load_model40_metadata(meta_path: Path | str) -> dict:
    """Load and validate the NLPL Model 40 metadata shipped with the repository."""
    path = Path(meta_path)
    require(path.exists(), "MODEL40_META_EXISTS", f"Missing Model 40 metadata: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))

    require(int(meta.get("id")) == EXPECTED_MODEL40_ID, "MODEL40_META_ID", "Expected NLPL model id 40")
    require(
        int(meta.get("vocabulary size")) == EXPECTED_MODEL40_VOCAB,
        "MODEL40_META_VOCAB",
        f"Expected vocabulary {EXPECTED_MODEL40_VOCAB}",
    )
    require(int(meta.get("dimensions")) == EXPECTED_DIM, "MODEL40_META_DIM", f"Expected dimension {EXPECTED_DIM}")

    contents = {str(item.get("filename")) for item in meta.get("contents", []) if isinstance(item, dict)}
    require("model.bin" in contents, "MODEL40_META_BIN", "Model 40 metadata does not declare model.bin")
    return meta


def _read_binary_word(raw) -> bytes:
    """Read one Word2Vec token, skipping record separators before the token."""
    ch = raw.read(1)
    while ch in (b" ", b"\n", b"\r", b"\t"):
        ch = raw.read(1)
    require(bool(ch), "MODEL40_EOF_WORD", "Unexpected EOF while reading Model 40 token")

    word = bytearray()
    while ch != b" ":
        word.extend(ch)
        ch = raw.read(1)
        require(bool(ch), "MODEL40_EOF_WORD", "Unexpected EOF while reading Model 40 token")
    return bytes(word)


def extract_required_vectors_from_bin(
    bin_path: Path | str,
    tokens: Iterable[str],
    expected_vocab: int = EXPECTED_MODEL40_VOCAB,
    expected_dim: int = EXPECTED_DIM,
) -> dict[str, np.ndarray]:
    """Stream Model 40's Word2Vec binary and retain only the requested vectors.

    This deliberately avoids loading the full multi-GB model into memory and avoids
    any precomputed embedding cache. The file header is validated before extraction.
    """
    path = Path(bin_path)
    require(path.exists(), "MODEL40_BIN_EXISTS", f"Missing Model 40 binary: {path}")
    require(path.stat().st_size > 0, "MODEL40_BIN_NONEMPTY", "Model 40 binary is empty")

    wanted = {str(token).casefold() for token in tokens}
    require(bool(wanted), "MODEL40_REQUIRED_TOKENS", "Required token set is empty")

    found: dict[str, np.ndarray] = {}
    with path.open("rb") as raw:
        header = raw.readline().decode("utf-8", errors="strict").strip().split()
        require(len(header) >= 2, "MODEL40_HEADER", "Invalid Word2Vec binary header")

        vocab, dim = int(header[0]), int(header[1])
        require(vocab == expected_vocab, "MODEL40_VOCAB", f"Expected vocab {expected_vocab}, got {vocab}")
        require(dim == expected_dim, "MODEL40_DIM", f"Expected dim {expected_dim}, got {dim}")

        vector_nbytes = 4 * dim
        for _ in range(vocab):
            word_bytes = _read_binary_word(raw)
            vector_bytes = raw.read(vector_nbytes)
            require(
                len(vector_bytes) == vector_nbytes,
                "MODEL40_EOF_VECTOR",
                "Unexpected EOF while reading Model 40 vector",
            )

            word = word_bytes.decode("utf-8", errors="ignore").casefold()
            if word in wanted and word not in found:
                vec = np.frombuffer(vector_bytes, dtype="<f4").copy()
                require(vec.shape == (dim,), "MODEL40_VECTOR_DIM", "Unexpected Model 40 vector dimension")
                require_finite(vec, "MODEL40_VECTOR_FINITE")
                found[word] = vec
                if len(found) == len(wanted):
                    break

    return found


def model40_coverage(required: Iterable[str], vectors: Mapping[str, np.ndarray]) -> dict:
    required_set = sorted({str(t).casefold() for t in required})
    found = sorted(set(required_set).intersection(vectors))
    missing = sorted(set(required_set).difference(vectors))
    return {
        "required_token_count": len(required_set),
        "found_token_count": len(found),
        "oov_required_tokens": missing,
        "coverage_ratio": (len(found) / len(required_set)) if required_set else 0.0,
    }


def mean_embedding(tokens: Iterable[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    toks = list(tokens)
    recognized = [t for t in toks if t in vectors]
    require(bool(recognized), "EMBEDDING_ZERO_COVERAGE", "No recognized tokens for text")
    mat = np.vstack([vectors[t] for t in recognized])
    vec = mat.mean(axis=0)
    require(vec.shape == (EXPECTED_DIM,), "MEAN_EMBEDDING_DIM", "Mean embedding must be 100D")
    require_finite(vec, "MEAN_EMBEDDING_FINITE")
    coverage = len(recognized) / len(toks) if toks else 0.0
    return vec, {
        "token_count": len(toks),
        "recognized_count": len(recognized),
        "coverage_ratio": coverage,
        "oov": [t for t in toks if t not in vectors],
    }


def embedding_matrix(texts: Iterable[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    rows, diagnostics = [], []
    for text in texts:
        toks = preprocess_tokens(text)
        vec, diag = mean_embedding(toks, vectors)
        rows.append(vec)
        diagnostics.append({"text": str(text), **diag})
    matrix = np.vstack(rows).astype(np.float32, copy=False)
    require(
        matrix.ndim == 2 and matrix.shape[1] == EXPECTED_DIM,
        "EMBEDDING_MATRIX_SHAPE",
        "Embedding matrix must have 100 columns",
    )
    require_finite(matrix, "EMBEDDING_MATRIX_FINITE")
    return matrix, diagnostics


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    require(
        a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1],
        "COSINE_SHAPE",
        "Cosine inputs must be 2D with matching dimensions",
    )
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
    require(
        np.all((scaled >= 0.0) & (scaled <= 1.0)),
        "SEMANTIC_RANGE",
        "Rescaled query scores must lie in [0,1]",
    )
    require(np.all((w >= 0.0) & (w <= 1.0)), "W_RANGE", "W must lie in [0,1]")
    return scaled, w
