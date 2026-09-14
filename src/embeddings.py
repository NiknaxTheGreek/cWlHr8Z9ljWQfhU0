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
    return canonical_json_hash({"token_regex":TOKEN_RE.pattern,"phone_regex":PHONE_RE.pattern,"year_regex":YEAR_RE.pattern,"drop_at":True,"drop_upper_single_char":True,"casefold":True,"possessive_strip":True})


def _resolve_member(zf: zipfile.ZipFile) -> str:
    candidates = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith((".txt", ".vec"))]
    require(bool(candidates), "MODEL40_MEMBER", "No text/vector member found in Model 40 archive")
    return max(candidates, key=lambda n: zf.getinfo(n).file_size)


def extract_required_vectors_from_zip(zip_path: Path | str, tokens: Iterable[str], expected_vocab: int = EXPECTED_MODEL40_VOCAB, expected_dim: int = EXPECTED_DIM) -> dict[str, np.ndarray]:
    wanted = set(tokens); found: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(zip_path) as zf:
        member = _resolve_member(zf)
        with zf.open(member, "r") as raw:
            header = raw.readline().decode("utf-8", errors="strict").strip().split()
            require(len(header) >= 2, "MODEL40_HEADER", "Invalid embedding header")
            vocab, dim = int(header[0]), int(header[1])
            require(vocab == expected_vocab, "MODEL40_VOCAB", f"Expected vocab {expected_vocab}, got {vocab}")
            require(dim == expected_dim, "MODEL40_DIM", f"Expected dim {expected_dim}, got {dim}")
            for line_b in raw:
                if len(found) == len(wanted): break
                parts = line_b.decode("utf-8", errors="ignore").rstrip().split()
                if len(parts) != dim + 1: continue
                word = parts[0].casefold()
                if word in wanted and word not in found:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                    require_finite(vec, "MODEL40_VECTOR_FINITE")
                    found[word] = vec
    return found


def save_compact_cache(npz_path: Path | str, metadata_path: Path | str, vectors: Mapping[str, np.ndarray], source_zip: Path | str, source_url: str, token_set: Iterable[str]) -> dict:
    tokens = sorted(vectors)
    matrix = np.vstack([np.asarray(vectors[t], dtype=np.float32) for t in tokens]) if tokens else np.empty((0, EXPECTED_DIM), np.float32)
    require(matrix.ndim == 2 and matrix.shape[1] == EXPECTED_DIM, "CACHE_DIM", "Cache vectors must be 100D")
    npz = Path(npz_path); npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, tokens=np.asarray(tokens), vectors=matrix)
    metadata = {"model":"NLPL model 40 English CoNLL17 Word2Vec","expected_vocab":EXPECTED_MODEL40_VOCAB,"dimension":EXPECTED_DIM,"source_url":source_url,"source_zip_sha256":sha256_file(source_zip),"preprocessing_hash":preprocessing_hash(),"required_token_set_hash":canonical_json_hash(sorted(set(token_set))),"cached_token_count":len(tokens),"cache_npz_sha256":sha256_file(npz)}
    atomic_write_json(metadata_path, metadata)
    return metadata


def load_compact_cache(npz_path: Path | str, metadata_path: Path | str, expected_token_set: Iterable[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    npz_path, metadata_path = Path(npz_path), Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["dimension"] == EXPECTED_DIM, "CACHE_METADATA_DIM", "Unexpected cache dimension")
    require(metadata["preprocessing_hash"] == preprocessing_hash(), "CACHE_PREPROCESSING_HASH", "Preprocessing hash mismatch")
    require(metadata["cache_npz_sha256"] == sha256_file(npz_path), "CACHE_FILE_HASH", "Cache file hash mismatch")
    if expected_token_set is not None:
        require(metadata["required_token_set_hash"] == canonical_json_hash(sorted(set(expected_token_set))), "CACHE_TOKEN_SET_HASH", "Required-token set hash mismatch")
    data = np.load(npz_path, allow_pickle=False)
    tokens = [str(x) for x in data["tokens"].tolist()]; matrix = np.asarray(data["vectors"], dtype=np.float32)
    require(matrix.shape == (len(tokens), EXPECTED_DIM), "CACHE_SHAPE", "Compact cache shape mismatch")
    require_finite(matrix, "CACHE_FINITE")
    return {t:matrix[i] for i,t in enumerate(tokens)}, metadata


def mean_embedding(tokens: Iterable[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    toks = list(tokens); recognized = [t for t in toks if t in vectors]
    require(bool(recognized), "EMBEDDING_ZERO_COVERAGE", "No recognized tokens for text")
    vec = np.vstack([vectors[t] for t in recognized]).mean(axis=0)
    require(vec.shape == (EXPECTED_DIM,), "MEAN_EMBEDDING_DIM", "Mean embedding must be 100D")
    require_finite(vec, "MEAN_EMBEDDING_FINITE")
    coverage = len(recognized) / len(toks) if toks else 0.0
    return vec, {"token_count":len(toks),"recognized_count":len(recognized),"coverage_ratio":coverage,"oov":[t for t in toks if t not in vectors]}
