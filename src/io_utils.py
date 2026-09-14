from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> str:
    p = Path(path)
    _atomic_replace_bytes(p, text.encode(encoding))
    return sha256_file(p)


def atomic_write_json(path: Path | str, value: Any) -> str:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    p = Path(path)
    digest = atomic_write_text(p, text)
    reloaded = json.loads(p.read_text(encoding="utf-8"))
    if reloaded != value:
        raise IOError(f"Post-write JSON reload mismatch for {p}")
    return digest


def atomic_write_csv(path: Path | str, frame: pd.DataFrame, **to_csv_kwargs: Any) -> str:
    p = Path(path)
    payload = frame.to_csv(index=False, **to_csv_kwargs).encode("utf-8")
    _atomic_replace_bytes(p, payload)
    reloaded = pd.read_csv(p)
    if list(reloaded.columns) != list(frame.columns) or len(reloaded) != len(frame):
        raise IOError(f"Post-write CSV structural mismatch for {p}")
    return sha256_file(p)
