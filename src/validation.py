from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


class Severity(str, Enum):
    HARD_FAIL = "HARD_FAIL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class Diagnostic:
    check_id: str
    severity: Severity
    status: str
    message: str
    evidence: Any = None
    affected_objects: tuple[str, ...] = ()
    remediation: str | None = None

    def serializable(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


def require(condition: bool, check_id: str, message: str) -> None:
    if not condition:
        raise AssertionError(f"[{check_id}] {message}")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], check_id: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    require(not missing, check_id, f"Missing required columns: {missing}")


def require_unique(frame: pd.DataFrame, column: str, check_id: str) -> None:
    require(frame[column].notna().all(), check_id, f"{column} contains nulls")
    require(frame[column].is_unique, check_id, f"{column} must be unique")


def require_finite(array: np.ndarray, check_id: str) -> None:
    require(bool(np.isfinite(np.asarray(array)).all()), check_id, "Array contains non-finite values")


def require_range(values: Iterable[float], low: float, high: float, check_id: str, atol: float = 1e-12) -> None:
    arr = np.asarray(list(values), dtype=float)
    require_finite(arr, check_id)
    require(bool(((arr >= low - atol) & (arr <= high + atol)).all()), check_id, f"Values outside [{low}, {high}]")
