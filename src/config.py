from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

SEED = 42
QUERIES: Tuple[str, str] = (
    "aspiring human resources",
    "seeking human resources",
)
RIDGE_ALPHAS: Tuple[float, ...] = tuple(float(x) for x in np.logspace(-3, 2, 15))

HR_SCORE_DIRECT = 1.0
HR_SCORE_ADJACENT = 0.5
HR_SCORE_IRRELEVANT = 0.0

NDCG_CUTOFFS: Tuple[int, ...] = (4, 7, 10, 17, 34)
FEEDBACK_DEPTH_TO_K = {0.10: 4, 0.20: 7, 0.50: 17}
LOW_NONZERO_COVERAGE_WARN_RATIO = 0.50
MSE_TIE_ATOL = 1e-12
CLEAN_RUN_ATOL = 1e-10


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "project_control").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root from current path")


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    seed: int = SEED
    queries: Tuple[str, str] = QUERIES
    ridge_alphas: Tuple[float, ...] = RIDGE_ALPHAS
    pca_variance: float = 0.95
    pca_solver: str = "full"
    outer_splits: int = 5
    outer_repeats: int = 10
    inner_splits: int = 4
    low_nonzero_coverage_warn_ratio: float = LOW_NONZERO_COVERAGE_WARN_RATIO

    @classmethod
    def from_root(cls, root: Path | str) -> "ProjectConfig":
        return cls(root=Path(root).resolve())

    def serializable(self) -> dict:
        data = asdict(self)
        data["root"] = str(self.root)
        data["queries"] = list(self.queries)
        data["ridge_alphas"] = list(self.ridge_alphas)
        return data
