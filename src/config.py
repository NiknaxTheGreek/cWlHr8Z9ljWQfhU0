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
H_MAP = {
    "direct": HR_SCORE_DIRECT,
    "adjacent": HR_SCORE_ADJACENT,
    "irrelevant": HR_SCORE_IRRELEVANT,
}

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


def project_root(start: Path | None = None) -> Path:
    """Reviewer-facing alias used by the authoritative notebook."""
    return find_project_root(start)


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


def get_model_spec(root: Path | str | None = None) -> dict:
    """Return the locked modeling contract in notebook-friendly form.

    The source modules retain ProjectConfig as their typed internal configuration,
    while the reviewer-facing notebook uses this explicit dictionary for display,
    audit calculations, and serialization.
    """
    resolved_root = find_project_root(Path(root) if root is not None else None)
    cfg = ProjectConfig.from_root(resolved_root)
    return {
        "seed": int(cfg.seed),
        "pca_variance": float(cfg.pca_variance),
        "pca_solver": str(cfg.pca_solver),
        "outer_splits": int(cfg.outer_splits),
        "outer_repeats": int(cfg.outer_repeats),
        "inner_splits": int(cfg.inner_splits),
        "alpha_grid": np.asarray(cfg.ridge_alphas, dtype=float),
        "ndcg_cutoffs": tuple(int(k) for k in NDCG_CUTOFFS),
        "tuning_objective": "mse",
    }
