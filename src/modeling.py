from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MSE_TIE_ATOL, NDCG_CUTOFFS, ProjectConfig, find_project_root
from .ranking import evaluation_metrics, ndcg_at_k
from .validation import require, require_finite


@dataclass(frozen=True)
class AlphaSelection:
    alpha: float
    objective: str
    score: float
    diagnostics: pd.DataFrame


def ridge_pipeline(alpha: float, config: ProjectConfig) -> Pipeline:
    """Locked production pipeline: PCA(95%, full) -> StandardScaler -> Ridge."""
    return Pipeline([
        ("pca", PCA(n_components=config.pca_variance, svd_solver=config.pca_solver)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def _inner_splits(n_samples: int, config: ProjectConfig) -> KFold:
    require(n_samples >= config.inner_splits, "INNER_CV_SIZE", "Not enough samples for inner CV")
    return KFold(n_splits=config.inner_splits, shuffle=True, random_state=config.seed)


def _larger_alpha_on_tie(alphas: Sequence[float], scores: Sequence[float], maximize: bool) -> int:
    a = np.asarray(alphas, dtype=float)
    s = np.asarray(scores, dtype=float)
    require(len(a) == len(s) and len(a) > 0, "ALPHA_SCORE_LENGTH", "alpha and score arrays must align")
    require_finite(a, "ALPHA_FINITE")
    require_finite(s, "ALPHA_SCORE_FINITE")
    best = np.max(s) if maximize else np.min(s)
    tied = np.where(np.isclose(s, best, atol=MSE_TIE_ATOL, rtol=0.0))[0]
    return int(tied[np.argmax(a[tied])])


def select_alpha_mse(X: np.ndarray, y: np.ndarray, config: ProjectConfig) -> AlphaSelection:
    """Select alpha by shuffled 4-fold inner CV MSE; larger alpha wins numerical ties."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    require(X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "MSE_TUNE_SHAPE", "X/y shape mismatch")
    require_finite(X, "MSE_TUNE_X_FINITE")
    require_finite(y, "MSE_TUNE_Y_FINITE")
    kf = _inner_splits(len(y), config)
    rows, means = [], []
    for alpha in config.ridge_alphas:
        fold_mse = []
        for tr, va in kf.split(X):
            model = ridge_pipeline(alpha, config)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            fold_mse.append(float(mean_squared_error(y[va], pred)))
        mean_mse = float(np.mean(fold_mse))
        means.append(mean_mse)
        rows.append({
            "alpha": float(alpha),
            "mean_mse": mean_mse,
            "sd_mse": float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0,
        })
    idx = _larger_alpha_on_tie(config.ridge_alphas, means, maximize=False)
    return AlphaSelection(float(config.ridge_alphas[idx]), "mse", float(means[idx]), pd.DataFrame(rows))


def select_alpha_inner_oof_ndcg(X: np.ndarray, y: np.ndarray, config: ProjectConfig, k: int = 10) -> AlphaSelection:
    """Diagnostic-only alpha selection from one complete inner-OOF ranking per alpha."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    require(X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "NDCG_TUNE_SHAPE", "X/y shape mismatch")
    require_finite(X, "NDCG_TUNE_X_FINITE")
    require_finite(y, "NDCG_TUNE_Y_FINITE")
    kf = _inner_splits(len(y), config)
    rows, scores = [], []
    for alpha in config.ridge_alphas:
        oof = np.empty(len(y), dtype=float)
        assigned = np.zeros(len(y), dtype=int)
        for tr, va in kf.split(X):
            model = ridge_pipeline(alpha, config)
            model.fit(X[tr], y[tr])
            oof[va] = model.predict(X[va])
            assigned[va] += 1
        require(np.all(assigned == 1), "INNER_OOF_COMPLETE", "Each sample must receive exactly one inner-OOF prediction")
        cutoff = min(int(k), len(y))
        score = ndcg_at_k(y, oof, cutoff)
        scores.append(score)
        rows.append({"alpha": float(alpha), "inner_oof_ndcg": float(score), "k": cutoff})
    idx = _larger_alpha_on_tie(config.ridge_alphas, scores, maximize=True)
    return AlphaSelection(float(config.ridge_alphas[idx]), f"inner_oof_ndcg@{min(k, len(y))}", float(scores[idx]), pd.DataFrame(rows))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_repeated_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    representative_ids: Sequence[int],
    config: ProjectConfig,
    tuning_objective: str = "mse",
    diagnostic_ndcg_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Typed core repeated nested CV retained for direct source-level use."""
    oof, folds, repeats, _, _ = _repeated_nested_cv_impl(
        X=X,
        y=y,
        ids=representative_ids,
        config=config,
        tuning_objective=tuning_objective,
        diagnostic_ndcg_k=diagnostic_ndcg_k,
    )
    return oof, folds, repeats


def summarize_repeat_metrics(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    require(len(repeat_metrics) > 0, "REPEAT_SUMMARY_EMPTY", "No repeat metrics to summarize")
    rows = []
    for metric in [c for c in repeat_metrics.columns if c != "repeat"]:
        vals = repeat_metrics[metric].to_numpy(float)
        rows.append({
            "metric": metric,
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })
    return pd.DataFrame(rows)


def fit_final_model(X: np.ndarray, y: np.ndarray, config: ProjectConfig) -> tuple[Pipeline, AlphaSelection, np.ndarray]:
    selection = select_alpha_mse(X, y, config)
    model = ridge_pipeline(selection.alpha, config)
    model.fit(X, y)
    pred = np.asarray(model.predict(X), dtype=float)
    require_finite(pred, "FINAL_PRED_FINITE")
    return model, selection, pred


def _config_from_spec(model_spec: dict) -> ProjectConfig:
    """Convert the explicit notebook model specification to the typed source config."""
    return ProjectConfig(
        root=find_project_root(),
        seed=int(model_spec["seed"]),
        ridge_alphas=tuple(float(x) for x in model_spec["alpha_grid"]),
        pca_variance=float(model_spec["pca_variance"]),
        pca_solver=str(model_spec["pca_solver"]),
        outer_splits=int(model_spec["outer_splits"]),
        outer_repeats=int(model_spec["outer_repeats"]),
        inner_splits=int(model_spec["inner_splits"]),
    )


def _repeated_nested_cv_impl(
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[int],
    config: ProjectConfig,
    tuning_objective: str,
    diagnostic_ndcg_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, np.ndarray], pd.DataFrame]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ids = np.asarray(ids, dtype=int)
    require(X.ndim == 2 and y.ndim == 1 and len(X) == len(y) == len(ids), "NESTED_CV_SHAPE", "X/y/IDs must align")
    require(len(np.unique(ids)) == len(ids), "NESTED_CV_IDS", "Representative IDs must be unique")
    require_finite(X, "NESTED_CV_X_FINITE")
    require_finite(y, "NESTED_CV_Y_FINITE")
    require(tuning_objective in {"mse", "ndcg"}, "NESTED_CV_OBJECTIVE", "Unsupported tuning objective")

    outer = RepeatedKFold(
        n_splits=config.outer_splits,
        n_repeats=config.outer_repeats,
        random_state=config.seed,
    )

    pred_rows: list[dict] = []
    fold_rows: list[dict] = []
    tuning_rows: list[dict] = []

    for split_index, (tr, te) in enumerate(outer.split(X)):
        repeat = split_index // config.outer_splits + 1
        fold = split_index % config.outer_splits + 1
        selector = (
            select_alpha_mse(X[tr], y[tr], config)
            if tuning_objective == "mse"
            else select_alpha_inner_oof_ndcg(X[tr], y[tr], config, diagnostic_ndcg_k)
        )

        diagnostics = selector.diagnostics.copy()
        diagnostics.insert(0, "repeat", repeat)
        diagnostics.insert(1, "outer_fold", fold)
        diagnostics["selected"] = np.isclose(
            diagnostics["alpha"].astype(float),
            selector.alpha,
            atol=MSE_TIE_ATOL,
            rtol=0.0,
        )
        diagnostics["tuning_objective"] = selector.objective
        tuning_rows.extend(diagnostics.to_dict("records"))

        model = ridge_pipeline(selector.alpha, config)
        model.fit(X[tr], y[tr])
        pred = np.asarray(model.predict(X[te]), dtype=float)
        require_finite(pred, "OUTER_PRED_FINITE")

        fold_rows.append({
            "repeat": repeat,
            "outer_fold": fold,
            "train_n": len(tr),
            "test_n": len(te),
            "selected_alpha": selector.alpha,
            "inner_objective": selector.objective,
            "inner_score": selector.score,
            "pca_components": int(model.named_steps["pca"].n_components_),
        })

        for idx, p in zip(te, pred):
            pred_rows.append({
                "repeat": repeat,
                "outer_fold": fold,
                "representative_id": int(ids[idx]),
                "G": float(y[idx]),
                "prediction_raw": float(p),
                "selected_alpha": selector.alpha,
            })

    oof = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows).sort_values(["repeat", "outer_fold"], kind="stable").reset_index(drop=True)
    tuning = pd.DataFrame(tuning_rows).sort_values(["repeat", "outer_fold", "alpha"], kind="stable").reset_index(drop=True)

    require(len(oof) == config.outer_repeats * len(y), "OOF_ROW_COUNT", "Unexpected OOF row count")
    require(len(folds) == config.outer_repeats * config.outer_splits, "FOLD_ROW_COUNT", "Unexpected fold row count")
    require(bool((oof.groupby(["repeat", "representative_id"]).size() == 1).all()), "OOF_UNIQUE_PER_REPEAT", "Each candidate must appear once per repeat")
    require(int(tuning["selected"].sum()) == len(folds), "TUNING_SELECTED_COUNT", "Exactly one alpha must be selected per outer fit")

    id_to_position = {int(candidate_id): pos for pos, candidate_id in enumerate(ids)}
    oof_by_repeat: dict[int, np.ndarray] = {}
    repeat_rows: list[dict] = []

    cutoffs = tuple(int(k) for k in NDCG_CUTOFFS)
    for repeat in range(1, config.outer_repeats + 1):
        grp = oof[oof["repeat"] == repeat]
        require(len(grp) == len(ids), "OOF_REPEAT_SIZE", "Each repeat must predict every candidate exactly once")
        pred_aligned = np.empty(len(ids), dtype=float)
        true_aligned = np.empty(len(ids), dtype=float)
        for row in grp.itertuples(index=False):
            pos = id_to_position[int(row.representative_id)]
            pred_aligned[pos] = float(row.prediction_raw)
            true_aligned[pos] = float(row.G)
        require(np.allclose(true_aligned, y, atol=1e-12, rtol=0.0), "OOF_TARGET_ALIGNMENT", "OOF targets do not align with source target")
        oof_by_repeat[repeat] = pred_aligned
        metrics = evaluation_metrics(y, pred_aligned, cutoffs)
        repeat_rows.append({"repeat": repeat, **metrics})

    repeats = pd.DataFrame(repeat_rows)
    oof = oof.sort_values(["repeat", "representative_id"], kind="stable").reset_index(drop=True)
    return oof, folds, repeats, oof_by_repeat, tuning


def repeated_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[int],
    model_spec: dict,
):
    """Notebook-facing production repeated nested CV with complete audit outputs."""
    config = _config_from_spec(model_spec)
    return _repeated_nested_cv_impl(
        X=X,
        y=y,
        ids=ids,
        config=config,
        tuning_objective="mse",
    )


def repeated_nested_cv_ndcg_diag(
    X: np.ndarray,
    y: np.ndarray,
    ids: Sequence[int],
    model_spec: dict,
    k: int = 10,
):
    """Notebook-facing diagnostic repeated CV with NDCG inner tuning."""
    config = _config_from_spec(model_spec)
    return _repeated_nested_cv_impl(
        X=X,
        y=y,
        ids=ids,
        config=config,
        tuning_objective="ndcg",
        diagnostic_ndcg_k=int(k),
    )


def select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    model_spec: dict,
) -> tuple[float, pd.DataFrame]:
    """Select the production full-data Ridge alpha and retain the full trace."""
    config = _config_from_spec(model_spec)
    selector = select_alpha_mse(np.asarray(X, dtype=float), np.asarray(y, dtype=float), config)
    trace = selector.diagnostics.copy()
    trace["selected"] = np.isclose(
        trace["alpha"].astype(float),
        selector.alpha,
        atol=MSE_TIE_ATOL,
        rtol=0.0,
    )
    return float(selector.alpha), trace


def fit_full(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    model_spec: dict,
) -> tuple[Pipeline, np.ndarray]:
    """Fit the locked full-data PCA-Ridge model and return raw predictions."""
    config = _config_from_spec(model_spec)
    model = ridge_pipeline(float(alpha), config)
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    pred = np.asarray(model.predict(np.asarray(X, dtype=float)), dtype=float)
    require_finite(pred, "FULL_PRED_FINITE")
    return model, pred
