import numpy as np

from src.config import ProjectConfig
from src.modeling import (
    fit_final_model,
    ridge_pipeline,
    run_repeated_nested_cv,
    select_alpha_inner_oof_ndcg,
    select_alpha_mse,
)


def small_config(tmp_path):
    return ProjectConfig(
        root=tmp_path,
        ridge_alphas=(0.01, 0.1, 1.0),
        outer_splits=5,
        outer_repeats=2,
        inner_splits=4,
    )


def synthetic():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 12))
    y = np.linspace(0.1, 0.9, 20) + rng.normal(scale=0.02, size=20)
    ids = np.arange(1, 21)
    return X, y, ids


def test_pipeline_order(tmp_path):
    pipe = ridge_pipeline(0.1, small_config(tmp_path))
    assert list(pipe.named_steps) == ["pca", "scaler", "ridge"]
    assert pipe.named_steps["pca"].n_components == 0.95
    assert pipe.named_steps["pca"].svd_solver == "full"


def test_mse_and_ndcg_selectors_are_deterministic(tmp_path):
    X, y, _ = synthetic()
    cfg = small_config(tmp_path)
    a = select_alpha_mse(X, y, cfg)
    b = select_alpha_mse(X, y, cfg)
    assert a.alpha == b.alpha
    assert a.objective == "mse"
    d = select_alpha_inner_oof_ndcg(X, y, cfg, k=10)
    assert d.alpha in cfg.ridge_alphas
    assert d.objective.startswith("inner_oof_ndcg@")


def test_repeated_nested_cv_bookkeeping(tmp_path):
    X, y, ids = synthetic()
    cfg = small_config(tmp_path)
    oof, folds, repeats = run_repeated_nested_cv(X, y, ids, cfg, tuning_objective="mse")
    assert len(oof) == 40
    assert len(folds) == 10
    assert len(repeats) == 2
    assert (oof.groupby(["repeat", "representative_id"]).size() == 1).all()


def test_final_fit_is_mse_tuned(tmp_path):
    X, y, _ = synthetic()
    cfg = small_config(tmp_path)
    model, selection, pred = fit_final_model(X, y, cfg)
    assert selection.objective == "mse"
    assert pred.shape == (20,)
    assert np.isfinite(pred).all()
