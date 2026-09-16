from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_sd_text(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def display_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def save_figure_atomic(fig: matplotlib.figure.Figure, path: Path | str, dpi: int = 160) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.stem}.", suffix=p.suffix, dir=p.parent)
    os.close(fd)
    try:
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
        if Path(tmp).stat().st_size <= 0:
            raise IOError(f"Empty figure output: {tmp}")
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    finally:
        plt.close(fig)


def save_population_flow(path: Path | str, labels, counts) -> None:
    labels = [str(x) for x in labels]
    counts = np.asarray(list(counts), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(labels))
    ax.bar(positions, counts)
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel("Candidates")
    ax.set_title("Candidate population flow")
    for x, value in zip(positions, counts):
        ax.text(x, value, f"{int(value)}", ha="center", va="bottom")
    save_figure_atomic(fig, path)


def save_oof_scatter(path: Path | str, y, preds) -> None:
    y = np.asarray(y, dtype=float)
    preds = np.asarray(preds, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(y, preds)
    lo = float(min(y.min(), preds.min()))
    hi = float(max(y.max(), preds.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Analytical target G")
    ax.set_ylabel("Mean OOF Ridge prediction")
    ax.set_title("Out-of-fold prediction versus target")
    save_figure_atomic(fig, path)


def save_rank_plot(path: Path | str, reference_ranks, predicted_ranks) -> None:
    reference_ranks = np.asarray(reference_ranks, dtype=float)
    predicted_ranks = np.asarray(predicted_ranks, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(reference_ranks, predicted_ranks)
    upper = float(max(reference_ranks.max(), predicted_ranks.max()))
    ax.plot([1, upper], [1, upper], linestyle="--")
    ax.set_xlabel("Analytical target rank")
    ax.set_ylabel("Mean OOF Ridge rank")
    ax.set_title("Target rank versus OOF Ridge rank")
    ax.invert_xaxis()
    ax.invert_yaxis()
    save_figure_atomic(fig, path)


def save_feedback(
    path: Path | str,
    df: pd.DataFrame,
    depth_col: str,
    before_col: str,
    after_col: str,
) -> None:
    frame = df.sort_values(depth_col).copy()
    x = frame[depth_col].to_numpy(dtype=float)
    before = frame[before_col].to_numpy(dtype=float)
    after = frame[after_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(x, before, marker="o", label="Pre-feedback")
    ax.plot(x, after, marker="o", label="Feedback-adjusted")
    ax.axhline(0, linewidth=0.5)
    ax.set_xlabel("Reviewed fraction")
    ax.set_ylabel("Depth-matched NDCG")
    ax.set_title("Management-feedback generalization")
    ax.legend()
    save_figure_atomic(fig, path)


def save_effort(
    path: Path | str,
    df: pd.DataFrame,
    actions_col: str,
    gain_col: str,
) -> None:
    frame = df.copy()
    x = frame[actions_col].to_numpy(dtype=float)
    y = frame[gain_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(x, y)
    for row_index, (xv, yv) in enumerate(zip(x, y), start=1):
        ax.annotate(str(row_index), (xv, yv), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Effective management actions")
    ax.set_ylabel("Fitted NDCG gain")
    ax.set_title("Management effort versus fitted gain")
    save_figure_atomic(fig, path)
