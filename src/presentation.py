from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .config import FEEDBACK_DEPTH_TO_K


def mean_sd_text(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def display_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"




def conceptual_review_labels(frame: pd.DataFrame) -> list[str]:
    """Return locked 10%/20%/50% labels from the corresponding reviewed-candidate counts."""
    label_by_k = {
        int(k): f"{int(round(100 * fraction))}% review"
        for fraction, k in FEEDBACK_DEPTH_TO_K.items()
    }
    for column in ("reviewed_n", "review_cutoff", "reviewed_candidates"):
        if column in frame.columns:
            values = [int(value) for value in frame[column]]
            if all(value in label_by_k for value in values):
                return [label_by_k[value] for value in values]
    if "review_stage" in frame.columns:
        labels = [str(value) for value in frame["review_stage"]]
        if all(label in set(label_by_k.values()) for label in labels):
            return labels
    raise ValueError("Feedback figure data do not contain a recognized locked review depth.")

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
    label_map = {
        "Raw rows": "Raw source\nrows",
        "Exact-title profiles": "Exact-title\nprofiles",
        "Valid unique profiles": "Valid unique\nprofiles",
        "HR modelling population": "HR modelling\npopulation",
    }
    labels = [label_map.get(str(x), str(x)) for x in labels]
    counts = np.asarray(list(counts), dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, counts)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Count")
    ax.set_title("Candidate population flow")
    ax.set_ylim(0, float(counts.max()) * 1.14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + float(counts.max()) * 0.018,
            f"{int(value)}",
            ha="center",
            va="bottom",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_atomic(fig, path)


def save_oof_scatter(
    path: Path | str,
    y,
    preds,
    ids=None,
    highlight_id: int | None = None,
    highlight_label: str | None = None,
) -> None:
    y = np.asarray(y, dtype=float)
    preds = np.asarray(preds, dtype=float)
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y, preds)

    lo = float(min(y.min(), preds.min()))
    hi = float(max(y.max(), preds.max()))
    pad = (hi - lo) * 0.04
    plot_lo = lo - pad
    plot_hi = hi + pad

    ax.plot([plot_lo, plot_hi], [plot_lo, plot_hi], linestyle="--")
    ax.set_xlim(plot_lo, plot_hi)
    ax.set_ylim(plot_lo, plot_hi)
    ax.set_aspect("equal", adjustable="box")

    if ids is not None and highlight_id is not None:
        ids = np.asarray(ids, dtype=int)
        matches = np.flatnonzero(ids == int(highlight_id))
        if len(matches) == 1:
            index = int(matches[0])
            label = highlight_label or f"ID {highlight_id}"
            ax.annotate(
                label,
                (y[index], preds[index]),
                xytext=(12, -4),
                textcoords="offset points",
                va="center",
            )

    ax.set_xlabel("Analytical target G")
    ax.set_ylabel("Mean OOF Ridge prediction")
    ax.set_title("Out-of-fold prediction versus target")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_atomic(fig, path)


def save_rank_plot(path: Path | str, reference_ranks, predicted_ranks) -> None:
    reference_ranks = np.asarray(reference_ranks, dtype=float)
    predicted_ranks = np.asarray(predicted_ranks, dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.scatter(reference_ranks, predicted_ranks)
    upper = float(max(reference_ranks.max(), predicted_ranks.max()))
    ax.plot([1, upper], [1, upper], linestyle="--")
    ax.set_xlim(0.5, upper + 0.5)
    ax.set_ylim(upper + 0.5, 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Analytical target rank")
    ax.set_ylabel("OOF Ridge rank")
    ax.set_title("Analytical target rank versus OOF Ridge rank")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_atomic(fig, path)


def save_feedback(
    path: Path | str,
    df: pd.DataFrame,
    depth_col: str,
    before_col: str,
    after_col: str,
) -> None:
    frame = df.sort_values(depth_col).copy()
    before = frame[before_col].to_numpy(dtype=float)
    after = frame[after_col].to_numpy(dtype=float)
    delta = after - before
    x = np.arange(len(frame))

    labels = conceptual_review_labels(frame)
    metric_labels = ["NDCG@4", "NDCG@7", "NDCG@17"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.bar(x, delta)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Δ held-out NDCG (feedback − pre-feedback)")
    ax.set_title("Effect of management feedback on held-out ranking")

    lower = float(delta.min()) * 1.22
    upper = abs(float(delta.min())) * 0.14
    ax.set_ylim(lower, upper)

    for bar, metric, value in zip(bars, metric_labels, delta):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) - 0.00018,
            f"{metric}\n{value:+.6f}",
            ha="center",
            va="top",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_atomic(fig, path)


def save_effort(
    path: Path | str,
    df: pd.DataFrame,
    actions_col: str,
    gain_col: str,
) -> None:
    frame = df.copy()
    x = frame[actions_col].to_numpy(dtype=float)
    y = frame[gain_col].to_numpy(dtype=float) * 1000.0

    labels = conceptual_review_labels(frame)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for xv, yv in zip(x, y):
        ax.vlines(xv, 0, yv, linestyles="dotted")
    ax.scatter(x, y)

    for label, xv, yv in zip(labels, x, y):
        offset_y = 8 if yv >= 0 else -14
        va = "bottom" if yv >= 0 else "top"
        ax.annotate(
            label,
            (xv, yv),
            xytext=(6, offset_y),
            textcoords="offset points",
            va=va,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Effective management actions")
    ax.set_ylabel("Fitted ΔNDCG (×10⁻³)")
    ax.set_title("Management effort versus fitted ranking gain")
    ax.set_xlim(0.4, 8.6)
    ax.set_ylim(-0.9, 0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_figure_atomic(fig, path)
