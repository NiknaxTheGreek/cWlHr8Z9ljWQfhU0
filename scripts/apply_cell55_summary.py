from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("01_Potential_Talents_Main.ipynb")

CELL55_SOURCE = '''# Build the comprehensive final NDCG summary table

ndcg_comparison_rows = []

for k in model_spec["ndcg_cutoffs"]:
    metric = f"ndcg_at_{k}"
    w_ndcg = ranking.ndcg_at(
        y_true=y,
        y_score=W_baseline_scores,
        k=k,
    )
    ridge_repeat_values = (
        cv_repeat_metrics[metric]
        .astype(float)
        .to_numpy()
    )
    ridge_mean = float(ridge_repeat_values.mean())
    ridge_sd = float(ridge_repeat_values.std(ddof=1))

    ndcg_comparison_rows.append(
        {
            "Experiment": "Automated ranking",
            "Population": "HR34",
            "Metric": f"NDCG@{k}",
            "Reference": "W-only",
            "Reference NDCG": float(w_ndcg),
            "Result": "PCA-Ridge",
            "Result NDCG": ridge_mean,
            "SD": ridge_sd,
            "Delta NDCG": ridge_mean - float(w_ndcg),
            "Effective actions": np.nan,
        }
    )

feedback_actions = (
    feedback_fitted_summary
    .set_index("review_cutoff")["effective_actions"]
    .to_dict()
)

for _, row in feedback_generalization_summary.iterrows():
    metric_name = str(row["primary_metric"])
    cutoff = int(metric_name.split("_")[-1])
    reviewed_n = int(row["reviewed_n"])
    review_pct = int(round(100 * reviewed_n / len(hr_analysis)))

    ndcg_comparison_rows.append(
        {
            "Experiment": f"Management feedback - {review_pct}%",
            "Population": "HR34",
            "Metric": f"NDCG@{cutoff}",
            "Reference": "Pre-feedback Ridge",
            "Reference NDCG": float(row["pre_feedback_mean"]),
            "Result": "Feedback-adjusted Ridge",
            "Result NDCG": float(row["feedback_mean"]),
            "SD": float(row["feedback_sd"]),
            "Delta NDCG": float(row["delta_vs_pre_feedback"]),
            "Effective actions": int(feedback_actions[reviewed_n]),
        }
    )

for _, row in top25_results_table.iterrows():
    ndcg_comparison_rows.append(
        {
            "Experiment": "Top-25 management audit",
            "Population": row["population"],
            "Metric": "NDCG@25",
            "Reference": "Automated order",
            "Reference NDCG": float(row["ndcg_at_25_before"]),
            "Result": "Management order",
            "Result NDCG": float(row["ndcg_at_25_after"]),
            "SD": np.nan,
            "Delta NDCG": float(row["delta_ndcg_at_25"]),
            "Effective actions": np.nan,
        }
    )

ndcg_comparison_table = pd.DataFrame(ndcg_comparison_rows)

print("Comprehensive final NDCG summary:")
display(
    ndcg_comparison_table.style.format(
        {
            "Reference NDCG": "{:.6f}",
            "Result NDCG": "{:.6f}",
            "SD": "{:.6f}",
            "Delta NDCG": "{:+.6f}",
            "Effective actions": lambda value: (
                "—" if pd.isna(value) else f"{int(value)}"
            ),
        },
        na_rep="—",
    )
)
'''


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 58

    matches = []
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "ndcg_comparison_rows = []" in source and "primary_ndcg_scorecard" in source:
            matches.append(i)

    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Cell 55 target, found {matches}")

    idx = matches[0]
    if idx != 54:
        raise RuntimeError(f"Expected Cell 55 at index 54, found index {idx}")

    nb["cells"][idx]["source"] = CELL55_SOURCE.splitlines(keepends=True)
    nb["cells"][idx]["outputs"] = []
    nb["cells"][idx]["execution_count"] = None

    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    print("Cell 55 replaced with one comprehensive summary table.")
    print("Cell count preserved:", len(nb["cells"]))


if __name__ == "__main__":
    main()
