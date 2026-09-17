from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path("01_Potential_Talents_Main.ipynb")
EXPECTED_CELL_COUNT = 58

F01_NARRATIVE_ID = "58253a66"
F02_CODE_ID = "1439a289"
F02_NARRATIVE_ID = "4dc912ce"

F01_NARRATIVE = """# Interpret Figure F01

Figure F01 shows four progressively smaller population stages: 104 raw source rows, 52 distinct exact-title profiles, 50 valid unique profiles, and 34 candidates in the final HR modelling population. The largest visible reduction occurs between the first two bars, where repeated profile text is consolidated from 104 source rows to 52 distinct exact-title profiles. The much smaller reduction from 52 to 50 reflects removal of the two invalid source records, after which normalized-title deduplication confirms the valid unique population. The final reduction from 50 to 34 occurs when the transparent occupational HR-relevance rules are applied.

This progression separates data-quality decisions from relevance modelling. Duplicate source text and invalid records are dealt with before occupational eligibility is considered, while semantic similarity is not used to decide which candidates enter the HR modelling population.

The resulting 34 candidates therefore represent a rule-defined HR-relevant population rather than a population selected by the later machine-learning model. This population is subsequently used to construct the analytical target, evaluate PCA–Ridge, and conduct the primary management-feedback experiments.
"""

F02_NARRATIVE = """# Interpret Figure F02

Figure F02 plots each candidate's analytical target score $G$ on the horizontal axis against that candidate's mean out-of-fold Ridge prediction on the vertical axis. The dashed diagonal represents perfect agreement between the target and the held-out prediction. Most candidates form a dense cluster near the upper-right portion of the figure and remain relatively close to this diagonal, indicating strong agreement for the predominantly direct-HR population.

One observation is clearly separated from the main cluster: candidate ID 4, the sole adjacent-HR candidate. Its analytical target is substantially lower because occupational relevance contributes $H=0.5$, whereas the candidate's embedding remains semantically similar to HR-related titles. Ridge therefore predicts a considerably higher relevance score than the constructed target for this case. This provides a visible example of information contained in the rule-based $H$ component that cannot necessarily be reconstructed from semantic title embeddings alone.

The remaining deviations around the diagonal show that PCA–Ridge approximates rather than exactly reproduces $G$. Importantly, every underlying prediction is out-of-fold: each candidate was excluded from the corresponding outer-fold training data when that prediction was generated.

The figure therefore provides evidence of held-out numerical agreement, while the repeated NDCG results remain the primary evaluation because the practical objective is candidate ranking rather than exact score reconstruction.
"""


def is_private_use(ch: str) -> bool:
    code = ord(ch)
    return (
        0xE000 <= code <= 0xF8FF
        or 0xF0000 <= code <= 0xFFFFD
        or 0x100000 <= code <= 0x10FFFD
    )


def clean_unicode(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch == "\ufffd" or is_private_use(ch):
            continue
        if ch in {"\u200b", "\u200c", "\u200d", "\ufeff"}:
            continue
        if ch == "\u00a0":
            cleaned.append(" ")
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def normalize_markdown_math(text: str) -> str:
    # Dollar-delimited math is the most portable convention across Jupyter,
    # nbconvert and GitHub notebook rendering.
    text = text.replace("\\(", "$")
    text = text.replace("\\)", "$")
    text = text.replace("\\[", "$$")
    text = text.replace("\\]", "$$")
    return text


def split_source(text: str) -> list[str]:
    return text.splitlines(keepends=True) or [""]


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != EXPECTED_CELL_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CELL_COUNT} cells, found {len(cells)}"
        )

    ids = {cell.get("id") for cell in cells}
    required_ids = {F01_NARRATIVE_ID, F02_CODE_ID, F02_NARRATIVE_ID}
    missing = required_ids - ids
    if missing:
        raise RuntimeError(f"Missing locked cell IDs: {sorted(missing)}")

    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        text = clean_unicode(text)
        text = normalize_markdown_math(text)
        cell["source"] = split_source(text)

    by_id = {cell.get("id"): cell for cell in cells}
    by_id[F01_NARRATIVE_ID]["source"] = split_source(F01_NARRATIVE)
    by_id[F02_NARRATIVE_ID]["source"] = split_source(F02_NARRATIVE)

    f02_code = "".join(by_id[F02_CODE_ID].get("source", []))
    old_call = """presentation.save_oof_scatter(
    path=f02_path,
    y=y,
    preds=mean_oof_prediction,
)
"""
    new_call = """presentation.save_oof_scatter(
    path=f02_path,
    y=y,
    preds=mean_oof_prediction,
    ids=ids,
    highlight_id=4,
    highlight_label=\"ID 4 · adjacent HR\",
)
"""
    if old_call not in f02_code and new_call not in f02_code:
        raise RuntimeError("F02 plotting call does not match the locked source")
    f02_code = f02_code.replace(old_call, new_call)
    by_id[F02_CODE_ID]["source"] = split_source(f02_code)

    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        if "\\(" in text or "\\)" in text or "\\[" in text or "\\]" in text:
            raise RuntimeError(f"Legacy math delimiter remains in markdown cell {index}")
        for ch in text:
            if ch == "\ufffd" or is_private_use(ch):
                raise RuntimeError(
                    f"Invalid Unicode code point U+{ord(ch):04X} in markdown cell {index}"
                )

    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    print("Notebook Markdown formatting normalized successfully.")
    print(f"Cell count preserved: {len(cells)}")
    print("Approved F01/F02 narratives applied.")
    print("Approved F02 plot call applied.")


if __name__ == "__main__":
    main()
