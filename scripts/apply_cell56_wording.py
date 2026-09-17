from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("01_Potential_Talents_Main.ipynb")

OLD_PARAGRAPH = (
    "Management feedback produces a different result. When the feedback-adjusted targets are fitted directly, "
    "Ridge can move toward the management-specified ordering. However, the matched repeated-CV comparisons at "
    "the 10%, 20%, and 50% review depths do not show improved generalization relative to the original pre-feedback "
    "model. The distinction between fitted improvement and out-of-fold improvement is therefore critical.\n"
)

NEW_PARAGRAPH = (
    "Management feedback produces a different result. The fitted experiments respond to the management-adjusted "
    "targets, but the amount of intervention does not translate into progressively greater fitted benefit: the 10%, "
    "20%, and 50% review stages contain 3, 1, and 8 effective actions respectively, while their fitted NDCG changes "
    "remain small and non-monotonic. More importantly, the matched repeated-CV comparisons show held-out NDCG changes "
    "of -0.002256, -0.005188, and -0.002668 at the corresponding review depths. The distinction between fitted response "
    "and held-out generalization is therefore critical: management feedback changes the fitted ordering, but none of "
    "the three experiments improves out-of-fold ranking performance.\n"
)

OLD_OBSERVATION = (
    "5. Management feedback changes the fitted ranking but does not improve repeated out-of-fold ranking performance "
    "in the available data.\n"
)

NEW_OBSERVATION = (
    "5. Management feedback changes the fitted ranking, but neither greater intervention nor the observed fitted changes "
    "translate into improved repeated out-of-fold ranking performance.\n"
)


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 58

    cell = nb["cells"][55]
    assert cell.get("cell_type") == "markdown"
    source = "".join(cell.get("source", []))
    assert source.startswith("# Model performance observations")

    if source.count(OLD_PARAGRAPH) != 1:
        raise RuntimeError("Expected exactly one management-feedback paragraph in Cell 56.")
    if source.count(OLD_OBSERVATION) != 1:
        raise RuntimeError("Expected exactly one observation 5 in Cell 56.")

    source = source.replace(OLD_PARAGRAPH, NEW_PARAGRAPH)
    source = source.replace(OLD_OBSERVATION, NEW_OBSERVATION)

    cell["source"] = source.splitlines(keepends=True)

    NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )

    print("Cell 56 wording updated.")
    print("Cell count preserved:", len(nb["cells"]))


if __name__ == "__main__":
    main()
