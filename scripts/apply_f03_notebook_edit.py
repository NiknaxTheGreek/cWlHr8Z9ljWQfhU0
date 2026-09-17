import json
import re
from pathlib import Path

NOTEBOOK = Path('01_Potential_Talents_Main.ipynb')
PRESENTATION = Path('src/presentation.py')

# --- Notebook: keep actual review fractions for computation, but display locked conceptual labels. ---
nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
assert len(nb['cells']) == 58

notebook_replacements = 0
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    text = ''.join(cell.get('source', []))
    original = text

    # Progressive feedback summary table: display 10/20/50 rather than 4/34, 7/34, 17/34 as percentages.
    pattern = re.compile(
        r'feedback_results_table\[\s*"review_percentage"\s*\]\s*=\s*\(\s*'
        r'100\s*\*\s*feedback_results_table\[\s*"review_fraction"\s*\]\s*\)',
        flags=re.MULTILINE,
    )
    replacement = '''review_percentage_by_k = {
    int(k): int(round(100 * fraction))
    for fraction, k in config.FEEDBACK_DEPTH_TO_K.items()
}
feedback_results_table["review_percentage"] = (
    feedback_results_table["review_cutoff"].map(review_percentage_by_k)
)'''
    text, n = pattern.subn(replacement, text)
    notebook_replacements += n

    # Comprehensive final NDCG table: use the same conceptual stage labels.
    old = 'review_pct = int(round(100 * reviewed_n / len(hr_analysis)))'
    new = 'review_pct = review_percentage_by_k[reviewed_n]'
    if old in text:
        text = text.replace(old, new)
        notebook_replacements += 1

    if text != original:
        cell['source'] = text.splitlines(keepends=True)

assert notebook_replacements == 2, f'Expected 2 notebook label replacements, got {notebook_replacements}'
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

# --- Presentation figures: derive labels from the locked conceptual depth-to-k configuration. ---
presentation = PRESENTATION.read_text(encoding='utf-8')

if 'from .config import FEEDBACK_DEPTH_TO_K' not in presentation:
    presentation = presentation.replace(
        'from matplotlib.ticker import MaxNLocator\n',
        'from matplotlib.ticker import MaxNLocator\n\nfrom .config import FEEDBACK_DEPTH_TO_K\n',
        1,
    )

helper = '''\n\ndef conceptual_review_labels(frame: pd.DataFrame) -> list[str]:
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
'''

if 'def conceptual_review_labels(' not in presentation:
    marker = 'def save_figure_atomic('
    idx = presentation.index(marker)
    presentation = presentation[:idx] + helper + '\n' + presentation[idx:]

old_feedback_labels = 'labels = [f"{int(round(value * 100))}% review" for value in frame[depth_col].to_numpy(dtype=float)]'
assert old_feedback_labels in presentation, 'Expected save_feedback percentage-label expression not found'
presentation = presentation.replace(
    old_feedback_labels,
    'labels = conceptual_review_labels(frame)',
    1,
)

old_effort_block = '''    if "review_fraction" in frame.columns:
        labels = [f"{int(round(value * 100))}% review" for value in frame["review_fraction"]]
    else:
        labels = [f"Review {index}" for index in range(1, len(frame) + 1)]'''
assert old_effort_block in presentation, 'Expected save_effort percentage-label block not found'
presentation = presentation.replace(
    old_effort_block,
    '    labels = conceptual_review_labels(frame)',
    1,
)

assert 'int(round(value * 100))' not in presentation
PRESENTATION.write_text(presentation, encoding='utf-8')

print('Applied conceptual 10%/20%/50% display labels; computational fractions and all model results unchanged.')
