import json
from pathlib import Path

p = Path('01_Potential_Talents_Main.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
assert len(nb['cells']) == 58

cell = next(c for c in nb['cells'] if c.get('id') == '3017bff9')
text = '''# Interpret Figure F03

Figure F03 compares each candidate's analytical target rank on the horizontal axis with the rank obtained from the candidate's mean out-of-fold Ridge prediction on the vertical axis. Rank 1 represents the highest-ranked candidate on both axes. The dashed diagonal therefore represents exact agreement between the analytical target ordering and the out-of-fold model ordering.

Most observations remain reasonably close to the diagonal, particularly toward the strongest and weakest ends of the ranking. This shows that PCA–Ridge broadly preserves the candidate ordering encoded by the analytical target while still producing individual rank movements.

Points above the diagonal are ranked more highly by Ridge than by the analytical target, whereas points below the diagonal are ranked lower by Ridge. Some candidates move several positions, showing that strong overall ranking performance does not imply exact reproduction of every candidate's position.

This pattern is consistent with the metric results. NDCG remains very high because it places greater emphasis on preserving highly relevant candidates near the top of the ranking, while Spearman and Kendall measure agreement across the complete ordering and are therefore more sensitive to rank movements throughout the full population.

Figure F03 consequently complements the numerical ranking metrics by showing where ordering disagreement occurs rather than summarizing that disagreement in a single statistic.
'''
cell['source'] = text.splitlines(keepends=True)
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Applied approved F03 narrative; 58 cells preserved.')
