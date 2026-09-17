from __future__ import annotations

import json
from pathlib import Path

PATH = Path('01_Potential_Talents_Main.ipynb')

F04 = '''# Interpret Figure F04

Figure F04 shows the change in held-out NDCG after incorporating management feedback at each of the three review depths. Each bar represents the feedback-adjusted repeated-CV result minus the corresponding pre-feedback repeated-CV result. The horizontal zero line therefore represents no change in held-out ranking performance.

All three bars fall below zero. At the 10% review stage, NDCG@4 changes from 0.989739 to 0.987483, a difference of -0.002256. At 20% review, NDCG@7 changes from 0.991032 to 0.985843, producing the largest observed decrease, -0.005188. At 50% review, NDCG@17 changes from 0.994684 to 0.992015, a difference of -0.002668.

The three bars correspond to different depth-matched NDCG cutoffs, so the purpose of the figure is not to compare the absolute NDCG level across review stages. Instead, it compares the direction and magnitude of the feedback effect relative to each stage's own pre-feedback reference.

Management-adjusted targets can be fitted by the model, but none of the three review depths produces an improvement in repeated held-out ranking performance. The available evidence therefore does not support replacing the frozen pre-feedback automated ranking with a model retrained to reproduce the management ordering.

Figure F05 examines the complementary question of whether greater management intervention at least produces progressively larger fitted gains before held-out generalization is considered.
'''

F05 = '''# Interpret Figure F05

Figure F05 compares the number of effective management actions at each review depth with the corresponding change in fitted NDCG. The horizontal axis therefore represents the amount of management intervention that actually changed the frozen ordering, while the vertical axis shows the resulting fitted ranking gain on a $10^{-3}$ scale. The dotted vertical guides show the deviation of each fitted result from the zero-change reference without implying a continuous trend between the three experiments.

The three review experiments do not show a monotonic relationship between intervention effort and fitted improvement. At 10% review, three effective management actions correspond to a fitted NDCG change of approximately $-0.496 \\times 10^{-3}$. At 20% review, only one effective action produces the largest positive fitted change, approximately $+0.535 \\times 10^{-3}$. At 50% review, eight effective actions produce a smaller positive fitted change of approximately $+0.252 \\times 10^{-3}$.

The magnitude of all three fitted changes is small, and increasing the number of management actions does not correspond to progressively greater fitted benefit. This indicates that the amount of intervention alone is not a useful proxy for how strongly the fitted ranking changes.

More importantly, fitted response and held-out generalization answer different questions. Figure F05 shows how strongly the model can respond to the management-adjusted targets within the fitted ranking, whereas Figure F04 shows whether those adjustments improve repeated held-out ranking performance. The latter remains negative at all three review depths.

Taken together, Figures F04 and F05 show that management feedback can alter the fitted ordering, but neither greater intervention nor those fitted changes provide evidence of improved generalization. This supports retaining the frozen pre-feedback automated ranking as the production result while preserving the management experiments as governance and sensitivity evidence.
'''


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(PATH.read_text(encoding='utf-8'))
    cells = nb['cells']
    assert len(cells) == 58, len(cells)

    # Locked ending structure: 49 F04 code, 50 F04 MD, 51 F05 code, 52 F05 MD.
    assert cells[48]['cell_type'] == 'code'
    assert cells[49]['cell_type'] == 'markdown'
    assert cells[50]['cell_type'] == 'code'
    assert cells[51]['cell_type'] == 'markdown'
    assert '# Interpret Figure F04' in ''.join(cells[49]['source'])
    assert '# Interpret Figure F05' in ''.join(cells[51]['source'])

    cells[49]['source'] = lines(F04)
    cells[51]['source'] = lines(F05)

    PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Applied approved F04/F05 narratives; 58-cell structure preserved.')


if __name__ == '__main__':
    main()
