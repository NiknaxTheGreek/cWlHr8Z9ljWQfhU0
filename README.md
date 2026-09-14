# Potential Talents

Reproducible candidate-ranking project for the Apziva Potential Talents case study.

> Construction status: Gate 2 passed. The repository is being rebuilt from the locked methodology. Numerical result sections remain evidence-gated until the new pipeline has executed and validated.

## Business objective
Reduce management screening effort by ranking genuine HR-related candidates while preserving an auditable path from source data through relevance rules, semantic similarity, modelling and management feedback.

## Locked analytical design
The final analytical target is `G = (H + W) / 2`, where `H` is automated HR relevance and `W` is semantic similarity to the two locked HR search queries. Historical manual labels are appendix-only and are not model ground truth.

The production model is `PCA(95%, full) -> StandardScaler -> Ridge`, tuned by MSE under repeated nested cross-validation. NDCG is the primary ranking metric.

## Repository structure
- `00_Project_Audit_and_Setup.ipynb` — supporting audit/setup/prototyping notebook.
- `01_Potential_Talents_Main.ipynb` — authoritative reviewer-facing analysis notebook (constructed after foundation promotion).
- `src/` — tested reusable production logic.
- `tests/` — targeted tests paired with promoted critical logic.
- `data/raw/` — immutable recovered source data and appendix-only manual labels.
- `project_control/` — reconciled contracts, registries, dependency/cell maps and execution state.
- `reports/` — authoritative Markdown reports and derived PDFs.
- `outputs/` — compact final reviewer-facing result artifacts.

## Results at a glance
Evidence-gated. This section will be populated only from validated final-run evidence.

## Management feedback findings
Evidence-gated. No recommendation will be written before the locked 10%/20%/50% feedback experiments complete.

## Reproduction
Release-gated. Final instructions will be frozen after two clean runs and a fresh-clone validation.
