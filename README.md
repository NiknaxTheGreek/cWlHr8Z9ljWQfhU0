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
- `01_Potential_Talents_Main.ipynb` — authoritative reviewer-facing analysis notebook.
- `scripts/fetch_model40.py` — deterministic one-time fetch and validation of the external NLPL Model 40 binary.
- `models/` — committed Model 40 metadata only; the multi-GB pretrained payload is external and ignored by Git.
- `src/` — tested reusable production logic.
- `tests/` — targeted tests paired with promoted critical logic.
- `data/raw/` — immutable recovered source data and appendix-only manual labels.
- `project_control/` — reconciled contracts, registries, dependency/cell maps and execution state.
- `reports/` — authoritative Markdown reports and derived PDFs.
- `outputs/` — compact final reviewer-facing result artifacts.

## Model 40 setup
The pretrained NLPL Model 40 payload is intentionally not stored in Git or Git LFS. It is an external model dependency, while the repository retains the source metadata and deterministic retrieval logic.

From the repository root, run:

```bash
python scripts/fetch_model40.py
```

The script downloads the official NLPL Model 40 archive from `https://vectors.nlpl.eu/repository/20/40.zip`, extracts `models/model.bin`, validates its expected file size and Word2Vec header, writes a local SHA-256 provenance manifest, and removes the temporary archive. `models/model.bin`, `models/model.txt`, and the local manifest are ignored by Git.

A backup copy of the complete Model 40 source package is maintained separately in project storage; it is not required to clone the repository.

## Results at a glance
Evidence-gated. This section will be populated only from validated final-run evidence.

## Management feedback findings
Evidence-gated. No recommendation will be written before the locked 10%/20%/50% feedback experiments complete.

## Reproduction
During construction, prepare Model 40 with `python scripts/fetch_model40.py`, then run `01_Potential_Talents_Main.ipynb` from a fresh kernel using **Restart Kernel → Run All**. Final release instructions will be frozen after two clean runs and a fresh-clone validation.
