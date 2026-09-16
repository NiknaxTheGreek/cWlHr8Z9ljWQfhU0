# Potential Talents — Master Execution & Orchestration Contract v1.1

This repository follows the final locked Apziva Potential Talents execution contract reconciled through four 100-question audits and Gate 2.

## Authority order
1. Locked 25-step analytical methodology.
2. Locked Audits 1–3 implementation decisions.
3. Fourth-audit execution/orchestration contract.
4. Target repository blueprint.
5. Master build sequence/dependency graph.
6. Notebook section plans.
7. Cell maps and implementation details.

Lower layers may operationalize higher layers but may not contradict them.

## Analytical locks
- Analytical target: `G = (H + W) / 2` only.
- `H`: direct HR = 1.0, adjacent HR = 0.5, irrelevant = 0.0.
- `W`: average of two rescaled cosine-similarity query scores.
- Queries: `aspiring human resources`; `seeking human resources`.
- Manual relevance labels are historical appendix evidence only. They are forbidden from analytical filtering, target construction, features, model selection, training, management ordering and feedback.
- Locked population audit: 104 raw rows -> 52 exact source-title representatives -> remove exactly 2 invalid non-candidate records -> normalized-title dedup -> 50 -> HR rules -> 34.
- Candidate/query embeddings use the official NLPL Model 40 English CoNLL17 Word2Vec 100D binary. `models/model.bin` is an external local dependency retrieved deterministically by `scripts/fetch_model40.py`; the embedding pipeline streams the validated official binary and retains only the vectors required for the current run. No precomputed embedding cache is an analytical dependency.
- Model input: `X` is the 34x100 candidate-embedding matrix; `y = G`.
- Production pipeline: `PCA(n_components=0.95, svd_solver='full') -> StandardScaler -> Ridge`.
- Production Ridge tuning objective is negative MSE. Genuine inner-OOF NDCG@10 tuning is diagnostic sensitivity only and never automatically replaces MSE tuning.
- Outer validation: `RepeatedKFold(5 splits x 10 repeats)`; inner validation: shuffled 4-fold CV.
- Each repeat produces one complete 34-candidate OOF ranking. Primary ranking cutoffs: NDCG@4, @7, @10, @17 and @34.
- Ridge metrics are summarized across the 10 complete OOF repeats as mean ± SD while all repeat-level evidence remains reproducible.
- `W`-only is one deterministic baseline; compare it against Ridge mean ± SD with Ridge-minus-W deltas.
- Raw predictions are authoritative. Any clipping is presentation-only.
- Final pre-feedback ranking freezes top4/top7/top17 and the full order before management feedback.
- One explicit `MANAGEMENT_TOP17_ORDER` is the only management ordering. Top7 and top4 are restrictions of that same ordering.
- Sequential feedback preserves the target-score multiset, preserves prior locks, leaves the unreviewed population unchanged, records no-ops without retraining, and retunes/refits after each effective action.
- Feedback depths: 10% = top4 evaluated primarily by NDCG@4; 20% = top7 by NDCG@7; 50% = top17 by NDCG@17.
- Final stage is the primary feedback endpoint. Best stage is supplementary; ties choose the earliest stage.
- Final repeated nested-CV sensitivity covers Stage 0 plus each immutable final 10%, 20% and 50% feedback target.

## Repository and artifact locks
- Exactly two notebooks: `00_Project_Audit_and_Setup.ipynb` and `01_Potential_Talents_Main.ipynb`.
- The main notebook is the authoritative reviewer-facing executable artifact; `src/` supports it and never replaces it.
- `00_Project_Audit_and_Setup.ipynb` has exactly nine fixed blocks: repository inventory; current-state assessment; target blueprint; migration plan; environment verification; data/external-resource inspection; helper/rule/test prototyping; embedding/cache verification; promotion/readiness checklist. The historical Block 8 label is retained for stable notebook mapping, but the production architecture uses the validated external full Model 40 binary rather than a committed compact cache.
- `01_Potential_Talents_Main.ipynb` has exactly fifteen locked top-level sections and the reconciled stable-ID cell map in `project_control/cell_map.json`.
- Authoritative reports are Markdown; PDFs are derived reviewer artifacts.
- Reviewer-facing outputs are intentionally compact. Detailed fold, exclusion, model-provenance and feedback audit tables remain reproducibly regenerable unless registered as release artifacts.
- All authoritative writes use atomic persistence plus post-write reload/integrity validation.
- Two clean final runs and an actual fresh-clone PASS are required before `REVIEWER_READY = TRUE`.

## Construction gate
Gate 2 passed after reconciliation. Production construction is authorized on the `phase6-construction` branch. Runtime gates such as Model 40 external-binary retrieval/provenance validation, explicit management ordering, two clean final runs and fresh-clone validation remain enforced at their proper stages.
