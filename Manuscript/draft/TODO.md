# Manuscript Backlog (central TODO)

Created 2026-10-14. Cross-references: `discrepancies.md` (D-xx), `research/experiment_recommendations.md` (E-xx), `research/contribution_matrix.md` (C-x).
Fields: ID · Category · Priority · Issue · Scientific justification · Proposed action · Expected outcome · Relevant files · Status · Author approval

## Critical

| ID | Cat. | Issue | Justification | Proposed action | Expected outcome | Files | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| X-00 | Writing | **Choose contribution framing** (A: method; B: rigorous benchmark — recommended; C: epidemiology) | The central claim (DCMFNet superiority) is contradicted by saved results (D-11). The framing determines which experiments matter. | Author reads `research/contribution_matrix.md` §4 and decides | Unblocks Intro/Results/Discussion drafting | contribution_matrix.md | **Completed 2026-10-19: Option B chosen by author, target AIIM** | Given |
| E-01 | Experiment | GLMM/GAMM interactions built by R recycling (97 of 546 terms) | Interaction findings and GLMM/GAMM metrics are artifacts of column order | Re-specify interactions (all-pairs penalized and/or pre-specified subset); re-fit | Valid statistical baselines and interaction evidence | Models/GLMM/Method/GLMM.R:42-54; GAMM.R | **Completed 2026-10-23** (author ran updated scripts; results in 04_results.md §3.2–3.4) | **Required** |
| E-02 | Experiment | R and Python models use different test partitions | No valid paired comparison DCMFNet vs GLMM/GAMM; thesis claims "same split" | Export Python cmpair partitions; re-run R models on them | Paired comparison possible | GLMM.R:203-213; legacy/train.py:61 | **Completed 2026-10-23** (author ran updated scripts; results in 04_results.md §3.2–3.4) | **Required** |
| E-03 | Experiment | Test-informed configuration selection; tuning data overlaps other seeds' test sets | Optimistic bias in DCMFNet metrics; TRIPOD+AI violation | Nested selection protocol; equal tuning budget for strong baselines | Unbiased final estimates | complexity_analysis.py:204-250; tuning.py | Not Started | **Required** |
| M-01 | Methodology | Outcomes mislabelled: "negative symptoms" = depressive items; "positive" = psychotic-like + hypomanic | Construct validity; psychiatric reviewers will reject | Author confirms instruments, item scales, normalization constants (46, 33); rename outcomes throughout | Correct, defensible outcome definitions | rename_catss_with_mapping.py:73-92 | Not Started | **Required** (input) |
| A-01 | Analysis | No paired uncertainty or significance for model comparisons | "Consistent improvement" claims need paired inference; AIIM asks for statistical significance where applicable | Corrected resampled t-test / pair bootstrap; Holm correction (E-08). Cheapest first step: paired 5-split analysis from saved per-seed CSVs | Defensible comparison statements | Models/*/Results/*per_seed.csv | **In progress 2026-10-23:** 5-split corrected t-tests from saved per-seed CSVs in §3.3; final version from prediction_evaluation.py | **Required** |
| C-02 | Compliance | Ethics approval number(s), date(s), consent/assent (minors), data governance | Mandatory ethics statement (Guide p.4, p.10) | Author supplies Swedish Ethical Review Authority / regional board details for CATSS and genotype linkage | Compliant ethics statement | 07_declarations.md | Not Started | **Required** (input) |
| C-03 | Compliance | Author list, affiliations, corresponding author, CRediT, funding, competing interests | Mandatory title page and declarations | Author supplies | Compliant front matter | 01_title_abstract.md; 07_declarations.md | Not Started | **Required** (input) |
| W-01 | Writing | Thesis central claims cannot be carried over (superiority, interaction "amplifier", "consistent with GLMM") | Unsupported (D-01, D-11, D-12) | Do not use them in the manuscript; draft to evidence | Avoids reviewer rejection for overclaiming | thesis Ch. 5–7 | In Progress (enforced in draft) | Not Required |

## High

| ID | Cat. | Issue | Justification | Proposed action | Expected outcome | Files | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| R-02 | Methodology | Tuning outputs in repo are from a later per-modality campaign; used hyperparameters come from git commit c49eec1; ≥5 campaigns (D-21) | Reproducibility; undisclosed selection | Restore c49eec1 tuning outputs into a clearly named results folder; document campaign history in Supplement | Traceable hyperparameter provenance | Results/Tuning; git c49eec1 | Not Started | Required (repo change) |
| E-04 | Experiment | Outcome-informed SHAP feature screening on a split overlapping test sets | Selection leakage | Show screening was outcome-independent, or move into folds; sensitivity analysis | Removes leakage concern | feature_importance.py:31-45 | Not Started | Required |
| E-05 | Experiment | Age-18 predictors (ACE18, SUD18) concurrent with outcome | Not prospective prediction; reverse causation | Re-run with ≤ age-15 predictors as primary/sensitivity | Clear prediction horizon | legacy/train.py:74-87 | Not Started | Required |
| E-06 | Experiment | No ablation showing fused stream adds value; gate "importance" not validated | C3 untested; attention ≠ explanation | Stream ablations + permuted-anchor + grouped permutation importance | Evidence for/against the interaction contribution | importance_analysis_avg.py; shap_analysis.py; GLMM_shap_updated.R; attribution_agreement.py | **Partly code-ready 2026-10-23** (gate vs SHAP vs grouped permutation, DCMFNet vs GLMM); stream ablations not yet written | Required |
| E-07 | Experiment | DCMFNet inputs unscaled | Gate comparability; optimization | Fold-wise standardization | Fair comparison | legacy/train.py:74 | Not Started | Required |
| E-09 | Analysis | No calibration metrics (only seed-42 scatter plots) | TRIPOD+AI; Van Calster 2019 | Calibration slope/intercept + smoothed plots | Calibration reported | — | Not Started | Required |
| A-02 | Analysis | Descriptives are for ~9,261, not the analysis sample (n = 4,033); no included vs excluded comparison | Selection bias under CCA (56% excluded) | Table 1 on analysis sample + comparison with excluded | TRIPOD+AI participant table | descriptive_statistics.R | Not Started | Required (needs data access) |
| M-02 | Methodology | Pos IGF depth: three conflicting specifications (Table B.1 "3"; json list; 5-seed list) | Reproducibility; interpretation mismatch | Author confirms which configuration is primary; document both | Consistent reporting | hyperparameters.json; Avg complexity results/hyperparameters.json | Not Started | Required (input) |
| M-03 | Methodology | SE-attention equations and claims do not match code | Methods accuracy | Rewrite as input-conditioned feature gating (done in draft §2.4) | Correct Methods | dcmfnet/model.py:50-77 | Completed (draft) | Not Required |
| M-04 | Methodology | Loss description differs from code (clip 20, no-grad focal, renormalization) | Methods accuracy | Rewrite from code (done in draft §2.5) | Correct Methods | dcmfnet/losses.py | Completed (draft) | Not Required |
| L-01 | Literature | AIIM paper analysis is abstract/metadata-level for 7 of 12 papers | Agent spec §4.2 requires full extraction | Author/institutional access to download PDFs into `Manuscript/draft/literature/pdfs/` for analysis | Complete comparative table | aiim_paper_analysis.md | Blocked (access) | Not Required |
| L-02 | Literature | Direct competitors (ML/DL with PRS + environment for psychosis/depressive outcomes) not yet identified | Novelty must be positioned against closest work | Targeted search (PubMed/Scopus via institution) | Accurate gap statement | reference_registry.md §F | Not Started | Not Required |
| E-16 | Experiment | Option B needs a methodological core beyond a negative benchmark (AIIM scope, Guide p.3) | A planted-interaction simulation separates "model cannot learn interactions at this n" from "data contain little interaction signal" | Semi-synthetic outcomes on the real predictor matrix with planted cross-group interactions of graded strength; DCMFNet vs additive models; see experiment_recommendations.md E-16 | Interaction-detectability threshold; central methods result | — | **Not pursued 2026-10-23** (BMC Psychiatry framing; not promised in the text) | Not Required |
| W-09 | Writing | Introduction (Option B) | Framing decided | Background rewritten 2026-10-23 in 02_background.md (BMC); AIIM version archived; contributions conditional on experiments; gap statement awaits L-02 | Intro draft | 02_background.md | In Progress | Not Required |
| C-04 | Compliance | Figures are 150-dpi PNGs; need vector/≥300 dpi, separate files, colour-blind-safe | Guide p.14–15 | Regenerate from saved CSVs as PDF/EPS | Compliant figures | legacy/train.py (dpi=150) | Not Started | Not Required (plots only from saved results) |
| C-06 | Compliance | TRIPOD+AI checklist not completed (thesis claims adherence) | Credibility; reviewer expectation | Complete checklist as Supplement after experiments | Transparent reporting | — | Not Started (Additional file 2; mandatory if transferred to DPR) | Not Required |

## Medium

| ID | Cat. | Issue | Justification | Proposed action | Expected outcome | Files | Status | Approval |
|---|---|---|---|---|---|---|---|---|
| A-03 | Analysis | `other_bullying15`/`other_abuse18` encode missing as −1 (kept by dropna) | Data integrity | Check frequency of −1 in analysis sample; recode or justify | Correct missing-data handling | feature_selection.py:116-130 | Not Started | Required |
| A-04 | Analysis | Sex-stratified performance (SAGER) | Guide p.8–9 | E-13 | Sex/gender dimension addressed | — | Not Started | Required |
| A-05 | Analysis | Verify LazyPredict preprocessing (scaling/imputation) vs DCMFNet raw inputs | Comparability (D-13) | Inspect lazypredict 0.3.0 source/version used | Documented preprocessing | lazy_predict.py | Not Started | Not Required |
| E-10 | Experiment | Tail-prediction claim untested; loss ablation absent | C5 | Pre-specified top-decile metrics + loss ablation | Evidence for/against C5 | — | Not Started | Required |
| E-12 | Experiment | Complete-case only | Selection bias | MI sensitivity analysis | Robustness | — | Not Started | Required |
| E-14 | Experiment | "Data-limited" explanation untested | Thesis Discussion claim | Learning curves | Supports/refutes ceiling claim | — | **Not pursued 2026-10-23** (BMC Psychiatry framing; discussed as an untested explanation) | Not Required |
| R-01 | Methodology | Script that creates `catss_final_data.csv` (merge of PRS/PCs, final columns) not in repo; PRS pipeline external | Reproducibility | Add script or document steps; describe PRS pipeline in Supplement | Reproducible pipeline | rename_catss_with_mapping.py | Not Started | Required (repo change) |
| M-05 | Methodology | Variables in GLMM differ from DCMFNet/ML (PC1/PC2 × batch factor vs batch_x_PC columns; ACE15 vs ACE grouping) | Input comparability | Tabulate exact inputs per model | Fairness documented | GLMM.R:22-36; lazy_predict.py:42-43 | Not Started | Not Required |
| W-02 | Writing | Twin-pair counts, exact n per split, number of features (105) need a participant flow diagram | TRIPOD+AI | Flow diagram from logs + author data | Figure 1 | best_model_training.log | Not Started | Not Required |
| W-03 | Writing | Methods §2.3 (PRS pipeline) depends on an unpublished master's thesis (Wang 2025) | AIIM discourages unpublished refs | Describe pipeline fully (QC, GWAS source, gene sets, PRS-CS params) | Self-contained Methods | — | Not Started | Required (input) |
| C-05 | Compliance | Code availability: tagged release + DOI; separate thesis-analysis code from synthetic/deployment code | Research data policy (Option B) | Create release after experiments | Citable code | repo | Not Started | Required |

## Low

| ID | Cat. | Issue | Proposed action | Status | Approval |
|---|---|---|---|---|---|
| W-04 | Writing | Thesis Table 5.2 Pos MAE SD shown as 0.000 (actual 0.0004) | Report 4 decimals | Completed (draft) | Not Required |
| W-05 | Writing | Thesis Table 5.3 labels L=2 as "Default" | Call it "tuned uniform depth" | Completed (draft) | Not Required |
| W-06 | Literature | Thesis reference metadata errors (Sullivan author order; DCMFNet venue; LightGBM/SE primary sources) | Use registry versions | Completed (registry) | Not Required |
| W-07 | Writing | Remove weak sources (Cureus review, meeting abstract) | Replace if claims needed | Not Started | Not Required |
| W-08 | Writing | README shows synthetic-data metrics as "Held-out test metrics" | Label as synthetic to avoid confusion with study results | Not Started | Required (repo change) |
| C-01 | Compliance | Check the BMC Psychiatry "Preparing your manuscript" page manually (not reachable by the agent); confirm every [VERIFY: JOURNAL REQUIREMENT] in 00_requirements_checklist.md | Manual check | Not Started | Not Required |
| J-01 | Compliance | Fallback-journal plan if AIIM rejects | Recorded in submission_readiness.md §Target journals (scope checked 2026-10-19; fees to re-check at transfer) | Completed | Not Required |
| E-15 | Experiment | External validation unavailable | State as limitation; name candidate cohorts | Not Started | Required (input) |

## Author decision 2026-10-22
The author has declined re-running models: "I don't see any need to re run any models. All the explanations are clearly mentioned in the master thesis report."
- Consequences: E-01–E-07, E-10–E-16 are set to **Declined (author)**. The related discrepancies stay open and must be disclosed in the manuscript (rule: discrepancies are never hidden).
- Still open for author approval: analyses of **saved results only** that refit no model (A-01/E-08 paired comparison from per-seed CSVs; E-09 calibration from saved predictions, if any exist).
- Pending author choice: how GLMM/GAMM results are handled without a refit (D-01, D-02). Options: drop them from the comparison, or report them as conducted with explicit disclosure.

## Summary (2026-10-19)
- Open Critical: **8** (E-01, E-02, E-03, M-01, A-01, C-02, C-03; W-01 in progress). X-00 completed (Option B).
- Open High: **14** (new: E-16, W-09 in progress; plus 2 completed in draft: M-03, M-04)
- Open Medium: **11** · Low: **4 open, 4 completed** (new: J-01)
- Requiring author approval/input: E-01–E-07, E-09, E-10, E-12, E-14, E-15, **E-16**, M-01, M-02, A-01–A-04, C-02, C-03, C-05, R-01, R-02, W-03, W-08
- **Option B minimum experiment package** (see experiment_recommendations.md §Option B): E-08 on saved CSVs → E-02 → E-01 → E-03/E-04 → E-06 → E-14 → E-16 → E-05/E-09/E-10.

## Author decision 2026-10-22 (GLMM/GAMM interactions)
The author decided that GLMM/GAMM interactions mirror the DCMFNet modality design (SUD15 anchor × each fused modality), not all 546 item-level products. They also asked whether to re-run the GLMM.
- The agent recommends keeping R, using the Python split IDs, and fitting one interaction per DCMFNet fusion block. The exact operationalisation awaits author choice: block summary scores, and how PRS is handled.
- Also align main effects with DCMFNet: add SUD18; split ACE into ACE15 and ACE18; use the same batch×PC columns (M-05).
- Code edits to GLMM.R / GAMM.R / the split export need explicit author authorisation. They must be run by someone with CATSS data access.

## Code created at author request 2026-10-22 (approved: "create a new GLMM_updated_split.R, GAMM_updated_split.R, split_data.py files")
- `Models/split_data.py`: exports the DCMFNet/ML twin-pair splits (seeds 42–46) to the working directory; all three scripts read and write in the folder they are run from. It checks the test-row counts against the final DCMFNet log.
- `Models/GLMM/Method/GLMM_updated_split.R` and `Models/GAMM/Method/GAMM_updated_split.R`:
  - use the shared splits;
  - DCMFNet-aligned main effects;
  - 24 interactions (SUD15 score × 7 modality scores, × SEX, × 16 cell-type PRS);
  - scores standardised with training-set statistics only;
  - R² as 1 − SSE/SST (D-22).
- Originals left unchanged. Smoke-tested on synthetic data only (scratchpad copy). Synthetic results are not study results.
- Open decisions, implemented as the agent's recommended defaults and changeable:
  - modality score = mean of standardised items;
  - PRS interactions kept per cell type.
- **Status:** E-01, E-02 → *Code ready; awaiting run on CATSS data by the author.*
- Follow-up scripts (approved 2026-10-22: "yes"), syntax-checked only, not run:
  - `Models/GLMM/Analysis/GLMM_Analysis_updated.R` and `Models/GAMM/Analysis/GAMM_Analysis_updated.R` read the `_updated` coefficient tables and group interactions by partner modality. The GAMM version adds a PRS smooth-term summary.
  - `Models/Model Comparison/models_comparison_plot_generation_updated.py` reads `GLMM_updated_summary.csv`, `GAMM_updated_summary.csv` and the DCMFNet `test_results_summary.csv` instead of hard-coded values.
  - Run order: split_data.py → GLMM/GAMM_updated_split.R → *_Analysis_updated.R → comparison plot.
- Attribution scripts (approved 2026-10-23: "yes, include the GLMM comparison too"), not run on study data:
  - `Models/DCMFNet/Analysis/shap_analysis.py`: retrains the 5 seeds (as importance_analysis_avg.py) and computes gates, SHAP (GradientExplainer, 100-row training background) and grouped permutation importance per modality.
  - `Models/GLMM/Analysis/GLMM_shap_updated.R`: exact interventional modality-level SHAP for the updated GLMM (interaction terms split by the two-player Shapley formula) and grouped permutation importance.
  - `Models/Model Comparison/attribution_agreement.py`: Spearman/top-3 agreement between sources, seed stability, and modality shares.
  - Checks: syntax; Shapley formula matches brute force on random numbers; the SHAP code path runs on an untrained DCMFNet with random inputs (shap 0.54.1 in a scratch folder).
  - Only aggregated values are saved; no row-level SHAP. Wording in the manuscript: model reliance, not causal effects.
- Data-access window (author, 2026-10-23): access ends in 5 days, so all data-dependent work runs now and the writing follows. Stream ablations deferred unless reviewers request them. Author has run split_data.py, GLMM_updated_split.R and GAMM_updated_split.R.
- Pre-expiry scripts (approved 2026-10-23: "go ahead with 1-3"), not run on study data:
  - `Data Preprocessing/Data Visualisation/analysis_sample_descriptives.R` → A-02, W-02, A-03 (flow, Table 1 included vs excluded with SMD, −1 code check; counts 1–4 shown as "<5").
  - `Models/DCMFNet/Analysis/dcmfnet_test_predictions.py` and `Models/Model Comparison/glmm_gamm_test_predictions.R` → test predictions for full and age-15-only predictor sets (E-05). Output files `SERVER_ONLY_predictions_*.csv` are participant-level and stay on the server.
  - `Models/Model Comparison/prediction_evaluation.py` → E-09 calibration, A-04 sex-stratified, E-10 top-decile metrics (metrics only; the loss ablation is not run), A-01 paired corrected resampled t-tests with Holm correction. Aggregated outputs only.
  - `build_model_spec` in both *_updated_split.R now drops empty modality blocks (needed for age15); no effect on the full model; 12 alignment tests pass.
  - Age-15 DCMFNet keeps the architecture and sets ACE18/SUD18 inputs to the training mean; GLMM/GAMM drop the columns (97 main effects, 22 interactions).
- Journal decision 2026-10-23: AIIM confirmed by the author, then **superseded the same day: BMC Psychiatry** (author added the Springer Nature/BMC LaTeX template and removed the AIIM guide; fallback by transfer: Diagnostic and Prognostic Research). JCE dropped: its exclusions cover prognostic research without a methods focus and single-country studies (author-supplied JCE guidance).

## Restructure for BMC Psychiatry (2026-10-23)
- Sections follow the BMC research-article format: Background, Methods, Results, Discussion (with Strengths and limitations), Conclusions, Abbreviations, Declarations. Checklist in `00_requirements_checklist.md`.
- New files: `02_background.md`, `03_methods.md`, `06_backmatter.md`. AIIM drafts moved to `archive/aiim/`.
- Methods rewritten to describe the analysis as now run: shared splits, updated LMM/GAMM, prediction evaluation, age-15 analysis, attribution analyses. Configuration selection is disclosed (D-04, D-09, D-21).
- Discussion and Conclusions drafted as prose. TODO paragraphs remain for attribution and for calibration/sensitivity results.
- LaTeX: `latex/build.py` turns the Markdown into `latex/manuscript.tex` (sn-jnl, sn-vancouver-num) and compiles the PDF. On 2026-10-23 it compiled with no errors or undefined citations (17 pages). `latex/references.bib` holds the 26 cited registry entries.
- Correction: the abstract had the lasso DEP R² as 0.235. That is LarsCV's value; LassoCV is 0.234 (Neg_benchmark_summary.csv). Fixed.
- New author decision needed: whether the comparator is called "LMM" or "GLMM" in the paper (it is Gaussian; the code says GLMM).
