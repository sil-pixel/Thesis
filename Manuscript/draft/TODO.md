# Manuscript backlog

Updated 2026-10-27. Target journal: BMC Psychiatry (fallback by transfer: Diagnostic and Prognostic Research). Data access ends 2026-10-28.
Cross-references: `discrepancies.md` (D-xx), `research/experiment_recommendations.md` (E-xx), `literature/reference_registry.md` (R-xx).
Q-xx are decisions for the author or supervisors (not to be confused with D-xx discrepancies). IDs are kept from the earlier backlog so that references in the draft and in `discrepancies.md` still resolve. The full history is in git.

# To do

## 1. Before data access ends (2026-10-28)

| ID | Task | Who | Notes |
|---|---|---|---|
| S-01 | Send the supervisor email (`supervisor_request.md`) with the dates filled in, and ask for any extra data analysis before the deadline | Author | Placeholders: names, draft date, request deadline |
| S-02 | Fill in the placeholders in `server_handover_README.md` (name, date, repo URL, final commit, contact after 28 Oct) and leave it on the server | Author | |
| G-01 | Commit the current work (draft files, `additional_file_1.md`, `additional_file_2.md`, `figures/`, `latex/build_additional.sh`, `attribution_importance_plot.py` and Fig. 4 files) | Author | Nothing has been committed by the agent |
| S-04 | Send the draft to the supervisors: `latex/manuscript.pdf` (figures on the last pages) and `latex/additional/additional_file_{1,2}.pdf` | Author | Rebuild first: `python3 latex/build.py` and `bash latex/build_additional.sh` |

## 2. Author or supervisor input (placeholders in the draft)

| ID | Task | Where in the draft |
|---|---|---|
| M-01 | Final outcome labels (PLE-HM, DEP are working labels); instruments, scales, reverse coding and the divisors 46 and 33 resolved 2026-10-28 | Methods; Discussion limitation; Table 1 |
| M-06 | Birth years and questionnaire collection years: whole cohort or final sample? (asked in the supervisor email, item 1). A-TAC, informant, sex source and data dates resolved 2026-10-28 | Methods |
| W-03 | Genotype and PRS paragraph drafted in Methods 2026-10-28; supervisor to confirm and fill 7 VERIFY points (arrays, QC, imputation, GWAS, PRS-CS settings, SNP-to-gene mapping, ancestry exclusion), or give a citable description (email item 2) | Methods: Genotype data and polygenic scores |
| C-02 | Ethics approvals (committee, reference numbers, dates) and consent procedure | Methods; Declarations |
| C-03 | Authors, affiliations, corresponding author, CRediT contributions, funding, competing interests, acknowledgements, APC coverage | Title page; Declarations |
| C-07 | Data access procedure/URL for CATSS | Declarations (availability of data) |
| C-08 | AI-use statement written in Methods (author's wording, extended to cover drafting and scripts); author to confirm before submission | Methods: Use of AI tools |
| Q-02 | Decide whether the tuned ML comparators stay in the main tables (current draft) or move to the supplement | Results Tables 3–5 |
| Q-03 | Choose title and keywords | Title page |
| Q-04 | Confirm the wording of the configuration-selection disclosure (E-03) | Methods; Discussion limitations |
| Q-05 | Coding direction of SES and ACE18 items, and that higher SCZ15 values mean more symptoms | Results (Table 1 text) |
| Q-06 | Item response scale per predictor group for the Table 1 footnote, or move group rows to Additional file 1 | Table 1 footnote |

## 3. Writing and references (agent can do; no data needed)

| ID | Task | Notes |
|---|---|---|
| C-10 | Optional: one sentence on known health inequalities (TRIPOD+AI 3c), e.g. sex differences in these symptoms, with a verified citation | Background |
| C-01 | Check the BMC Psychiatry submission guidelines by hand and resolve every `[VERIFY: JOURNAL REQUIREMENT]` (abstract limit, declarations, clinical trial number) | `00_requirements_checklist.md` |
| R-02 | Restore the outputs of the used tuning run (git c49eec1) into the repo, next to the later run | Needs author approval (repo change); described in Additional file 1: Table S1 |
| C-05 | Tagged GitHub release archived on Zenodo for a DOI (author: link Zenodo, publish release, send DOI); README manuscript section with run order added 2026-10-29 | Methods; Declarations |
| R-01 | Script or documented steps that create `catss_final_data.csv` (merge of PRS and PCs) | Needs author approval for repo changes |

## 4. Not pursued (disclosed as limitations; can be revisited if reviewers ask)

| ID | Analysis | Status in the draft |
|---|---|---|
| E-03 | Nested, test-free selection of the DCMFNet configuration | Selection on test data disclosed in Methods and limitations; the ML comparators were tuned without test data |
| E-04 | Moving the feature screening inside the training folds | Disclosed as a limitation |
| E-06 | Stream ablations (fused vs independent stream) and permuted-anchor test | Attribution analysis done instead; ablations not run |
| E-07 | Standardised inputs for DCMFNet | Methods states that inputs were not standardised |
| E-10 | Ablation of the tail-weighted loss | Top-decile metrics reported; ablation not run |
| E-12 | Multiple imputation sensitivity analysis | Complete-case limitation with Table 1 comparison |
| E-14 | Learning curves | Discussed as an untested explanation |
| E-15 | External validation | Stated as a limitation (single cohort) |
| E-16 | Planted-interaction simulation | Not promised in the text |

# Done

| ID | Item | Date |
|---|---|---|
| X-00 | Framing chosen: rigorous benchmark (Option B); journal BMC Psychiatry, fallback Diagnostic and Prognostic Research (J-01) | 2026-10-19 / 2026-10-23 |
| E-01 | LMM and GAMM refitted with 24 interactions that mirror the DCMFNet fusion blocks | 2026-10-23 |
| E-02 | All models on the same five twin-pair-grouped splits (`split_data.py`; all seeds match the DCMFNet log) | 2026-10-23 |
| M-05 | LMM/GAMM main effects aligned with the DCMFNet inputs (SUD18, ACE15/ACE18, batch × PC columns) | 2026-10-23 |
| A-01 | Paired corrected resampled t-tests with Holm correction (main family and comparator family) | 2026-10-24 / 2026-10-25 |
| A-02 | Analysis-sample flow and Table 1, included vs excluded with SMD | 2026-10-24 |
| A-03 | No −1 or other negative codes in the analysis sample | 2026-10-24 |
| A-04 | Sex-stratified performance | 2026-10-24 |
| E-05 | Age-15-only predictor set for all models | 2026-10-24 |
| E-09 | Calibration-in-the-large, slope, decile curves (Fig. 3) | 2026-10-24 |
| E-10a | Top-decile precision, AUROC and bias | 2026-10-24 |
| E-06a | Gates vs SHAP vs permutation importance for DCMFNet, LMM and GAMM; agreement and stability; Fig. 4 | 2026-10-24 / 2026-10-25 |
| D-23 | Retrained DCMFNet reproduces the final run (seeding order fixed with `train_seed_models()`) | 2026-10-24 |
| E-17 | Tuned ML comparators (linear regression, elastic net, gradient boosting) with twin-grouped inner CV, scored with the same code | 2026-10-25 |
| A-05 | LazyPredict screening: estimators called directly, inputs unscaled (stated in Methods) | 2026-10-25 |
| A-06a | Additional file 1: Tables S2 and S2b | 2026-10-25 |
| L-03 | Missing citations added with verified metadata: R-ADAM, R-ZOU, R-SKLEARN, R-ROBERTS (grouped CV), R-CEDERLOF (adolescent PLEs, CATSS); R-PLOEG replaces R-RILEY for the sample-size argument | 2026-10-27 |
| L-04 | Content checks (abstracts; Kapoor full text) for R-DIFORTI, R-AGERBO, R-PGC3, R-YAO2025, R-ROBINSON, R-KAPOOR; Background wording adjusted to what the sources show | 2026-10-27 |
| A-06 | Additional file 1 complete: Tables S1 (search + tuning history, from git c49eec1), S3a–c (all GLMM/GAMM terms, seed 42), S4 (105 predictors), S5 (depth sweep, seed 42) | 2026-10-27 |
| C-04 | Figures: Fig. 1 flow, Fig. 2 architecture, Fig. 3 redrawn with manuscript labels (colour-blind safe, markers), Fig. 4 copied; `figures/make_figures.py`; appended to the review PDF | 2026-10-27 |
| C-06 | TRIPOD+AI checklist (Additional file 2), mapped item by item; gaps listed as C-09, C-10, Q-08 | 2026-10-27 |
| W-10 | Read-through: abstract, Results, Discussion and email checked against the CSVs; one rounding error fixed (DEP males, elastic net 0.10 not 0.11); sample-size and not-for-clinical-use sentences added | 2026-10-27 |
| Q-07 / D-24 | `SCZ15_PROD_seen_hallucinations9` is an age-15 item with a naming error (author); noted in Table S4 | 2026-10-27 |
| V-01 / V-04 | Version files exported (`versions_python.txt`, `versions_R.txt`); software versions and the GAMM basis (thin-plate, k = 10, mgcv default) filled in Methods. Author to confirm these are the environments the analyses ran in | 2026-10-28 |
| L-02 | "No direct competitor" statement checked by a PubMed block search (9 hits; PMIDs 41399528, 36411277, 34899225 read by the author, no conflict); VERIFY markers in Background and Discussion replaced by a search note | 2026-10-28 |
| W-02 | Participant flow starts at 13,146 CATSS twins (thesis Figs 3.1, 3.3; author confirmed); 3,885 excluded for missing exposure or outcome; Fig. 1 and Methods updated. 9,160 genotyped (of 13,146) vs 6,643 with PRS (of 9,261): different denominators | 2026-10-28 |
| W-08 | README restructured: manuscript section (summary, folders, run order), thesis abstract with a note on the re-analysis, and Bodhica synthetic-data section with metrics labelled as synthetic | 2026-10-29 |
| Q-01 | "GLMM" used throughout (author decision); Methods notes that with a Gaussian outcome it is a linear mixed model | 2026-10-27 |
| Q-08 | TRIPOD+AI 18c, 18d, 19: no published protocol, not registered, no patient and public involvement (author); stated in Methods | 2026-10-27 |
| C-09 | 95% CIs for the paired R² differences added to Table 4 (corrected resampled t, from paired_comparisons.csv); Methods, Results and checklist item 23a updated | 2026-10-27 |
| M-02 | Table 2 matches the final configuration in `hyperparameters.json` (author confirmed); parameter counts recomputed from the code (105,501 / 71,623) | 2026-10-26 |
| E-04a | Variable screening described from `feature_selection.py` and the LightGBM/SHAP step; treated as outcome-informed (limitation kept) | 2026-10-26 |
| M-03, M-04 | Gating and loss described from the code | 2026-10-19 |
| W-01 | Thesis claims not carried over; the paper does not mention the thesis | 2026-10-25 |
| W-09 | Background, including the gap statement with the L-02 studies | 2026-10-25 |
| W-11 | Results, Discussion (incl. strengths and limitations), Conclusions and abstract drafted from the verified results | 2026-10-25 |
| W-04–W-07 | Thesis table and reference errors corrected; weak sources not cited | 2026-10-19 |
| S-03 | Supervisor email and server handover note drafted | 2026-10-24 / 2026-10-25 |
| L-01 | AIIM paper analysis: no longer needed after the journal change | dropped 2026-10-23 |
