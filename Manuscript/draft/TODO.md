# Manuscript backlog

Updated 2026-10-26. Target journal: BMC Psychiatry (fallback by transfer: Diagnostic and Prognostic Research). Data access ends 2026-10-28.
Cross-references: `discrepancies.md` (D-xx), `research/experiment_recommendations.md` (E-xx), `literature/reference_registry.md` (R-xx).
Q-xx are decisions for the author or supervisors (not to be confused with D-xx discrepancies). IDs are kept from the earlier backlog so that references in the draft and in `discrepancies.md` still resolve. The full history is in git.

# To do

## 1. Before data access ends (2026-10-28)

| ID | Task | Who | Notes |
|---|---|---|---|
| V-01 | Save `versions_R.txt` (`sessionInfo()`) and `versions_python.txt` (`pip freeze`) on the server and export them | Author | Needed for the software versions in Methods (Python, PyTorch, scikit-learn, Optuna, SHAP, R, lme4, mgcv) and the GAMM basis check (V-04) |
| S-01 | Send the supervisor email (`supervisor_request.md`) with the dates filled in, and ask for any extra data analysis before the deadline | Author | Placeholders: names, draft date, request deadline |
| S-02 | Fill in the placeholders in `server_handover_README.md` (name, date, repo URL, final commit, contact after 28 Oct) and leave it on the server | Author | |
| G-01 | Commit the current work (draft files, new scripts, `attribution_importance_plot.py` and Fig. 4 files, `additional_file_1.md`) | Author | Nothing has been committed by the agent |

## 2. Author or supervisor input (placeholders in the draft)

| ID | Task | Where in the draft |
|---|---|---|
| M-01 | Outcome instruments, item response scales, the constants 46 and 33, and final outcome labels (PLE-HM, DEP are working labels) | Methods; Discussion limitation; Table 1 |
| M-06 | Age-9 instruments and informants (A-TAC?), cohort/birth years, data extraction date; sex from register or self-report | Methods |
| W-02 | Participant flow before the 9,261 eligible (thesis: 13,146 at start; 9,160 genotyped vs 6,643 with PRS) | Methods; Fig. 1 |
| W-03 | PRS pipeline: GWAS summary statistics and their ancestry, LD reference, PRS-CS settings, genotyping platforms, imputation, QC | Methods; Discussion (generalisability) |
| C-02 | Ethics approvals (committee, reference numbers, dates) and consent procedure | Methods; Declarations |
| C-03 | Authors, affiliations, corresponding author, CRediT contributions, funding, competing interests, acknowledgements, APC coverage | Title page; Declarations |
| C-07 | Data access procedure/URL for CATSS | Declarations (availability of data) |
| C-08 | Statement on the use of large language models, if the journal requires it | Methods `[VERIFY: JOURNAL REQUIREMENT]` |
| Q-01 | Decide "LMM" or "GLMM" in the paper; then relabel tables and figures (Fig. 4 labels say GLMM) | Methods, Results, Fig. 3–4 |
| Q-02 | Decide whether the tuned ML comparators stay in the main tables (current draft) or move to the supplement | Results Tables 3–5 |
| Q-03 | Choose title and keywords | Title page |
| Q-04 | Confirm the wording of the configuration-selection disclosure (E-03) | Methods; Discussion limitations |
| Q-05 | Coding direction of SES and ACE18 items, and that higher SCZ15 values mean more symptoms | Results (Table 1 text) |
| Q-06 | Item response scale per predictor group for the Table 1 footnote, or move group rows to Additional file 1 | Table 1 footnote |

## 3. Writing and references (agent can do; no data needed)

| ID | Task | Notes |
|---|---|---|
| L-02 | Confirm the "no direct competitor" statement with a PubMed/Scopus search | `[VERIFY]` in Background and Discussion; current evidence in reference_registry.md §G (non-systematic search 2026-10-25) |
| L-03 | Missing citations: Kingma & Ba 2015 (Adam); Zou & Hastie 2005 (elastic net); Pedregosa et al. 2011 (scikit-learn); twin/family dependence and grouped cross-validation; adolescent psychotic-like experiences and later psychopathology (candidate Kelleher/Cederlöf 2017) | Each needs Crossref metadata and a content check before use |
| L-04 | Content checks for references cited with `[VERIFY: content]`: R-DIFORTI, R-AGERBO, R-PGC3, R-YAO2025, R-ROBINSON, R-KAPOOR, R-RILEY | Read abstract or full text; keep or replace |
| A-06 | Additional file 1: Table S1 (Optuna search space), Table S3 (full LMM and GAMM coefficients), Table S4 (variables by predictor group), Table S5 (depth-configuration results), tuning history (D-21) | S2 and S2b done; all from exported files |
| R-02 | Tuning provenance: the used hyperparameters come from git commit c49eec1, and there were several tuning campaigns (D-21) | Describe in Additional file 1; restoring the c49eec1 outputs into the repo needs author approval |
| C-04 | Figures: Fig. 1 flow diagram (from `analysis_sample_flow.csv` once W-02 is known); Fig. 2 architecture diagram from code; check that Fig. 3–4 are vector and colour-blind safe | Fig. 3 (`calibration_curves_full.pdf`) and Fig. 4 (`attribution_importance_by_group.pdf`) exist |
| C-06 | TRIPOD+AI checklist (Additional file 2) | Mandatory if transferred to Diagnostic and Prognostic Research |
| C-01 | Check the BMC Psychiatry submission guidelines by hand and resolve every `[VERIFY: JOURNAL REQUIREMENT]` (abstract limit, declarations, clinical trial number) | `00_requirements_checklist.md` |
| V-04 | Fill in software versions and the GAMM basis dimension from V-01 | Methods |
| C-05 | Tagged code release with DOI; separate the analysis code from synthetic-data and deployment code | Methods; Declarations |
| R-01 | Script or documented steps that create `catss_final_data.csv` (merge of PRS and PCs) | Needs author approval for repo changes |
| W-08 | README shows synthetic-data metrics as "held-out test metrics"; label them as synthetic | Needs author approval (repo change) |
| W-10 | Final read-through: consistency of numbers between abstract, Results, Discussion and tables; abstract stays ≤ 350 words (now 348) | Before sending to supervisors |

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
| M-02 | Table 2 matches the final configuration in `hyperparameters.json` (author confirmed); parameter counts recomputed from the code (105,501 / 71,623) | 2026-10-26 |
| E-04a | Variable screening described from `feature_selection.py` and the LightGBM/SHAP step; treated as outcome-informed (limitation kept) | 2026-10-26 |
| M-03, M-04 | Gating and loss described from the code | 2026-10-19 |
| W-01 | Thesis claims not carried over; the paper does not mention the thesis | 2026-10-25 |
| W-09 | Background, including the gap statement with the L-02 studies | 2026-10-25 |
| W-11 | Results, Discussion (incl. strengths and limitations), Conclusions and abstract drafted from the verified results | 2026-10-25 |
| W-04–W-07 | Thesis table and reference errors corrected; weak sources not cited | 2026-10-19 |
| S-03 | Supervisor email and server handover note drafted | 2026-10-24 / 2026-10-25 |
| L-01 | AIIM paper analysis: no longer needed after the journal change | dropped 2026-10-23 |
