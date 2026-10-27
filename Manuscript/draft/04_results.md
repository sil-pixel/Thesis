<!--
STATUS 2026-10-24: FULL DRAFT of Results, BMC Psychiatry structure (unnumbered headings; Tables 1–2 are in Methods, so Results tables start at 3).
All model results come from the shared splits and prediction_evaluation.py; DCMFNet was retrained and reproduces the
Final Best Model (D-23 resolved). The DCMFNet configuration-selection issue (E-03) is not resolved and must be
disclosed with these results. Synthetic-data results (README; artifacts/) are excluded by design (D-17).
Outcome labels PLE-HM and DEP are working labels pending M-01. Not yet done: stream ablations (E-06 remainder).
-->

# Results

## Analysis sample
The analysis sample comprised 4,033 individuals from 2,690 twin pairs with complete data on all 105 predictors and both outcomes. Each of the five test sets held 25% of the twin pairs, between 998 and 1,028 individuals. <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1-3, 361, 685, 1063, 1414 -->
Of the 9,261 eligible individuals (5,756 twin pairs), 4,033 (43.5%) had complete data (Fig. 1). In the analysis sample, 1,343 twin pairs contributed both twins and 1,347 contributed one. Missing genotype data was the largest single reason for exclusion (2,618 individuals), followed by missing data on symptoms at age 15 (907), adverse experiences at age 18 (626) and parental socioeconomic indicators (526). Each individual is counted at the first step they failed, in the order shown in Fig. 1. <!-- src: Data Preprocessing/Data Analysis/Sample Analysis/analysis_sample_flow.csv; analysis_sample_twin_structure.csv -->

Included and excluded participants were broadly similar (Table 1).
- **Outcomes:** the analysis sample had slightly lower mean outcome scores (PLE-HM 0.132 vs 0.145; DEP 0.243 vs 0.266; SMD −0.13 for both).
- **Symptoms at age 15:** the largest difference was in the SCZ15 item mean (0.414 vs 0.476; SMD −0.22). `[VERIFY: higher SCZ15 item values = more symptoms]`
- **Other groups:** the item means for parental socioeconomic indicators (SMD 0.19) and adverse experiences at age 18 (SMD 0.18) also differed. `[VERIFY: coding direction of SES and ACE18 items before interpreting the sign]`
- **Substance use at age 15:** all six items differed little (|SMD| ≤ 0.10).
- **Sex:** missing for 26.0% of excluded participants. Among those with known sex, 58.0% of included and 60.6% of excluded participants were female.

<!-- src: Data Preprocessing/Data Analysis/Sample Analysis/table1_included_vs_excluded.csv. 60.6% = 2,345 / (1,526 + 2,345), computed from the exported counts. SMD = included minus excluded; for sex it was computed among those with known sex. -->

**Table 1.** Characteristics of the analysis sample and of excluded eligible participants.

| Characteristic | Included (n = 4,033) | Excluded (n = 5,228) | SMD |
|---|---|---|---|
| Twin pairs, n | 2,690 | 3,704 | |
| Female, n (%) | 2,338 (58.0) | 2,345 (44.9) | −0.05 |
| Male, n (%) | 1,695 (42.0) | 1,526 (29.2) | |
| Sex missing, n (%) | 0 (0.0) | 1,357 (26.0) | |
| PLE-HM at 18, mean (SD) | 0.132 (0.101) | 0.145 (0.109) | −0.13 |
| PLE-HM at 18, median [IQR] | 0.130 [0.043, 0.196] | 0.130 [0.065, 0.217] | |
| DEP at 18, mean (SD) | 0.243 (0.175) | 0.266 (0.186) | −0.13 |
| DEP at 18, median [IQR] | 0.212 [0.121, 0.333] | 0.212 [0.121, 0.394] | |
| Cigarettes at 15, mean (SD) | 0.321 (0.834) | 0.404 (0.912) | −0.10 |
| Snus at 15, mean (SD) | 0.166 (0.604) | 0.183 (0.642) | −0.03 |
| Alcohol at 15, mean (SD) | 3.450 (0.979) | 3.401 (1.005) | 0.05 |
| Cannabis at 15, mean (SD) | 0.061 (0.316) | 0.069 (0.334) | −0.02 |
| Non-medical painkillers at 15, mean (SD) | 0.115 (0.489) | 0.120 (0.489) | −0.01 |
| Other drugs at 15, mean (SD) | 0.083 (0.375) | 0.094 (0.393) | −0.03 |
| PRS (16), item mean (SD) | −0.088 (0.095) | −0.081 (0.096) | −0.07 |
| SCZ15 (23), item mean (SD) | 0.414 (0.263) | 0.476 (0.294) | −0.22 |
| ADHD9 (19), item mean (SD) | 0.152 (0.258) | 0.168 (0.270) | −0.06 |
| ASD9 (17), item mean (SD) | 0.066 (0.133) | 0.075 (0.148) | −0.06 |
| ACE15 (7), item mean (SD) | 1.199 (0.479) | 1.239 (0.532) | −0.08 |
| ACE18 (4), item mean (SD) | 0.902 (0.152) | 0.873 (0.174) | 0.18 |
| SUD18 (4), item mean (SD) | 0.906 (0.874) | 0.939 (0.920) | −0.04 |
| SES (4), item mean (SD) | 2.594 (0.460) | 2.501 (0.513) | 0.19 |

<!-- src: table1_included_vs_excluded.csv. Group rows: mean of the raw items per person, then mean (SD) over people; excluded values use the people with data on that group. Number of items in parentheses. Batch × PC terms omitted (technical covariates; SMD 0.08). Missing-data percentages per group in the excluded set (PRS 50.1%, SCZ15 48.5%, SES 31.8%, ASD9 28.3%, ADHD9 26.3%, ACE18 19.6%, ACE15 8.2%, SUD18 0.5%) can go in a table footnote or Additional file. -->

*Footnote to Table 1:* SMD, standardised mean difference (included minus excluded); for sex, computed among participants with known sex. Item scales differ between groups `[AUTHOR INPUT: response scale per group, or move group rows to Additional file 1]`. Percentages for sex use all participants in each column as the denominator.

## Predictive performance

All models were evaluated on the same five twin-pair-grouped train/test splits, and all predictions were scored with the same code. R² is 1 − SSE/SST throughout, and SDs are across the five splits (population SD). <!-- src: Data Preprocessing/split_summary.csv (matches_dcmfnet_log = True for all seeds); Models/Model Comparison/prediction_evaluation.py core_metrics(), summarise() -->

**Table 3.** Test-set performance, mean ± SD over five splits.

| Model | R² | RMSE | MAE | Spearman ρ |
|----------------|----------------|----------------|----------------|----------------|
| *PLE-HM* | | | | |
| DCMFNet | 0.2058 ± 0.0189 | 0.0903 ± 0.0013 | 0.0712 ± 0.0004 | 0.4497 ± 0.0193 |
| GLMM | 0.2279 ± 0.0156 | 0.0890 ± 0.0014 | 0.0700 ± 0.0012 | 0.4641 ± 0.0183 |
| GAMM | 0.2274 ± 0.0153 | 0.0891 ± 0.0014 | 0.0699 ± 0.0011 | 0.4633 ± 0.0180 |
| Linear regression | 0.2393 ± 0.0212 | 0.0884 ± 0.0017 | 0.0695 ± 0.0012 | 0.4728 ± 0.0215 |
| Elastic net | 0.2603 ± 0.0169 | 0.0871 ± 0.0016 | 0.0691 ± 0.0010 | 0.4915 ± 0.0182 |
| Gradient boosting | 0.2441 ± 0.0188 | 0.0881 ± 0.0016 | 0.0700 ± 0.0008 | 0.4792 ± 0.0194 |
| *DEP* | | | | |
| DCMFNet | 0.1984 ± 0.0208 | 0.1562 ± 0.0017 | 0.1242 ± 0.0019 | 0.4509 ± 0.0135 |
| GLMM | 0.2025 ± 0.0220 | 0.1559 ± 0.0024 | 0.1226 ± 0.0016 | 0.4460 ± 0.0154 |
| GAMM | 0.1998 ± 0.0230 | 0.1561 ± 0.0026 | 0.1228 ± 0.0017 | 0.4421 ± 0.0163 |
| Linear regression | 0.2150 ± 0.0253 | 0.1546 ± 0.0026 | 0.1216 ± 0.0017 | 0.4531 ± 0.0203 |
| Elastic net | 0.2373 ± 0.0146 | 0.1524 ± 0.0024 | 0.1197 ± 0.0012 | 0.4733 ± 0.0193 |
| Gradient boosting | 0.2287 ± 0.0163 | 0.1533 ± 0.0027 | 0.1203 ± 0.0017 | 0.4645 ± 0.0194 |

<!-- src: Models/Model Comparison/prediction_metrics_summary.csv (predictor_set = full). DCMFNet rows are from the retrained models, which reproduce Models/DCMFNet/Results/Final Best Model/test_results_summary.csv (D-23, resolved 2026-10-24); GLMM/GAMM rows equal GLMM_updated_summary.csv / GAMM_updated_summary.csv. Linear regression, elastic net and gradient boosting: ml_comparators_test_predictions.py; linear regression equals the lazy_predict.py screening per split to 4 decimals (Models/ML/Results/{Pos,Neg}_benchmark_per_seed.csv). -->
<!-- note: DCMFNet configuration was selected partly on test data (D-04, D-09, D-21; TODO E-03), so its row may be optimistic. GLMM/GAMM were not tuned. The ML comparators were tuned inside the training splits only. -->

Mean R² was higher for the GLMM and GAMM than for DCMFNet on both outcomes. The difference was 0.022 for PLE-HM and 0.004 for DEP (GLMM minus DCMFNet). GLMM and GAMM performed almost identically. For DEP, DCMFNet had a higher Spearman correlation than the mixed models (0.451 vs 0.446 and 0.442). <!-- src: same file -->

All three machine-learning comparators had higher mean R² than DCMFNet and the mixed models for both outcomes:
- **elastic net:** 0.260 (PLE-HM) and 0.237 (DEP), the best of all models;
- **gradient boosting:** 0.244 and 0.229;
- **linear regression without penalty:** 0.239 and 0.215.

The comparators also had the highest Spearman correlations (0.47–0.49 for PLE-HM, 0.45–0.47 for DEP). <!-- src: prediction_metrics_summary.csv (full) -->

The tuning favoured strong shrinkage and simple structure:
- **Elastic net:** the selected L1 ratio was 0.05, close to ridge regression, in 15 of 20 fits.
- **Gradient boosting:** the selected trees had a maximum depth of 2 or 3 in 14 of 20 fits, and at most 6 leaves in the other six.

<!-- src: Models/ML/Results/ML_comparators_tuning.csv (20 fits per model = 5 splits × 2 outcomes × 2 predictor sets; boosting max_depth 2 or 3 in 14 fits; depth 4 with 6 leaves or depth 6 with 4 leaves in the others) -->

In the default-settings screening of 40 algorithms, cross-validated penalised linear models ranked highest (lasso R² 0.261 for PLE-HM; LARS 0.235 for DEP), close to the tuned elastic net (Additional file 1: Table S2). Several algorithms failed at their default settings, so the screening does not represent what these model classes can achieve. <!-- src: Models/ML/Results/{Pos,Neg}_benchmark_summary.csv (LassoCV 0.2614; LarsCV 0.2346; default Lasso/ElasticNet R² −0.001/−0.002, same as DummyRegressor; default MLPRegressor −0.36/−0.34) -->

## Paired comparisons across splits

The models were compared split by split with the corrected resampled t-test [@R-NADEAU]. P-values were Holm-adjusted within each metric across 18 comparisons:
- the three model pairs, with each predictor set, for both outcomes (12 comparisons);
- full versus age-15-only predictors for each model and outcome (6 comparisons).

Comparisons of DCMFNet with the three machine-learning comparators formed a second family of 18 comparisons, with its own Holm correction:
- DCMFNet versus each comparator, with each predictor set, for both outcomes (12 comparisons);
- full versus age-15-only predictors for each comparator and outcome (6 comparisons).

<!-- src: Models/Model Comparison/paired_comparisons.csv (18 rows per metric); prediction_evaluation.py paired_comparisons(), holm() -->

**Table 4.** Paired differences in test R² with all predictors (first model minus second), five splits.

| Comparison | Mean ΔR² (SD) | 95% CI | Splits where first model was better | p | p (Holm) |
|----------------------|----------------|-------------------|----------|------|---------|
| *PLE-HM* | | | | | |
| DCMFNet − GLMM | −0.0221 (0.0133) | −0.0491 to 0.0049 | 1 of 5 | 0.085 | 0.91 |
| DCMFNet − GAMM | −0.0216 (0.0129) | −0.0477 to 0.0045 | 1 of 5 | 0.083 | 0.91 |
| GLMM − GAMM | 0.0005 (0.0007) | −0.0010 to 0.0020 | 4 of 5 | 0.40 | 1.00 |
| *DEP* | | | | | |
| DCMFNet − GLMM | −0.0040 (0.0253) | −0.0553 to 0.0472 | 2 of 5 | 0.84 | 1.00 |
| DCMFNet − GAMM | −0.0013 (0.0270) | −0.0561 to 0.0534 | 2 of 5 | 0.95 | 1.00 |
| GLMM − GAMM | 0.0027 (0.0025) | −0.0024 to 0.0078 | 4 of 5 | 0.22 | 1.00 |
| ***Comparators (separate Holm family)*** | | | | | |
| *PLE-HM* | | | | | |
| DCMFNet − linear regression | −0.0335 (0.0162) | −0.0665 to −0.0006 | 0 of 5 | 0.048 | 0.33 |
| DCMFNet − elastic net | −0.0545 (0.0141) | −0.0831 to −0.0260 | 0 of 5 | 0.006 | 0.085 |
| DCMFNet − gradient boosting | −0.0383 (0.0106) | −0.0598 to −0.0168 | 0 of 5 | 0.008 | 0.094 |
| *DEP* | | | | | |
| DCMFNet − linear regression | −0.0166 (0.0270) | −0.0714 to 0.0382 | 2 of 5 | 0.45 | 0.45 |
| DCMFNet − elastic net | −0.0389 (0.0201) | −0.0796 to 0.0019 | 0 of 5 | 0.057 | 0.34 |
| DCMFNet − gradient boosting | −0.0303 (0.0217) | −0.0743 to 0.0137 | 0 of 5 | 0.13 | 0.34 |

*Footnote to Table 4:* 95% CI, confidence interval for the mean difference from the corrected resampled t-test (4 degrees of freedom), not adjusted for multiple comparisons; an interval excludes 0 exactly when the unadjusted p is below 0.05.

<!-- src: paired_comparisons.csv, family = baselines, metric = r2, setting "(full)"; DCMFNet is setting_a, so signs and seeds_a_better are as in the file. -->
<!-- CI computed 2026-10-27 from paired_comparisons.csv: mean ± t(0.975, 4) × sd_difference × sqrt(1/5 + 1/3) (= mean / t_corrected, checked to 1e-15); signs reversed with the comparison for the main family. -->
<!-- src: paired_comparisons.csv, metric = r2, setting "(full)". The file reports GLMM/GAMM minus DCMFNet; signs reversed here and "splits better" = 5 − seeds_a_better (no ties). -->

**DCMFNet and the mixed models.** None of the differences was statistically significant, before or after correction. RMSE, MAE and Spearman's ρ gave the same pattern (all Holm-adjusted p ≥ 0.90).

**DCMFNet and the comparators.** DCMFNet had lower R² than each comparator on every split, except against linear regression for DEP (better on 2 of 5 splits). The differences were −0.034 to −0.055 for PLE-HM (unadjusted p 0.006–0.048) and −0.017 to −0.039 for DEP (p 0.057–0.45). None remained significant after Holm correction; the smallest adjusted p was 0.085. <!-- src: paired_comparisons.csv, family = baselines, metric = r2 (full) -->

**Interpretation.** With five splits these tests have little power, so the results show no detectable advantage for DCMFNet. They do not show that the models are equivalent: for DEP, the confidence intervals for DCMFNet against the mixed models included differences of about 0.05 in R² in either direction, and for PLE-HM they ranged from about 0.05 lower to 0.005 higher for DCMFNet. <!-- src: paired_comparisons.csv, metrics rmse, mae, spearman_rho, setting (full) -->

## Interaction and smooth terms in the mixed models

Coefficients come from the fits on the first split (seed 42); all terms are given in Additional file 1: Table S3. The GLMM and GAMM each included 24 SUD15 interaction terms:
- SUD15 score × each fused modality score (7);
- SUD15 score × sex;
- SUD15 score × each of the 16 cell-type PRS.

<!-- src: Models/GLMM/Results/{Pos,Neg}_GLMM_updated_all_coefs.csv (24 interaction rows each) -->

- **PLE-HM.** Three of the 24 interactions had unadjusted p < 0.05 in both the GLMM and the GAMM: SUD15 × the PRS for medium spiny neurons, MGE interneurons and upper-layer intratelencephalic neurons (|β| 0.005–0.006, p 0.018–0.038).
- **DEP.** One interaction had unadjusted p < 0.05: SUD15 × SCZ15 (β = −0.006, p = 0.017 in the GLMM).
- **Multiple testing.** About 1.2 terms in 24 would reach p < 0.05 by chance alone. None of these terms remains significant after correction for 24 tests (smallest Holm-adjusted p ≥ 0.40).
- We therefore found no robust evidence of SUD15 interactions in the statistical models.

<!-- src: Models/GLMM/Results/{Pos,Neg}_GLMM_updated_significant_table.csv; Models/GAMM/Results/GAMM_updated_{Pos,Neg}_significant_table.csv -->

In the GAMM, 11 of the 16 PRS smooths were close to linear for each outcome (effective degrees of freedom < 1.1). The maximum was 2.54 for PLE-HM and 1.92 for DEP. Smooths with unadjusted p < 0.05:
- **PLE-HM:** the all-cell PRS and the PRS for hippocampal CA1–3 and miscellaneous cells;
- **DEP:** the all-cell PRS only.

<!-- src: Models/GAMM/Results/GAMM_updated_{Pos,Neg}_smooth_terms.csv -->

<!-- MOVED to Additional file (not a main result for BMC Psychiatry):
## 3.5. Depth configuration (as conducted; 5 seeds)
With depth set per block rather than uniformly, test RMSE was 0.0898 ± 0.0015 vs 0.0905 ± 0.0010 for PLE-HM (uniform L = 3) and 0.1569 ± 0.0014 vs 0.1569 ± 0.0028 for DEP (uniform L = 2). ( src: Models/DCMFNet/Results/Complexity Analysis/Avg complexity results/{Pos,Neg}_complexity_comparison.csv ) These rows use the 5-seed per-block configuration `[2,2,4,2,2,1,1,1,2]`, **not** the final model's configuration (D-05).

-->

## Attribution: gates, SHAP and permutation importance

**Gate values.** DCMFNet's gates hardly differed between predictor groups.
- **Fused stream:** the mean gate value of every group lay between 0.499 and 0.507 for both outcomes, with SDs across splits of at most 0.006. This is the midpoint of the logistic gate.
- **Independent stream:** gate values ranged from 0.44 to 0.56.
- **Stability:** the ranking of groups by gate value was not stable across splits. The mean between-split Spearman correlation was 0.17 (PLE-HM) and 0.12 (DEP) for the fused gates and 0.16 and 0.39 for the independent gates, with minimum values down to −0.55.

<!-- src: Models/Model Comparison/attribution_modality_summary.csv (DCMFNet_gate_fused importance_mean 0.4989–0.5069, importance_std ≤ 0.0061; DCMFNet_gate_independent 0.4439–0.5620); attribution_stability.csv -->

**Gates compared with SHAP.** Gate values agreed poorly with DCMFNet's own SHAP values:
- fused gates: mean Spearman ρ 0.04 for PLE-HM and 0.36 for DEP;
- independent gates: −0.02 and 0.21;
- overlap of the three highest-ranked groups: 0.27–0.53.

In contrast, DCMFNet's SHAP rankings were stable across splits (mean ρ 0.79 and 0.85), as were its permutation rankings (0.73 and 0.82). <!-- src: attribution_agreement_summary.csv (DCMFNet_gate_fused vs DCMFNet_SHAP 0.040 / 0.360; DCMFNet_gate_independent vs DCMFNet_SHAP −0.022 / 0.215; top3_overlap 0.27–0.53); attribution_stability.csv (DCMFNet_SHAP 0.793 / 0.852; DCMFNet_permutation 0.735 / 0.818) -->

**Which groups the models relied on.** SHAP values and permutation importance gave a similar picture for all three models (Fig. 4).
- **Symptoms at age 15 (SCZ15)** had the largest share of SHAP importance:
  - DCMFNet: 34% (PLE-HM) and 33% (DEP);
  - GLMM: 30% and 28%.
- **Age-18 groups:** the adverse experiences and substance use measured at age 18 came next. They accounted for 31% (PLE-HM) and 30% (DEP) of DCMFNet's SHAP importance and 34% and 29% of the GLMM's.
- **Substance use at age 15** (the anchor group of DCMFNet) accounted for 4.7–6.1% of SHAP importance in every model.
- **Permutation importance:** permuting the SCZ15 items reduced test R² by 0.24 (PLE-HM) and 0.22 (DEP) for DCMFNet, and by 0.22 and 0.20 for the GLMM. The reduction was at most 0.008 for substance use at age 15 in all models.

<!-- src: attribution_modality_summary.csv share_mean: DCMFNet_SHAP SCZ15 0.340 / 0.327, ACE18+SUD18 0.137+0.178 = 0.314 (Pos), 0.192+0.107 = 0.299 (Neg), SUD15 0.052 / 0.061; GLMM_SHAP SCZ15 0.296 / 0.278, ACE18+SUD18 0.201+0.134 = 0.335 (Pos), 0.193+0.096 = 0.289 (Neg), SUD15 0.057 / 0.047; GAMM_SHAP SUD15 0.056 / 0.047. importance_mean permutation: DCMFNet SCZ15 0.241 / 0.217, GLMM 0.225 / 0.198; SUD15 DCMFNet 0.0076 / 0.0044, GLMM 0.0059 / 0.0016, GAMM 0.0058 / 0.0016. Permuting SCZ15 reduces R² by more than the model's R², i.e. test R² becomes negative. -->

**Agreement between models.**
- **DCMFNet vs GLMM, SHAP:** mean Spearman ρ 0.64 (PLE-HM) and 0.81 (DEP); top-three overlap 0.93 and 0.80.
- **DCMFNet vs GLMM, permutation importance:** ρ 0.75 and 0.80; top-three overlap 1.00 and 0.93.
- **GLMM vs GAMM:** practically identical attributions (SHAP ρ 0.99 and 1.00).

With 11 groups these correlations are coarse. These attributions describe what the fitted models use for prediction, not causal effects of the predictors. <!-- src: attribution_agreement_summary.csv (DCMFNet_SHAP vs GLMM_SHAP 0.642 / 0.805, top3 0.93 / 0.80; DCMFNet_permutation vs GLMM_permutation 0.745 / 0.800, top3 1.00 / 0.93; GLMM_SHAP vs GAMM_SHAP 0.995 / 1.000) -->

## Calibration, subgroup and sensitivity analyses

**Table 5.** Calibration, upper-tail and subgroup results, mean (SD) over five splits.

| Model | CITL | Slope | Top-10% precision | Top-10% AUROC | R² (F) | R² (M) | R² (age 15) |
|----------------|--------------|----------|-----------|-------------|------------|------------|------------|
| *PLE-HM* | | | | | | | |
| DCMFNet | −0.007 (0.005) | 0.82 (0.02) | 0.39 (0.05) | 0.76 (0.02) | 0.212 (0.030) | 0.194 (0.013) | 0.157 (0.016) |
| GLMM | −0.002 (0.003) | 0.86 (0.03) | 0.40 (0.03) | 0.76 (0.02) | 0.236 (0.025) | 0.213 (0.007) | 0.168 (0.008) |
| GAMM | −0.002 (0.003) | 0.86 (0.03) | 0.41 (0.03) | 0.76 (0.02) | 0.236 (0.024) | 0.211 (0.006) | 0.166 (0.008) |
| Linear regression | −0.002 (0.003) | 0.89 (0.04) | 0.42 (0.04) | 0.77 (0.03) | 0.243 (0.032) | 0.232 (0.010) | 0.179 (0.009) |
| Elastic net | −0.002 (0.003) | 1.07 (0.05) | 0.42 (0.04) | 0.78 (0.03) | 0.262 (0.027) | 0.255 (0.009) | 0.198 (0.010) |
| Gradient boosting | −0.002 (0.003) | 1.07 (0.07) | 0.41 (0.04) | 0.78 (0.03) | 0.239 (0.032) | 0.248 (0.008) | 0.183 (0.014) |
| *DEP* | | | | | | | |
| DCMFNet | −0.017 (0.012) | 0.87 (0.04) | 0.35 (0.03) | 0.76 (0.01) | 0.178 (0.025) | 0.097 (0.017) | 0.126 (0.018) |
| GLMM | −0.003 (0.006) | 0.82 (0.07) | 0.33 (0.03) | 0.77 (0.01) | 0.180 (0.030) | 0.107 (0.026) | 0.143 (0.013) |
| GAMM | −0.003 (0.006) | 0.82 (0.06) | 0.33 (0.03) | 0.76 (0.01) | 0.177 (0.031) | 0.104 (0.028) | 0.141 (0.012) |
| Linear regression | −0.003 (0.007) | 0.85 (0.07) | 0.33 (0.02) | 0.77 (0.01) | 0.193 (0.032) | 0.120 (0.030) | 0.152 (0.015) |
| Elastic net | −0.003 (0.006) | 0.99 (0.08) | 0.35 (0.03) | 0.78 (0.01) | 0.205 (0.022) | 0.170 (0.026) | 0.174 (0.009) |
| Gradient boosting | −0.003 (0.006) | 1.00 (0.09) | 0.35 (0.04) | 0.78 (0.01) | 0.195 (0.020) | 0.163 (0.027) | 0.166 (0.006) |

<!-- src: Models/Model Comparison/prediction_metrics_summary.csv (full and age15 rows); prediction_metrics_by_sex_summary.csv (full rows) -->

*Footnote to Table 5:* CITL, calibration-in-the-large: the mean observed minus the mean predicted score; negative values mean that the model over-predicted on average. Slope: calibration slope. R² (F), R² (M): females and males. R² (age 15): predictors measured up to age 15 only. Top 10%: the 10% of test participants with the highest observed scores (12.1% for PLE-HM and 11.1% for DEP, because of tied scores). Precision is the fraction of the model's top 10% of predictions that fell in this group; a model without information would score about 0.12 and 0.11. Sex-specific R² uses the variance within each sex, so it excludes the variance explained by the sex difference itself. All columns except R² (age 15) use all predictors.

**Calibration.**
- **Calibration-in-the-large:**
  - the mixed models and the three comparators were well calibrated: the mean prediction was within 0.003 of the mean observed score for both outcomes (within 0.014 on every split for the mixed models);
  - DCMFNet over-predicted DEP on average (−0.017; range −0.029 to 0.000 across splits) and PLE-HM slightly (−0.007).
- **Calibration slopes:**
  - DCMFNet, the mixed models and linear regression had slopes below 1 (0.82–0.89), meaning that their predictions were spread somewhat too widely;
  - the elastic net and gradient boosting had slopes close to 1 for DEP (0.99 and 1.00) and slightly above 1 for PLE-HM (1.07 for both), meaning that their PLE-HM predictions were slightly too narrow;
  - DCMFNet's slope was lower than the elastic net's for PLE-HM (difference −0.26; Holm-adjusted p = 0.028), and this also held with age-15 predictors only (p = 0.002);
  - no other difference in slope was significant after correction (smallest Holm-adjusted p 0.085);
  - for the main models, split-specific slopes ranged from 0.74 to 0.96.
- **Deciles of predicted score**, averaged over the five splits (Fig. 3):
  - DCMFNet and the mixed models overestimated the observed mean in the highest decile, for DEP (predicted 0.448 in the GLMM and 0.455 in DCMFNet; observed 0.404 and 0.406) and for PLE-HM (0.252 and 0.261; observed 0.234 and 0.226);
  - the mixed models underestimated the lowest decile for DEP (0.108 vs 0.135), whereas DCMFNet did not (0.134 vs 0.135).
- **GLMM vs GAMM:** the two calibration curves were almost identical.

<!-- src: prediction_metrics_summary.csv (calibration_in_the_large_mean, calibration_slope_mean, full); prediction_metrics_per_seed.csv (ranges: mixed models −0.013 to 0.006, DCMFNet DEP −0.029 to −0.000; main-model slopes 0.742–0.957, full); paired_comparisons.csv metric calibration_slope (family main: p_holm 1.0 for all model pairs; family baselines: DCMFNet − ElasticNet Pos full −0.256, p_holm 0.028; age15 −0.254, p_holm 0.0015; next smallest 0.085); calibration_bins.csv (decile means averaged over seeds, full predictor set) -->

**Upper tail.**
- About two in five of each model's top 10% of predictions for PLE-HM (0.39–0.42), and one in three for DEP (0.33–0.35), belonged to the 10% with the highest observed scores (Table 5). The chance level is about 0.12 and 0.11.
- The AUROC for identifying this group was 0.76–0.78 for all models; the elastic net and gradient boosting had the highest values (0.78).
- All models underestimated the scores of this group:
  - mean bias −0.12 to −0.14 for PLE-HM;
  - mean bias −0.26 to −0.28 for DEP.

  The underestimation was smallest for DCMFNet (−0.124 and −0.257) and largest for the elastic net and gradient boosting (−0.136 to −0.139 and −0.275 to −0.278).

<!-- src: prediction_metrics_summary.csv (top_precision_at_10pct_mean, top_auroc_mean, top_mean_bias_mean, top_observed_fraction_mean; full) -->

**Sex.** Most models explained more within-sex variance in females than in males, most clearly for DEP (Table 5).
- **DEP in males:** R² was 0.10–0.12 for DCMFNet, the mixed models and linear regression, and 0.16–0.17 for the elastic net and gradient boosting.
- **PLE-HM:** the two sexes were similar for the elastic net (0.262 and 0.255), and gradient boosting did slightly better in males (0.248 vs 0.239).

These differences were not formally tested. <!-- src: prediction_metrics_by_sex_summary.csv (full) -->

**Predictors measured by age 15.** Without the age-18 groups (ACE18, SUD18), test R² fell by 0.05–0.07 in every model.
- **Size of the drop:**
  - DCMFNet: 0.049 (PLE-HM) and 0.072 (DEP);
  - GLMM: 0.060 and 0.060;
  - GAMM: 0.061 and 0.059.
- **Significance of the drop (Holm-adjusted p):**
  - DCMFNet: 0.002 and 0.006;
  - GLMM: 0.041 (PLE-HM) and 0.066 (DEP);
  - GAMM: 0.036 and 0.075.
- **Comparators:** the drop was 0.060–0.063 (Holm-adjusted p 0.010–0.094 in the comparator family).
- **Between models:** with age-15 predictors only, the ranking was unchanged:
  - the mixed models remained slightly ahead of DCMFNet, by 0.010–0.016 (all Holm-adjusted p = 1.00);
  - every comparator was ahead of DCMFNet on all five splits, by 0.023–0.048 (Holm-adjusted p ≥ 0.11);
  - the elastic net again did best (0.198 for PLE-HM, 0.174 for DEP).
- **DEP in males:** with age-15 predictors only, DCMFNet explained almost none of the within-sex variance (R² 0.010, SD 0.018). The other models reached 0.058–0.061 (GLMM, GAMM), 0.065 (linear regression), 0.096 (gradient boosting) and 0.105 (elastic net).

<!-- src: paired_comparisons.csv ("(full)" vs "(age15)" rows for each model, metric r2; GLMM/GAMM vs DCMFNet "(age15)" rows); prediction_metrics_summary.csv (age15 rows); prediction_metrics_by_sex_summary.csv (age15 rows). For DCMFNet, ACE18 and SUD18 were set to their training means rather than removed (Methods). -->
