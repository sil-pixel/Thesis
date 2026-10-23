<!--
STATUS 2026-10-23: PARTIAL DRAFT, BMC Psychiatry structure (unnumbered headings; Tables 1–2 are in Methods, so Results tables start at 3). §3.2–3.4 now use the corrected GLMM/GAMM (E-01, E-02 run by the author on the shared
splits, with DCMFNet-aligned interactions and R² = 1 − SSE/SST). Still pending, from the pre-expiry runs:
Table 1 and flow (A-02, W-02), attribution (E-06), calibration / sex / top-decile metrics and the age-15 sensitivity
analysis (E-09, A-04, E-10, E-05). The DCMFNet configuration-selection issue (E-03) is not resolved and must be
disclosed with these results. Synthetic-data results (README; artifacts/) are excluded by design (D-17).
Outcome labels PLE-HM and DEP are working labels pending M-01.
-->

# Results

## Analysis sample
The analysis sample comprised 4,033 individuals from 2,690 twin pairs with complete data on all 105 predictors and both outcomes. Each of the five test sets held 25% of the twin pairs, between 998 and 1,028 individuals. <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1-3, 361, 685, 1063, 1414 -->
`[TODO: Table 1, analysis sample vs excluded (A-02), and Fig. 1, participant flow (W-02) — from analysis_sample_descriptives.R]`

## Predictive performance

All models were evaluated on the same five twin-pair-grouped train/test splits. `[VERIFY: split_summary.csv reports matches_dcmfnet_log = True for all seeds]` R² is 1 − SSE/SST throughout, and SDs are across the five splits (population SD). <!-- src: Models/GLMM/Method/GLMM_updated_split.R (header items 1, 4, 5); Models/GAMM/Method/GAMM_updated_split.R -->

**Table 3.** Test-set performance, mean ± SD over five splits.

| Model | Outcome | R² | RMSE | MAE | Spearman ρ |
|---|---|---|---|---|---|
| DCMFNet | PLE-HM | 0.2058 ± 0.0189 | 0.0903 ± 0.0013 | 0.0712 ± 0.0004 | 0.4497 ± 0.0193 |
| GLMM | PLE-HM | 0.2279 ± 0.0156 | 0.0890 ± 0.0014 | 0.0700 ± 0.0012 | 0.4641 ± 0.0183 |
| GAMM | PLE-HM | 0.2274 ± 0.0153 | 0.0891 ± 0.0014 | 0.0699 ± 0.0011 | 0.4633 ± 0.0180 |
| DCMFNet | DEP | 0.1984 ± 0.0208 | 0.1562 ± 0.0017 | 0.1242 ± 0.0019 | 0.4509 ± 0.0135 |
| GLMM | DEP | 0.2025 ± 0.0220 | 0.1559 ± 0.0024 | 0.1226 ± 0.0016 | 0.4460 ± 0.0154 |
| GAMM | DEP | 0.1998 ± 0.0230 | 0.1561 ± 0.0026 | 0.1228 ± 0.0017 | 0.4421 ± 0.0163 |

<!-- src: DCMFNet: Models/DCMFNet/Results/Final Best Model/test_results_summary.csv; GLMM: Models/GLMM/Results/GLMM_updated_summary.csv; GAMM: Models/GAMM/Results/GAMM_updated_summary.csv -->
<!-- note: DCMFNet configuration was selected partly on test data (D-04, D-09, D-21; TODO E-03), so its row may be optimistic. GLMM/GAMM were not tuned. -->

Mean R² was higher for the GLMM and GAMM than for DCMFNet on both outcomes. The difference was 0.022 for PLE-HM and 0.004 for DEP (GLMM minus DCMFNet). GLMM and GAMM performed almost identically. For DEP, DCMFNet had the highest Spearman correlation (0.451 vs 0.446 and 0.442). <!-- src: same files -->

Penalised linear models fitted with default settings on the same splits did better than all three models. The cross-validated lasso had mean R² 0.261 (SD 0.019) for PLE-HM and 0.234 (0.017) for DEP (Additional file 1: Table S2). <!-- src: Models/ML/Results/Pos_benchmark_summary.csv (LassoCV r2_mean 0.2614, r2_std 0.0192); Neg_benchmark_summary.csv (LassoCV 0.2341, 0.0166; best DEP model LarsCV 0.2346). These models ignore twin pairs and were scored by lazy_predict.py, not by prediction_evaluation.py. -->

`[TODO: add the default-setting ML baselines (same splits as DCMFNet) as a supplementary table; the best linear baselines had mean R² 0.261 (LassoCV, PLE-HM) and 0.235 (LarsCV, DEP) — Models/ML/Results/{Pos,Neg}_benchmark_summary.csv. Decide with the author whether they appear in Table 3.]`

## Paired comparisons across splits

The models were compared split by split with the corrected resampled t-test (Nadeau and Bengio), using the per-split metrics, with Holm correction across the six comparisons per metric (three model pairs × two outcomes) [@R-NADEAU]. <!-- src: computed 2026-10-23 from Models/DCMFNet/Results/Final Best Model/{Pos,Neg}_test_results_per_seed.csv, Models/GLMM/Results/GLMM_updated_{Pos,Neg}_per_seed.csv, Models/GAMM/Results/GAMM_updated_{Pos,Neg}_per_seed.csv; test/train ratio 1/3 -->

**Table 4.** Paired differences in test R² (first model minus second), five splits.

| Outcome | Comparison | Mean ΔR² (SD) | Splits where first model was better | p | p (Holm) |
|---|---|---|---|---|---|
| PLE-HM | DCMFNet − GLMM | −0.0221 (0.0133) | 1 of 5 | 0.086 | 0.50 |
| PLE-HM | DCMFNet − GAMM | −0.0216 (0.0129) | 1 of 5 | 0.083 | 0.50 |
| PLE-HM | GLMM − GAMM | 0.0005 (0.0007) | 4 of 5 | 0.40 | 1.00 |
| DEP | DCMFNet − GLMM | −0.0041 (0.0253) | 2 of 5 | 0.84 | 1.00 |
| DEP | DCMFNet − GAMM | −0.0013 (0.0270) | 2 of 5 | 0.95 | 1.00 |
| DEP | GLMM − GAMM | 0.0027 (0.0025) | 4 of 5 | 0.22 | 0.87 |

None of the differences was statistically significant. RMSE gave the same pattern: PLE-HM DCMFNet − GLMM +0.0013, p = 0.085; DEP +0.0004, p = 0.84. With five splits these tests have little power, so the results show no detectable advantage for DCMFNet. They do not show that the models are equivalent. <!-- src: same computation -->
`[TODO: replace with prediction_evaluation.py paired_comparisons.csv once run (retrained DCMFNet, identical metric code; adds MAE, Spearman, calibration slope)]`

## Interaction and smooth terms in the mixed models

Coefficients come from the fits on the first split (seed 42). The GLMM and GAMM each included 24 SUD15 interaction terms:
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
`[TODO: full coefficient table as Additional file (all terms, not only p < 0.05)]`

<!-- MOVED to Additional file (not a main result for BMC Psychiatry):
## 3.5. Depth configuration (as conducted; 5 seeds)
With depth set per block rather than uniformly, test RMSE was 0.0898 ± 0.0015 vs 0.0905 ± 0.0010 for PLE-HM (uniform L = 3) and 0.1569 ± 0.0014 vs 0.1569 ± 0.0028 for DEP (uniform L = 2). ( src: Models/DCMFNet/Results/Complexity Analysis/Avg complexity results/{Pos,Neg}_complexity_comparison.csv ) These rows use the 5-seed per-block configuration `[2,2,4,2,2,1,1,1,2]`, **not** the final model's configuration (D-05).

-->

## Attribution: gates, SHAP and permutation importance
`[TODO: E-06 — from shap_analysis.py, GLMM_shap_updated.R and attribution_agreement.py]`

## Calibration, subgroup and sensitivity analyses
`[TODO: E-09 calibration, A-04 sex-stratified, E-10 top-decile, E-05 age-15-only predictors — from prediction_evaluation.py]`
