# Discrepancy Log

All entries were identified on **2026-10-14** during the initial implementation and provenance audit.
Severity: **Critical** (threatens validity of a reported result or central claim) · **High** · **Medium** · **Low**.
Line numbers refer to the files as of commit `7ede66b` (HEAD at audit time).

---

## D-01 · GLMM/GAMM interaction terms are an arbitrary subset (R vector recycling) · **Critical**
- **Topic / section:** Statistical baselines; Methods §2.5, Results (interaction findings)
- **Conflict:**
  - Thesis §4.2.1: "interaction terms between SUD15 and **all** other covariates were included."
  - Code builds terms with `paste0(SUD15_vars, ":", SCZ15_vars)` etc. R's `paste0` pairs vectors element-wise **with recycling**. It does not form a Cartesian product. With 6 substance variables and 23/19/17/11/4/16/1 covariates, only one recycled term is created per covariate.
  - The saved coefficients confirm **97** interaction terms (`Models/GLMM/Results/Pos_GLMM_all_coefs.csv`) versus 6 × 91 = 546 intended. Example: cannabis × ACE18 and other-drugs × ACE18 are absent, while painkiller-opioids × ACE18 is present.
- **Sources:** `Models/GLMM/Method/GLMM.R:42-54` <!-- src: Models/GLMM/Method/GLMM.R:42-54 -->; `Models/GAMM/Method/GAMM.R` (same construction, `[VERIFY: line numbers]`); `Models/GLMM/Results/Pos_GLMM_all_coefs.csv`
- **Consequence:** The "interaction effects" in thesis Figs 5.1, 5.4 and 5.5 (e.g., opioid × ACE18 reversal, "other drugs × PRS consistently positive") depend on column ordering, not on a scientific specification. The GLMM/GAMM performance numbers are those of an arbitrarily specified model. Any claim that DCMFNet's importance "is consistent with" GLMM interactions is unsupported.
- **Draft follows:** No GLMM/GAMM interaction findings are reported. Predictive metrics are reported only with an explicit caveat, pending re-estimation.
- **Author decision required:** Approve re-specification (e.g., `outer()`/`expand.grid` to form all pairs, or a principled pre-specified subset, with penalization) and re-running GLMM/GAMM. → TODO E-01
- **Status:** Open

## D-02 · Statistical baselines and DCMFNet evaluated on different test sets · **Critical**
- **Topic / section:** Experimental protocol; Methods §2.7, Results Table 2
- **Conflict:**
  - Thesis §4.5: "All models were trained using **the same train–test split** to ensure a fair comparison."
  - DCMFNet/ML: `sklearn.model_selection.train_test_split(unique cmpair, test_size=0.25, random_state=seed)` <!-- src: Models/DCMFNet/Method/legacy/train.py:61-68; Models/ML/Method/lazy_predict.py:29-33 -->
  - GLMM/GAMM: `set.seed(seed); sample(unique_ids, size=n_test)` in R <!-- src: Models/GLMM/Method/GLMM.R:203-213 -->. The same seed integers produce **different** partitions across the two languages.
- **Consequence:** DCMFNet vs. GLMM/GAMM differences mix model effects with split effects. Paired comparison is impossible. The "consistent improvement over GLMM/GAMM" claim is not established. (DCMFNet vs. the LazyPredict baselines *is* paired, because they share identical splits.)
- **Draft follows:** States the actual protocol and restricts paired claims to DCMFNet vs. ML baselines.
- **Author decision required:** Export the Python partitions (cmpair IDs per seed) and re-run R models on them. → TODO E-02
- **Status:** Open

## D-03 · "Negative symptoms" outcome is a depressive-symptom score; "positive" mixes psychotic and manic items · **Critical (construct validity)**
- **Topic / section:** Outcome definition; Title, Abstract, Methods §2.2, Discussion
- **Conflict:**
  - Thesis title, abstract and README: "schizophrenia symptom severity", "positive and negative psychotic symptoms".
  - Thesis §3.1 (one sentence): "negative symptoms are **proxied by depressive symptoms**".
  - Code: `SCZ18_Neg` sums 11 items: poor_appetite, depressed, felt_effort, restless, unhappy, lonely, others_unfriendly, not_enjoyed_life, sad, people_dislike_me, could_not_get_going. These look like a depression scale (possibly SMFQ-derived). `SCZ18_Pos` sums 24 items (7 psychotic-like experiences + 17 hypomanic/manic items). <!-- src: Data Preprocessing/Data Visualisation/rename_catss_with_mapping.py:73-81 -->
  - Normalization constants 46 (Pos) and 33 (Neg) imply item ranges that are not documented (24 items → max 46?). <!-- src: rename_catss_with_mapping.py:91-92 -->
- **Consequence:** Reviewers in psychiatry will reject "negative symptoms" for a depression score. Labeling a manic+psychotic composite as "schizophrenia positive symptoms" is also contestable. All clinical interpretation depends on this.
- **Draft follows:** Uses neutral labels: "psychotic-like and hypomanic experiences (PLE-HM) score" and "depressive symptom score". The outcomes are not called schizophrenia symptoms. `[AUTHOR INPUT]` on final naming.
- **Author input:** Instrument names (e.g., CATSS age-18 self-report; SMFQ? PLIKSi? MDQ?), item response scales, rationale for 46/33. → TODO M-01
- **Status:** Open

## D-04 · Test-set-informed model configuration selection (IGF depth) · **Critical**
- **Topic / section:** Model selection; Methods §2.6
- **Conflict:**
  - `complexity_analysis.py` Phase 1 selects depths on validation RMSE (acceptable). Phase 2 compares "optimal per-modality" vs "uniform" on the **test sets** <!-- src: Models/DCMFNet/Analysis/complexity_analysis.py:204-250 -->.
  - Thesis §5.6.1: "Based on these results, the best-performing configuration was selected for subsequent analysis."
  - Git history: `5c35f0e "best models from both complexity and tuning"`, `6173d10 "importance analysis with Neg back to default layers tuned"`. The Pos final config `[5,2,1,5,4,1,1,1,5]` equals the seed-42 Phase-1 optimum. Its Phase-2 *test* R² was 0.262 vs 0.213 for uniform (`Results/Complexity Analysis/Pos_complexity_comparison.csv`). Neg reverted to uniform L=2 after its per-modality config was worse on test.
  - Thesis §5.6.1 also states that joint depth tuning "consistently led to overfitting **on the test set**", which implies test performance was inspected during development.
- **Consequence:** The final reported test metrics (Table 5.2) are optimistically biased by selection on test data. The magnitude is probably small, but this is a TRIPOD+AI and reviewer red flag.
- **Draft follows:** Discloses the selection procedure. `[VERIFY: RESULT PROVENANCE]`
- **Author decision required:** Re-run with selection confined to validation data (nested), or report both configurations as pre-specified alternatives. → TODO E-03
- **Status:** Open

## D-05 · Three conflicting specifications of the Positive-model IGF depth · **High**
- **Conflict:**
  - Thesis Table B.1: "Number of layers: 3" (Pos), 2 (Neg).
  - `Models/DCMFNet/hyperparameters.json` Pos: `[5, 2, 1, 5, 4, 1, 1, 1, 5]` (per-modality; seed-42 single-seed sweep). This is what generated `Results/Final Best Model/*` = thesis Table 5.2 (R² 0.206).
  - `Results/Complexity Analysis/Avg complexity results/hyperparameters.json` Pos: `[2, 2, 4, 2, 2, 1, 1, 1, 2]` (5-seed sweep). This is what generated thesis Table 5.3 "Positive Optimal" (R² 0.2145).
  - Thesis text §5.6.1 interprets the 5-seed sweep ("ADHD9 more complex", depth 4). The reported final model, however, uses ADHD9 depth = 1.
- **Consequence:** The interpretation of "modality-specific interaction complexity" does not correspond to the model whose performance and importance are reported. Tables 5.2 and 5.3 report different Pos models without saying so.
- **Draft follows:** Reports the configuration that produced each number. `[AUTHOR INPUT]` on which is primary.
- **Status:** Open

## D-06 · "Squeeze-and-Excitation" description does not match the implementation · **High**
- **Conflict:**
  - Thesis §4.3.2: a squeeze step $z_c = \frac{1}{b}\sum_{i=1}^{b} X_{i,c}$ (batch mean), then excitation. The thesis claims that "unlike soft attention, which assigns weights at the instance level, SE attention captures global feature importance."
  - Code (thesis-era commit `6173d10` and current `dcmfnet/model.py:76-77`): `return inputs * self.excitation(inputs)`. There is **no squeeze/pooling**. Gates are computed per instance from that instance's own features, so this is an instance-level, input-conditioned sigmoid gating (similar to a gated linear unit / feature-wise gating), not SE channel attention. <!-- src: Models/DCMFNet/Method/dcmfnet/model.py:50-77 -->
- **Consequence:** The Methods equation and the stated motivation are wrong. The "importance" analysis interprets per-instance gates averaged over test samples (see D-12).
- **Draft follows:** Describes the implemented operation as input-conditioned feature gating "inspired by SE". Equation corrected.
- **Status:** Open (writing fix; no code change needed)

## D-07 · Loss function description differs from implementation · **Medium**
- **Conflict:**
  - Thesis: $w_i = 1/\mathrm{freq}(y_i)$; $L_{focal} = w_i |y_i-\hat y_i|^\gamma L_\delta$.
  - Code: bin weights are $1/(\text{count}+1)$, normalized to mean 1 and **clipped at 20**, over equal-width bins on [0,1]. The focal term is computed **without gradient** and normalized by its batch mean. The product of weights is renormalized to batch mean 1. <!-- src: Models/DCMFNet/Method/dcmfnet/losses.py:15-76 -->
  - Empty high bins receive the largest weights, so bins with zero training samples dominate the normalization (training log shows weights 0.003 for the most common bin vs 0.34 for [0.54,0.62)). <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:12-27 -->
- **Draft follows:** Implementation.
- **Status:** Open (writing fix)

## D-08 · Train/test split described as outcome-stratified; it is not · **Medium**
- Thesis §4.1: "75–25 split, while preserving the distribution of the outcome variables."
- Code: `train_test_split` on unique `cmpair` IDs with no `stratify` argument. <!-- src: Models/DCMFNet/Method/legacy/train.py:61-63 -->
- Twin pairs are kept together (correctly).
- **Draft follows:** Code. **Status:** Open

## D-09 · Hyperparameter tuning data overlap with later test sets · **High**
- Optuna tuning used the seed-42 training portion with an inner 80/20 split. <!-- src: Models/DCMFNet/Method/legacy/tuning.py:86-92, 216-217 --> The final evaluation re-draws test sets for seeds 43–46. Those test sets overlap the data used to choose hyperparameters. The same applies to the depth sweep (each seed's training data informs a configuration that is then tested on other seeds' test sets).
- In addition, the feature-screening LightGBM/SHAP step used its own 70/30 split on the full cohort <!-- src: Data Preprocessing/Data Analysis/Feature Importance/feature_importance.py:31-45 -->, so outcome-informed feature screening saw data later used as test data.
- The ML baselines were *not* tuned, so this leakage favours DCMFNet.
- **Consequence:** DCMFNet's test estimates for seeds 43–46 are optimistically biased relative to untuned baselines.
- **Status:** Open → TODO E-03, E-04

## D-10 · Descriptive statistics describe a different sample than the analysis sample · **High**
- Thesis Table 3.1 is computed on ~9,261 individuals (e.g., Male 3,221 + Female 4,683 with 14.7% missing). The models use the complete-case sample **n = 4,033** (`Data shape after dropping na: (4033, 125)`). <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1 -->
- Flow numbers in thesis Figs 3.1 and 3.3 also differ (13,146 → 9,271 → 9,160 genotyped; 13,146 → 9,261 valid exposure/outcome; final 4,033).
- **Consequence:** TRIPOD+AI requires characteristics of the analysis sample and a comparison with excluded participants (selection bias under complete-case analysis; 56% excluded).
- **Status:** Open → TODO A-02

## D-11 · Claims of DCMFNet superiority contradicted by saved baseline results · **Critical (central claim)**
- Thesis abstract and conclusions: "DCMFNet provided modest but consistent improvements"; Discussion: "DCMFNet offers improved flexibility and predictive capacity".
- Saved results on **identical splits** (5 seeds, mean):

  | Model | Pos R² | Pos RMSE | Pos ρ | Neg R² | Neg RMSE | Neg ρ |
  |---|---|---|---|---|---|---|
  | DCMFNet (final) | 0.206 | 0.0903 | 0.450 | 0.198 | 0.1562 | 0.451 |
  | LinearRegression (OLS, untuned, no interactions) | 0.239 | 0.0884 | 0.473 | 0.215 | 0.1546 | 0.453 |
  | LassoCV | 0.261 | 0.0871 | 0.492 | 0.234 | 0.1527 | 0.472 |

  <!-- src: Models/DCMFNet/Results/Final Best Model/test_results_summary.csv; Models/ML/Results/Pos_benchmark_summary.csv; Models/ML/Results/Neg_benchmark_summary.csv -->
- On these splits, DCMFNet's mean R² is exceeded by **20 of 40** LazyPredict baselines for Pos and **17 of 40** for Neg (all run at default settings; some, e.g., LassoCV, select their penalty by internal CV). These include OLS, ridge/lasso variants, random forest (Pos 0.221) and gradient boosting (Pos 0.229; Neg 0.214). DCMFNet does exceed LightGBM and HistGradientBoosting at default settings (Pos 0.196/0.199). The default sklearn MLPRegressor fails (R² ≈ −0.36), which is a relevant comparison point for a neural model. The only models it beats are GLMM/GAMM, which were evaluated on different splits (D-02) with a mis-specified interaction set (D-01).
- An additive linear model without interactions outperforming both the interaction GLMM and DCMFNet is itself evidence *against* the thesis's interpretation that risk "is organized through interactions" (at least as far as prediction is concerned).
- **Consequence:** The central methodological claim cannot be made. This drives the framing decision.
- **Status:** Open → author decision (framing)

## D-12 · "Importance" = mean sigmoid gate value; interpretation overstated · **High**
- Implementation: per-modality mean of the per-instance gate values from `attn_fused`, `attn_independent`, `attn_final`, averaged over test samples and 5 seeds. <!-- src: Models/DCMFNet/Analysis/importance_analysis_avg.py:43-114 -->
- Problems: (i) Gate values multiply **unscaled** raw inputs (DCMFNet receives no feature standardization; inputs range from binary SEX to ordinal 0–7 to PRS ~−2.6–0.7), so a gate value is not comparable across modalities. (ii) A gate does not measure contribution to the output, which also depends on input magnitude and downstream weights. (iii) Stream comparison averages gates over very different numbers of dimensions. (iv) Values cluster near 0.5 (the sigmoid midpoint). (v) Attention/gating weights are not explanations (Jain & Wallace 2019).
- Thesis claims "consistent with GLMM" and that "SUD15 is more important in the interaction setting", which rely on D-01 and on (i)–(v).
- **Status:** Open → TODO E-06 (validated attribution, e.g., permutation importance / SHAP on DCMFNet and on baselines; ablation of fused stream)

## D-13 · DCMFNet ignores twin-pair dependence during training; inputs unscaled · **Medium**
- Mixed models include a random intercept for `cmpair`. DCMFNet and the ML baselines do not model pair dependence (only grouped splitting). Acknowledged in thesis §6.6.
- DCMFNet receives raw features. LazyPredict applies standard scaling internally `[VERIFY: lazypredict 0.3.0 preprocessing]`. The models therefore do not receive identical preprocessing.
- **Status:** Open (report as limitation; TODO A-05 to verify LazyPredict preprocessing)

## D-14 · Thesis frames data as both "cross-sectional" and longitudinal; temporal ordering of predictors · **High**
- Thesis §3.1 says CATSS provides "cross-sectional questionnaire-based data at pre-defined ages 9, 15, and 18"; the conclusions say "cross-sectional, complete-case sample".
- Predictors include **age-18** measures (ACE18, SUD18) collected concurrently with the age-18 outcome. Therefore this is not a prospective prediction of age-18 outcomes from earlier exposures. ACE18 and SUD18 could be consequences or correlates of the outcome.
- **Consequence:** The prediction target and horizon must be stated precisely. A sensitivity analysis excluding age-18 predictors is needed for any "risk prediction" framing.
- **Status:** Open → TODO E-05

## D-15 · Missing-value coding inconsistencies · **Medium**
- `other_bullying15` and `other_abuse18` replace NaN with **−1** and cast to int <!-- src: Data Preprocessing/Feature Selection/feature_selection.py:116-130 -->. These values therefore survive `dropna()` as an ordinal value of −1 rather than being treated as missing. Elsewhere "don't know/don't want to answer" is set to −1 and then to missing (thesis §3.6; `change_missingness`).
- Table 3.1 reports ACE15 range 1–6 and ACE18 as binary, which is inconsistent with −1 values if present. `[VERIFY]`
- **Status:** Open → TODO A-03

## D-16 · Final dataset construction not fully reproducible from the repository · **Medium**
- `catss_final_data.csv` is read by all models but the script that produces it from `catss_modalities.csv`/`catss_aggregated_output.csv` plus PRS/PC merge is not in the repository. `rename_catss_with_mapping.py` contains steps reading and writing `catss_final_data.csv` in an unclear order. The cell-type PRS pipeline is external (Wang 2025, master's thesis).
- **Status:** Open → TODO R-01

## D-17 · README "held-out test metrics" are from synthetic data · **Critical if misused; currently Low**
- README reports Pos R² 0.793, Neg R² 0.374. These come from `artifacts/dcmfnet_*.pt`, trained on `synthetic_data/data/synthetic_dcmfnet.csv` (commits `f0926f6`–`7ede66b`, Aug 2026), **not** on CATSS.
- **Draft follows:** Never cited. Recommend labelling the README section explicitly as synthetic. → TODO W-08
- **Status:** Open

## D-18 · Thesis Table 5.2 reports DCMFNet Pos MAE SD as 0.000 · **Low**
- Source value is 0.000418. <!-- src: Models/DCMFNet/Results/Final Best Model/test_results_summary.csv -->. It is a rounding artifact. Report as 0.0004.
- **Status:** Open (writing)

## D-19 · Thesis Table 5.3 labels Neg "Default (L=2)"; the default depth is 5 · **Low**
- `hyperparameters.json` default `num_layers: 5`. L=2 is the Optuna-tuned uniform depth.
- **Status:** Open (writing)

## D-20 · Reference metadata errors in thesis bibliography · **Low**
- [3] Sullivan et al. 2003: author order is rendered "Neale MC Sullivan PF Kendler KS". The correct order is Sullivan PF, Kendler KS, Neale MC; Arch Gen Psychiatry 60:1187–92.
- [16] DCMFNet: the venue is *Graphics Interface 2024* (ACM DL), pp. 1–12; the thesis says "Association for Computing Machinery. GI '24".
- [19] LightGBM is cited to Zhou's *Ensemble Methods* book. The primary source is Ke et al., NeurIPS 2017.
- [20] SE Networks is cited as arXiv. The peer-reviewed version is CVPR 2018, doi:10.1109/CVPR.2018.00745.
- **Status:** Open (handled in reference registry)

## D-21 · Saved tuning artifacts do not correspond to the hyperparameters used; multiple tuning campaigns · **High**
- `Models/DCMFNet/Results/Tuning/*_best_params.txt` and `*_optuna_results.csv` (current) come from a **per-modality-depth** Optuna run (29 Apr 2026). Their best parameters (e.g., Pos lr 5.29e-4, 12 epochs) differ from `hyperparameters.json` (Pos lr 1.00e-4, 23 epochs).
- The hyperparameters actually used were produced by an earlier **uniform-depth** run (search space: depth 1–7, lr 1e-5–1e-2 log, batch {4,6,8,16,32}, epochs 10–30, weight decay 1e-5–1e-2 log, dropout 0.1–0.5, SE reduction 2–8, hidden_dim_min 2–16, loss {mse, huber}, δ 0.01–0.2, γ 0.5–3.0, bins 5–20, scheduler patience 2–5, factor 0.3–0.7, early-stopping patience 3–8; 50 trials; MedianPruner(5,5); seed-42 inner split). That run is preserved only in git commit `c49eec1` (25 Apr 2026: `DCMFNet/tuning.py`, `DCMFNet/Results/Tuning/*`). Its best uniform depth was Pos 3 / Neg 2, matching thesis Table B.1.
- Git history shows ≥ 5 tuning campaigns between 25 and 29 Apr (`c49eec1`, `9079d3e`, `cfd5901`, `a68071b`, latest). Commit messages ("best models with default layers again", "best models from both complexity and tuning") indicate that the choice *among campaigns* was made after evaluating "best models" on the test sets.
- **Consequence:** Reproducibility gap (the repo's tuning outputs are not the ones used) and an additional, undisclosed selection step (D-04).
- **Draft follows:** Reports the `c49eec1` search space as the provenance of the used hyperparameters. `[VERIFY: RESULT PROVENANCE]`
- **Status:** Open → TODO R-02

## Addendum 2026-10-22 (thesis re-read after the author's decision not to re-run models)
- **D-03:** The thesis does disclose that the "negative symptoms" outcome is proxied by depressive symptoms and that "positive" includes manic features (thesis §3.1, PDF p.15). The labelling issue is therefore one of naming, not concealment. The manuscript should use descriptive outcome names.
- **D-11:** The thesis acknowledges that several standard ML models achieve stronger predictive performance than DCMFNet (thesis §5.5 and Discussion, PDF p.39–40). Its Discussion explains the gap as sample size relative to model complexity (§6.4, PDF p.49).
- **D-04 (partial):** The thesis mentions that jointly optimised per-block depths "consistently led to overfitting on the test set" (PDF p.41). That confirms test-set information entered configuration choice.
- **Not resolved by the thesis text:** D-01 (thesis §4.2.1 states interactions between SUD15 and *all* other covariates; the code fits 97 of 546), D-02 (thesis §4.5 states the same train–test split for all models; the R scripts draw different partitions) and D-08 (thesis §4.1 states outcome-preserving splits; the code does not stratify). These are differences between the stated and executed analysis. Explanation cannot reconcile them; either a refit or disclosure is needed.
- **D-02 demonstration (2026-10-22, placeholder IDs 1–2,690 only, no study data):**
  - Python `train_test_split(random_state=s)` draws 673 test pairs; R `set.seed(s); sample(...)` draws 672.
  - For seeds 42–46 the two test sets share only 23–26% of pairs, which is what chance alone would give.
  - Cause: different random-number algorithms, different sampling procedures and a different rounding of the test size (ceil vs floor). Twin pairs may also appear in a different order in the R and Python data frames.
  - The demo used a scratchpad script only; no repository files were changed.

## D-22 (found 2026-10-22): R² is defined differently in R and Python
- **R scripts:** `GLMM.R` and `GAMM.R` (`plot_predictions`) compute `r2 <- cor(actual, predicted)^2`, the squared Pearson correlation.
- **Python:** DCMFNet (`legacy/train.py:174`) and the ML benchmark (`lazy_predict.py:60`) use `sklearn.metrics.r2_score`, i.e. 1 − SSE/SST.
- **Consequence:** the squared correlation ignores bias and miscalibration, and is never below the 1 − SSE/SST value for the same predictions. The thesis R² values for GLMM/GAMM are therefore not on the same scale as DCMFNet/ML. Severity: High (comparison table).
- **Resolution:** the new `*_updated_split.R` scripts report `r2` as 1 − SSE/SST and keep `r2_cor2` for continuity.

## Addendum 2026-10-23: corrected GLMM/GAMM results (E-01, E-02)
- The author re-ran the GLMM and GAMM with `GLMM_updated_split.R` and `GAMM_updated_split.R`. These use the shared DCMFNet splits, interactions aligned with the DCMFNet modalities, R² = 1 − SSE/SST and population SD.
  <!-- src: Models/GLMM/Results/GLMM_updated_summary.csv; Models/GAMM/Results/GAMM_updated_summary.csv -->
- **The comparison reverses.** The thesis reported GLMM R² of 0.181 (PLE-HM) and 0.179 (DEP), below DCMFNet. The corrected GLMM gives 0.228 and 0.202, above DCMFNet's 0.206 and 0.198. The paired differences are not significant (04_results.md §3.3).
- **Not only a change of R² definition.** On the old definition (squared correlation), the corrected GLMM gives 0.236 and 0.216, still well above the thesis values. The gain therefore comes from the corrected specification and the shared splits.
- **Consequence for the thesis claims.** The thesis claims of DCMFNet superiority over the statistical models (D-11) and of SUD15 interaction effects (D-01) are not supported by the corrected analyses. Only 3 of 24 (PLE-HM) and 1 of 24 (DEP) interactions reach unadjusted p < 0.05, and none survives multiplicity correction.
- **Status:** D-01 and D-02 resolved by re-analysis. The old LMM/GAMM rows are removed from the manuscript tables.

## D-23 (found 2026-10-24): retrained DCMFNet does not match the configuration reported in the manuscript · **Resolved 2026-10-24**
- `shap_analysis.py` and `dcmfnet_test_predictions.py` retrain DCMFNet from the `hyperparameters.json` in the server working folder.
- **PLE-HM:** the retrained per-seed test R² (0.2594, 0.2167, 0.1942, 0.1777, 0.2252; mean 0.215) equals `Complexity Analysis/Avg complexity results/Best model across seeds/Pos_test_results_per_seed.csv` to 7 digits. That run used per-block depths [2,2,4,2,2,1,1,1,2] (`Avg complexity results/hyperparameters.json`; complexity log line 13808, mean R² 0.2145). The manuscript reports depths [5,2,1,5,4,1,1,1,5] and R² 0.206 (`Final Best Model`; `Models/DCMFNet/hyperparameters.json`). <!-- src: Models/DCMFNet/Results/Retrained Results/DCMFNet_retrained_test_metrics.csv; Models/Model Comparison/prediction_metrics_summary.csv -->
- **DEP:** the retrained per-seed R² (0.2002, 0.2231, 0.2005, 0.1989, 0.1724; mean 0.199) matches neither `Final Best Model` (mean 0.198) nor the averaged-complexity run (mean 0.197).
- **Affected outputs:** all DCMFNet rows in `prediction_metrics_*`, `paired_comparisons.csv`, `calibration_bins.csv`, `prediction_metrics_by_sex_*`, the age-15 DCMFNet results, `DCMFNet_attribution_per_seed.csv` and `attribution_agreement_*`. The GLMM/GAMM outputs are not affected.
- **Draft follows:** nothing yet. **Status:** Open. Author to confirm which `hyperparameters.json` was on the server; if it was the averaged-complexity file, rerun the DCMFNet scripts with `Models/DCMFNet/hyperparameters.json` before access ends (2026-10-28).
- **Update 2026-10-24 (after the rerun with `Models/DCMFNet/hyperparameters.json`):**
  - **PLE-HM:** reproduces `Final Best Model` exactly (0.2292, 0.1980, 0.2264, 0.1808, 0.1947).
  - **DEP:** gives 0.2141, 0.2130, 0.2048, 0.1501, 0.1680 (mean 0.190 ± 0.026), against 0.2023, 0.2197, 0.2190, 0.1863, 0.1648 (mean 0.198 ± 0.021). The configuration is identical in both logs (bins 10, γ 2.618, lr 3.22e-4, 20 epochs), but the predictions already differ after epoch 1, so the initial weights differ.
  - **Cause:** in the final run, `torch.manual_seed(seed)` was called once per seed, and Pos and then Neg were trained in the same loop. The DEP network was therefore initialised from the random-number state left after the PLE-HM training. <!-- src: legacy/train.py:508-512; git 5c35f0e DCMFNet/train.py --> `dcmfnet_test_predictions.py` and `shap_analysis.py` reseed before each model. <!-- src: dcmfnet_test_predictions.py:76-81 --> The DEP difference is due to random initialisation alone. It is not a configuration or data error.
  - **Consistency:** the SHAP retrain and the prediction retrain give identical DEP models, so all of the new DCMFNet outputs describe the same models.
  - **Status:** author chose to reproduce the original models (2026-10-24). `dcmfnet_test_predictions.py` and `shap_analysis.py` now train both models of a seed with `train_seed_models()`: seed once, train Pos, evaluate it on a shuffled test loader (as `evaluate_final_test`), then train Neg. To rerun: the two DCMFNet scripts, then `prediction_evaluation.py` and `attribution_agreement.py`. Check against `Final Best Model` per-seed R².
- **Update 2026-10-24 (rerun with `train_seed_models()`, commit 98d2fd4): resolved.**
  - **PLE-HM:** unchanged, equal to `Final Best Model` to 7 digits.
  - **DEP:** 0.20234, 0.21968, 0.21897, 0.18632, 0.16479 (mean 0.1984), against 0.20226, 0.21968, 0.21897, 0.18632, 0.16479 (mean 0.1984). Seeds 43–46 agree to 7 digits; seed 42 differs by 8.5 × 10⁻⁵ (cause not traced, possibly a non-deterministic GPU/CPU operation; it does not change any reported value at 3 decimals). <!-- src: Models/DCMFNet/Results/Retrained Results/DCMFNet_retrained_test_metrics.csv; Models/DCMFNet/Results/Final Best Model/Neg_test_results_per_seed.csv -->
  - The prediction and SHAP retrains give the same R² per seed, so all DCMFNet outputs in `Models/Model Comparison/` now describe the published models. **Draft follows:** the reproduced values; the seed-42 difference is noted here only.
