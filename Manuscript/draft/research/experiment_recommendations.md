# Experiment and Analysis Recommendations

None of these analyses have been run. All require author approval (agent rule: no new experiments or code changes without authorization). Every experiment states the claim it tests and a **pre-specified decision rule**, so that results are not selected after the fact.

## Tier 1 — required for any framing (validity)

### E-01 Re-specify and re-fit GLMM/GAMM interactions (fixes D-01)
- **Claim evaluated:** whether pre-specified SUD15 × covariate interactions improve prediction over main effects, and which interactions are supported.
- **Design:** Build all pairs with `as.vector(outer(SUD15_vars, X_vars, paste, sep=":"))` (546 terms). Because 546 interaction terms on ~3,000 rows is ill-posed, fit (a) main-effects-only GLMM; (b) a theory-driven subset pre-registered by the author (e.g., cannabis/other drugs × PRS_all, ACE, SCZ15); (c) a penalized all-pairs model (glmnet with grouped twin CV, or `glmmLasso`). Report out-of-sample metrics on E-02 partitions. Apply multiplicity control (FDR) for coefficient claims.
- **Decision rule:** Interactions are "supported for prediction" only if (b) or (c) improves paired test RMSE over (a) with a corrected-resampled-t 95% CI excluding 0 (E-08).

### E-02 Common partitions across languages (fixes D-02)
- Export `cmpair` IDs per seed from the Python `random_split` and read them in R. Re-run GLMM/GAMM. Also store partitions as artifacts for reproducibility.

### E-03 Leakage-free model selection for DCMFNet (fixes D-04, D-09)
- **Design:** Nested protocol. For each outer split (seed), run Optuna + depth selection on the outer-training data only (inner grouped CV or inner 80/20), then evaluate once on the outer test set. Pre-specify that the configuration choice (uniform vs per-modality depth) is made on inner validation only. Budget permitting, use fewer trials (e.g., 30) × 5 outer splits.
- **Fairness:** Give the strongest baselines (LassoCV already internally tuned; add tuned LightGBM/XGBoost, MLP, TabNet or FT-Transformer) an equal tuning budget within the same nested protocol.
- **Decision rule:** Report all models from this protocol as primary results. The thesis numbers are retained only as "development-phase" results in the Supplement, if at all.

### E-04 Feature screening inside the training folds (fixes D-09)
- Either show that feature screening was outcome-independent (missingness/correlation thresholds only), or move SHAP-guided grouping inside outer-training folds. At minimum, run a sensitivity analysis with an outcome-independent feature set.

### E-05 Temporal sensitivity analysis (D-14)
- **Claim evaluated:** prediction of age-18 symptoms from information available by age 15.
- **Design:** Re-run primary models excluding ACE18 and SUD18 (concurrent with outcome). Report both "prospective (≤ age 15)" and "concurrent" predictor sets.
- **Decision rule:** Pre-specify the prospective set as primary if the framing is risk prediction.

### E-06 Validated attribution + fusion ablation (D-12; tests C3)
- **Ablations (paired, same splits):** (i) independent stream only (removes all IGF blocks); (ii) fused stream only; (iii) full model; (iv) full model with SUD15 anchor replaced by a permuted copy (breaks exposure-specific fusion).
- **Attribution:** grouped permutation importance (per modality, on test sets) for DCMFNet and LassoCV/GBDT. For interaction-specific evidence, use pairwise interaction measures (e.g., Friedman's H or SHAP interaction values on a tree model) `[CITATION NEEDED]`.
- **Decision rule:** C3 is supported only if (iii) beats (i) on paired RMSE with CI excluding 0, **and** (iv) degrades performance.

### E-07 Standardize inputs for DCMFNet (D-13)
- Fit z-scaling on training folds only. This is required for meaningful gate comparison and may change performance. Report as a protocol change.

## Tier 2 — statistical adequacy

### E-08 Paired uncertainty for model comparisons (A-01)
- **Estimand:** mean difference in test RMSE (and R², Spearman ρ) between model pairs across repeated grouped random splits.
- **Method:** corrected resampled t-test (Nadeau & Bengio 2003) [R-NADEAU] over ≥ 10 repeated splits (5 is low), or a pair-level bootstrap within each test set. Holm correction across the pre-specified comparison set (DCMFNet vs each of: OLS, LassoCV, tuned GBDT, GLMM main-effects).
- **Unit of analysis:** twin pair (resampling must keep pairs together).
- **Note:** The saved per-seed CSVs already allow a 5-split paired comparison between DCMFNet and the LazyPredict models without new training. This is a recomputation from saved outputs and still needs author approval. It is the cheapest high-value step.

### E-09 Calibration (TRIPOD+AI)
- Calibration slope and intercept (regress observed on predicted), calibration plots with LOESS on held-out data for all models. The thesis calibration plots are seed-42 scatter plots only.

### E-10 Tail / high-risk performance (tests C5)
- Pre-specify: top-decile identification (sensitivity/PPV for observed top 10% of the outcome), MAE within the top decile, and Spearman ρ. Compare with and without the imbalanced loss (loss ablation: MSE, Huber, Huber+reweighting, full).
- **Decision rule:** C5 is supported only if the full loss improves the top-decile metrics with paired CI excluding 0 without worse overall calibration.

### E-11 Twin dependence in DCMFNet (optional)
- Not required for prediction, since grouped splits already avoid pair leakage. Document it as a limitation. Optional: pair-level random-effect head or post-hoc pair-mean correction.

## Tier 3 — strengthening and scope

- **E-12 Missing-data sensitivity:** multiple imputation (MICE within training folds) vs complete case. Compare characteristics of included (n = 4,033) vs excluded participants (A-02).
- **E-13 Sex-stratified performance** (SAGER requirement; AIIM p.8–9): report metrics by sex, with caution for small subgroups.
- **E-14 Learning curve:** performance vs training size for linear vs DCMFNet. This directly tests the "data-limited" explanation offered in the thesis.
- **E-15 External validation:** not available within CATSS. State it as a limitation and name candidate cohorts `[AUTHOR INPUT]` (e.g., ALSPAC, TRAILS, Generation R) without claiming access.

- **E-16 Planted-interaction simulation (added 2026-10-19 for Option B; the main methodological contribution):**
  - **Design:** keep the real predictor matrix X of the analysis sample. Generate semi-synthetic outcomes y = f_add(X) + γ·g_int(X) + ε.
    - f_add: the additive fit of the corrected penalised linear model.
    - g_int: a small set of anchor × group product terms (substance use × PRS / ACE / SCZ15), standardised.
    - ε: noise calibrated so that the γ = 0 R² matches the observed R².
  - **Grid:** γ from 0 to clearly large, on the same grouped splits, with ≥ 10 replicates per γ.
  - **Models:** DCMFNet (fixed protocol from E-03), a penalised additive model, the penalised interaction model (E-01) and GBDT.
  - **Pre-specified decision rule:** report the smallest γ at which DCMFNet beats the additive model with paired 95% CIs excluding 0. Compare that γ with the interaction strength estimated from the real data (E-01).
    - If the real-data estimate lies far below the threshold, conclude "insufficient interaction signal at this n".
    - If it lies above the threshold and DCMFNet still does not gain on real data, conclude "architecture/training limitation".
  - **Data governance:** run on the secure server only. Semi-synthetic outcomes are methodological artefacts and must never be reported as clinical findings. Only aggregate metrics leave the server.
  - **Approval:** Required (new experiment; data access).

## Option B minimum package (order, after author approval)
1. E-08 paired comparison from the saved per-seed CSVs. It is cheap, needs no data access, and gives an immediate read on the DCMFNet vs OLS/Lasso gap.
2. E-02 shared partitions → E-01 corrected GLMM/GAMM.
3. E-03 + E-04 leakage-free selection and screening (equal tuning budget for strong baselines).
4. E-06 fusion-stream ablation; E-14 learning curves.
5. E-16 planted-interaction simulation.
6. E-05 prospective predictors; E-09 calibration; E-10 upper tail.
Optional: E-12, E-13 (SAGER), E-11.
