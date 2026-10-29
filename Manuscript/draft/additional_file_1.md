# Additional file 1

<!--
STATUS 2026-10-26: all tables built from exported, aggregated files (no server needed) by
scratchpad build_af1.py (Tables S1, S3, S4, S5) and a one-off script (Tables S2, S2b). Sources:
  Table S1: git c49eec1:DCMFNet/tuning.py (search space), c49eec1:DCMFNet/Results/Tuning/{Pos,Neg}_optuna_results.csv
            and _best_params.txt (= Models/DCMFNet/hyperparameters.json except the PLE-HM depths) — D-21
  Table S2/S2b: Models/ML/Results/{Pos,Neg}_benchmark_summary.csv; ML_comparators_tuning.csv
  Table S3: Models/GLMM/Results/{Pos,Neg}_GLMM_updated_all_coefs.csv; Models/GAMM/Results/GAMM_updated_{Pos,Neg}_all_parametric_coefs.csv, _smooth_terms.csv (seed 42 fits)
  Table S4: model terms in Pos_GLMM_updated_all_coefs.csv (105 main effects)
  Table S5: Models/DCMFNet/Results/Complexity Analysis/{Pos,Neg}_depth_sweep_results.csv, _optimal_depths.csv, _complexity_comparison.csv,
            complexity_analysis.log (single split, seed 42; baseline depth 3 / 2). Git ef25fa8 (29 Apr 2026).
Rebuild if these files change.
-->

Contents:

- Table S1. DCMFNet hyperparameter search and tuning history
- Table S2. Default-settings screening of 40 machine-learning algorithms
- Table S2b. Hyperparameters selected for the tuned comparators
- Table S3. Fixed effects and smooth terms of the mixed models
- Table S4. Predictors by group
- Table S5. Selection of the fusion depths
- Table S6. Test-set performance on all metrics

## Table S1. DCMFNet hyperparameter search

Hyperparameters were searched with Optuna (tree-structured Parzen estimator sampler with default settings, no fixed sampler seed; median pruner with 5 start-up trials and 5 warm-up epochs), 50 trials per outcome. Each trial trained the network on 80% of the seed-42 training set (split by twin pair) and was scored by the lowest validation RMSE on the remaining 20%. In the search, one fusion depth applied to all blocks. Of the 50 trials, 28 (PLE-HM) and 32 (DEP) ran to completion and the rest were pruned. The best validation RMSE was 0.0880 (PLE-HM) and 0.1484 (DEP). The selected values were used for the final models, except that the PLE-HM fusion depths were then set per block (Table S5).

| Hyperparameter | Search range | PLE-HM selected | DEP selected |
|---|---|---|---|
| Learning rate | 10⁻⁵ to 10⁻² (log scale) | 1.00e-04 | 3.22e-04 |
| Batch size | 4, 6, 8, 16, 32 | 4 | 6 |
| Maximum epochs | 10 to 30 | 23 | 20 |
| Weight decay | 10⁻⁵ to 10⁻² (log scale) | 2.95e-05 | 1.79e-04 |
| Fusion depth (all blocks) | 1 to 7 | 3 (replaced by per-block depths, Table S5) | 2 |
| Dropout | 0.1 to 0.5 | 0.234 | 0.326 |
| Gate reduction ratio r | 2 to 8 | 6 | 5 |
| Gate minimum width h_min | 2 to 16 | 13 | 5 |
| Base loss | MSE, Huber | Huber | Huber |
| Huber δ | 0.01 to 0.2 | 0.086 | 0.116 |
| Focal γ | 0.5 to 3.0 | 1.32 | 2.62 |
| Bins K | 5 to 20 | 13 | 10 |
| Scheduler patience | 2 to 5 | 4 | 2 |
| Scheduler factor | 0.3 to 0.7 | 0.432 | 0.549 |
| Early-stopping patience | 3 to 8 | 5 | 7 |

**Tuning history.** The search above is one of at least five tuning runs carried out over five days, which differed in their search spaces (for example, uniform or per-block depths). The hyperparameters used in this study come from the run shown here; the choice among runs also took the test-set performance of the resulting models into account. The outputs of the other runs are kept in the version history of the code repository. Because of this, and because the seed-42 training set used for tuning overlaps the test sets of the other four splits, DCMFNet's test performance may be optimistic (see Methods and the limitations in the Discussion). <!-- src: D-21; git c49eec1, 9079d3e, cfd5901, a68071b and the 29 Apr run (Models/DCMFNet/Results/Tuning/, per-block depths, not used) -->

## Table S2. Default-settings screening of 40 machine-learning algorithms

Test-set R² and Spearman's ρ, mean (SD) over the five twin-pair-grouped splits used for all models. Each algorithm was fitted with its scikit-learn default settings on the 105 unscaled inputs, without tuning and without modelling twin pairs; estimators with a built-in penalty search (for example LassoCV, RidgeCV, LassoLarsIC) chose their penalty within the training split, without grouping by twin pair. Algorithms are ordered by R² for PLE-HM. These results describe default settings only and were not used to select models.

| Algorithm | PLE-HM R² | PLE-HM Spearman ρ | DEP R² | DEP Spearman ρ |
|---------------------|-------------|-------------|-------------|-------------|
| LassoCV | 0.261 (0.019) | 0.492 (0.019) | 0.234 (0.017) | 0.472 (0.020) |
| LassoLarsCV | 0.261 (0.019) | 0.492 (0.019) | 0.234 (0.017) | 0.472 (0.020) |
| ElasticNetCV | 0.261 (0.019) | 0.492 (0.019) | 0.234 (0.017) | 0.471 (0.020) |
| LassoLarsIC | 0.261 (0.019) | 0.492 (0.019) | 0.234 (0.017) | 0.471 (0.019) |
| LarsCV | 0.259 (0.021) | 0.491 (0.018) | 0.235 (0.016) | 0.472 (0.019) |
| BayesianRidge | 0.255 (0.021) | 0.486 (0.022) | 0.226 (0.020) | 0.461 (0.020) |
| RidgeCV | 0.246 (0.022) | 0.478 (0.022) | 0.218 (0.024) | 0.456 (0.021) |
| KernelRidge | 0.244 (0.021) | 0.475 (0.021) | 0.215 (0.025) | 0.454 (0.020) |
| Ridge | 0.242 (0.021) | 0.474 (0.021) | 0.215 (0.025) | 0.454 (0.021) |
| HuberRegressor | 0.240 (0.022) | 0.479 (0.020) | 0.206 (0.023) | 0.454 (0.022) |
| TransformedTargetRegressor | 0.239 (0.021) | 0.473 (0.022) | 0.215 (0.025) | 0.453 (0.020) |
| LinearRegression | 0.239 (0.021) | 0.473 (0.022) | 0.215 (0.025) | 0.453 (0.020) |
| Lars | 0.236 (0.022) | 0.470 (0.021) | 0.214 (0.025) | 0.452 (0.019) |
| GradientBoostingRegressor | 0.229 (0.019) | 0.461 (0.015) | 0.214 (0.012) | 0.455 (0.019) |
| SGDRegressor | 0.228 (0.029) | 0.469 (0.021) | 0.208 (0.018) | 0.451 (0.017) |
| OrthogonalMatchingPursuitCV | 0.225 (0.023) | 0.463 (0.018) | 0.227 (0.015) | 0.462 (0.019) |
| OrthogonalMatchingPursuit | 0.225 (0.023) | 0.463 (0.018) | 0.227 (0.015) | 0.462 (0.019) |
| RandomForestRegressor | 0.221 (0.021) | 0.455 (0.013) | 0.185 (0.016) | 0.421 (0.027) |
| TweedieRegressor | 0.211 (0.016) | 0.458 (0.021) | 0.181 (0.013) | 0.438 (0.018) |
| ExtraTreesRegressor | 0.209 (0.026) | 0.446 (0.018) | 0.138 (0.030) | 0.396 (0.035) |
| HistGradientBoostingRegressor | 0.199 (0.015) | 0.438 (0.006) | 0.174 (0.025) | 0.420 (0.026) |
| LGBMRegressor | 0.196 (0.018) | 0.433 (0.010) | 0.182 (0.029) | 0.426 (0.030) |
| SVR | 0.176 (0.015) | 0.408 (0.008) | 0.155 (0.026) | 0.413 (0.025) |
| NuSVR | 0.158 (0.027) | 0.423 (0.014) | 0.140 (0.026) | 0.412 (0.024) |
| BaggingRegressor | 0.132 (0.029) | 0.374 (0.021) | 0.116 (0.019) | 0.380 (0.014) |
| PoissonRegressor | 0.110 (0.008) | 0.435 (0.021) | 0.114 (0.010) | 0.413 (0.021) |
| LinearSVR | 0.070 (0.077) | 0.439 (0.036) | 0.101 (0.056) | 0.423 (0.018) |
| KNeighborsRegressor | 0.034 (0.014) | 0.315 (0.017) | 0.018 (0.031) | 0.302 (0.030) |
| AdaBoostRegressor | 0.028 (0.034) | 0.387 (0.029) | 0.005 (0.033) | 0.420 (0.015) |
| PassiveAggressiveRegressor‡ | 0.024 (0.249) | 0.404 (0.034) | −0.226 (0.511) | 0.405 (0.035) |
| QuantileRegressor† | −0.000 (0.000) | — | −0.052 (0.056) | — |
| ElasticNet† | −0.001 (0.001) | — | −0.002 (0.002) | — |
| LassoLars† | −0.001 (0.001) | — | −0.002 (0.002) | — |
| DummyRegressor† | −0.001 (0.001) | — | −0.002 (0.002) | — |
| Lasso† | −0.001 (0.001) | — | −0.002 (0.002) | — |
| MLPRegressor‡ | −0.359 (0.144) | 0.295 (0.022) | −0.344 (0.153) | 0.289 (0.020) |
| DecisionTreeRegressor‡ | −0.574 (0.066) | 0.210 (0.018) | −0.552 (0.073) | 0.210 (0.007) |
| ExtraTreeRegressor‡ | −0.578 (0.111) | 0.226 (0.030) | −0.575 (0.067) | 0.224 (0.020) |
| GaussianProcessRegressor‡ | −1.567 (0.043) | −0.329 (0.008) | −1.759 (0.093) | −0.213 (0.032) |
| RANSACRegressor‡ | −9.902 (4.375) | 0.185 (0.034) | −6.420 (1.827) | 0.126 (0.030) |

† Predicted a constant (or nearly constant) value on at least one outcome, so Spearman's ρ is undefined (—); for the lasso, elastic net and LARS lasso this is the effect of the default penalty (α = 1) on these outcome scales.
‡ Negative mean R² on at least one outcome: predictions were worse than the test-set mean.

<!-- src: Models/ML/Results/{Pos,Neg}_benchmark_summary.csv (r2_mean, r2_std, spearman_rho_mean, spearman_rho_std; SD as computed by lazy_predict.py) -->

## Table S2b. Hyperparameters selected for the tuned comparators

Selected for each split, predictor set and outcome by 5-fold cross-validation within the training split, with folds grouped by twin pair and mean squared error (MSE) as the criterion. Linear regression has no hyperparameters. "All" uses all 105 predictors; "Age 15" omits adverse experiences and substance use at age 18.

**Elastic net** (inputs standardised; α is the penalty strength, L1 ratio 1 = lasso, close to 0 = ridge).

| Predictors | Outcome | Split | α | L1 ratio | Inner-CV MSE (×10³) | Test R² |
|---|---|---|---|---|---|---|
| All | PLE-HM | 42 | 0.0393 | 0.05 | 7.78 | 0.293 |
| All | PLE-HM | 43 | 0.0399 | 0.05 | 7.47 | 0.250 |
| All | PLE-HM | 44 | 0.0381 | 0.05 | 7.49 | 0.257 |
| All | PLE-HM | 45 | 0.0319 | 0.05 | 7.55 | 0.244 |
| All | PLE-HM | 46 | 0.0399 | 0.05 | 7.66 | 0.258 |
| All | DEP | 42 | 0.0086 | 0.30 | 22.74 | 0.258 |
| All | DEP | 43 | 0.0025 | 1.00 | 21.89 | 0.251 |
| All | DEP | 44 | 0.0275 | 0.10 | 22.49 | 0.230 |
| All | DEP | 45 | 0.0440 | 0.05 | 22.42 | 0.221 |
| All | DEP | 46 | 0.0039 | 1.00 | 22.63 | 0.226 |
| Age 15 | PLE-HM | 42 | 0.0354 | 0.05 | 8.29 | 0.214 |
| Age 15 | PLE-HM | 43 | 0.0419 | 0.05 | 8.02 | 0.184 |
| Age 15 | PLE-HM | 44 | 0.0399 | 0.05 | 8.10 | 0.204 |
| Age 15 | PLE-HM | 45 | 0.0300 | 0.05 | 8.19 | 0.192 |
| Age 15 | PLE-HM | 46 | 0.0367 | 0.05 | 8.24 | 0.197 |
| Age 15 | DEP | 42 | 0.0484 | 0.05 | 24.24 | 0.182 |
| Age 15 | DEP | 43 | 0.0542 | 0.05 | 23.51 | 0.180 |
| Age 15 | DEP | 44 | 0.0113 | 0.30 | 24.33 | 0.179 |
| Age 15 | DEP | 45 | 0.0582 | 0.05 | 24.31 | 0.175 |
| Age 15 | DEP | 46 | 0.0586 | 0.05 | 24.29 | 0.157 |

**Gradient boosting** (scikit-learn HistGradientBoostingRegressor; 40 random configurations per fit; early stopping off).

| Predictors | Outcome | Split | Learning rate | Trees | Max depth | Max leaves | Min leaf size | L2 penalty | Inner-CV MSE (×10³) | Test R² |
|---|---|---|---|---|---|---|---|---|---|---|
| All | PLE-HM | 42 | 0.032 | 225 | 2 | 5 | 72 | 0.00136 | 7.90 | 0.277 |
| All | PLE-HM | 43 | 0.026 | 352 | 2 | 10 | 115 | 0.00113 | 7.59 | 0.227 |
| All | PLE-HM | 44 | 0.014 | 487 | 6 | 4 | 104 | 2.18 | 7.71 | 0.254 |
| All | PLE-HM | 45 | 0.040 | 348 | 2 | 20 | 79 | 0.188 | 7.67 | 0.232 |
| All | PLE-HM | 46 | 0.031 | 256 | 4 | 6 | 62 | 0.456 | 7.79 | 0.231 |
| All | DEP | 42 | 0.032 | 225 | 2 | 5 | 72 | 0.00136 | 23.14 | 0.254 |
| All | DEP | 43 | 0.016 | 312 | 3 | 25 | 44 | 0.0158 | 22.27 | 0.239 |
| All | DEP | 44 | 0.014 | 487 | 6 | 4 | 104 | 2.18 | 22.87 | 0.225 |
| All | DEP | 45 | 0.051 | 292 | 2 | 12 | 109 | 0.257 | 22.79 | 0.207 |
| All | DEP | 46 | 0.031 | 256 | 4 | 6 | 62 | 0.456 | 23.07 | 0.219 |
| Age 15 | PLE-HM | 42 | 0.043 | 158 | 2 | 13 | 34 | 0.126 | 8.38 | 0.205 |
| Age 15 | PLE-HM | 43 | 0.029 | 288 | 2 | 30 | 132 | 0.00151 | 8.17 | 0.164 |
| Age 15 | PLE-HM | 44 | 0.014 | 487 | 6 | 4 | 104 | 2.18 | 8.27 | 0.189 |
| Age 15 | PLE-HM | 45 | 0.014 | 203 | 3 | 24 | 36 | 0.548 | 8.34 | 0.182 |
| Age 15 | PLE-HM | 46 | 0.013 | 449 | 2 | 19 | 143 | 0.127 | 8.32 | 0.176 |
| Age 15 | DEP | 42 | 0.019 | 317 | 3 | 15 | 181 | 0.154 | 24.68 | 0.169 |
| Age 15 | DEP | 43 | 0.016 | 312 | 3 | 25 | 44 | 0.0158 | 23.58 | 0.161 |
| Age 15 | DEP | 44 | 0.014 | 487 | 6 | 4 | 104 | 2.18 | 24.59 | 0.173 |
| Age 15 | DEP | 45 | 0.051 | 292 | 2 | 12 | 109 | 0.257 | 24.64 | 0.156 |
| Age 15 | DEP | 46 | 0.013 | 449 | 2 | 19 | 143 | 0.127 | 24.69 | 0.170 |

<!-- src: Models/ML/Results/ML_comparators_tuning.csv (selected_hyperparameters, inner_cv_mse, test_r2) -->

## Table S3. Fixed effects and smooth terms of the mixed models

Estimates from the models fitted on the training set of the first split (seed 42). Outcomes are scaled to 0–1. Main effects entered on their original scales. The interaction terms use standardised scores: each group "score" is the mean of the group's items after standardising each item, standardised again, and each PRS in an interaction is standardised (z). Means and SDs came from the training set. p values are unadjusted. In the generalised additive mixed model (GAMM), each PRS entered as a thin-plate regression spline (Table S3c) instead of a linear term, and the twin-pair random intercept was a random-effect smooth. GLMM, generalised linear mixed model (Gaussian outcome, identity link). <!-- src: GLMM_updated_split.R extract_significant_coefs (seed 42 model), add_interaction_scores; GAMM_updated_split.R RANDOM_EFFECT_TERM -->

### Table S3a. PLE-HM: fixed effects of the linear and additive mixed models

| Term | GLMM β (SE) | GLMM p | GAMM β (SE) | GAMM p |
|---|---|---|---|---|
| **Substance use, age 15 (anchor)** | | | | |
| `SUD15_OtherDrugs15` | 0.0171 (0.0082) | 0.0373 | 0.0175 (0.0082) | 0.0333 |
| `SUD15_Painkillers_opioids15` | 0.0081 (0.0058) | 0.163 | 0.0083 (0.0058) | 0.153 |
| `SUD15_Snuff15` | 0.0055 (0.0049) | 0.258 | 0.0055 (0.0049) | 0.260 |
| `SUD15_Cannabis15` | 0.0102 (0.0092) | 0.267 | 0.0108 (0.0092) | 0.239 |
| `SUD15_Alcohol15` | -0.0022 (0.0029) | 0.460 | -0.0018 (0.0029) | 0.533 |
| `SUD15_Cigarettes15` | -0.0011 (0.0043) | 0.803 | -0.0008 (0.0043) | 0.853 |
| **Polygenic risk scores** | | | | |
| `PRS_all` | 0.0199 (0.0056) | < 0.001 | smooth, Table S3c | – |
| `PRS_Miscellaneous` | 0.0366 (0.0189) | 0.0531 | smooth, Table S3c | – |
| `PRS_Hippocampal_CA1_3` | -0.0299 (0.0197) | 0.129 | smooth, Table S3c | – |
| `PRS_Amygdala_excitatory` | -0.0239 (0.0191) | 0.211 | smooth, Table S3c | – |
| `PRS_Deep_layer_intratelencepha` | -0.0231 (0.0205) | 0.260 | smooth, Table S3c | – |
| `PRS_CGE_interneuron` | -0.0165 (0.0186) | 0.374 | smooth, Table S3c | – |
| `PRS_Cerebellar_inhibitory` | -0.0143 (0.0176) | 0.415 | smooth, Table S3c | – |
| `PRS_MGE_interneuron` | -0.0116 (0.0187) | 0.535 | smooth, Table S3c | – |
| `PRS_Deep_layer_corticothalamic` | 0.0098 (0.0185) | 0.597 | smooth, Table S3c | – |
| `PRS_Hippocampal_CA4` | 0.0093 (0.0191) | 0.626 | smooth, Table S3c | – |
| `PRS_Upper_layer_intratelenceph` | 0.0077 (0.0192) | 0.688 | smooth, Table S3c | – |
| `PRS_Thalamic_excitatory` | 0.0055 (0.0168) | 0.746 | smooth, Table S3c | – |
| `PRS_LAMP5_LHX6_and_Chandelier` | -0.0058 (0.0192) | 0.764 | smooth, Table S3c | – |
| `PRS_Medium_spiny_neuron` | 0.0047 (0.0192) | 0.807 | smooth, Table S3c | – |
| `PRS_Hippocampal_dentate_gyrus` | 0.0033 (0.0181) | 0.857 | smooth, Table S3c | – |
| `PRS_Eccentric_medium_spiny_neu` | 0.0004 (0.0179) | 0.983 | smooth, Table S3c | – |
| **Symptoms, age 15** | | | | |
| `SCZ15_Too_much_energy15` | 0.0108 (0.0024) | < 0.001 | 0.0109 (0.0024) | < 0.001 |
| `SCZ15_Seen_hallucinations15` | 0.0155 (0.0040) | < 0.001 | 0.0156 (0.0040) | < 0.001 |
| `SCZ15_Talking_fast15` | 0.0091 (0.0025) | < 0.001 | 0.0091 (0.0025) | < 0.001 |
| `SCZ15_Read_others_minds15` | 0.0083 (0.0024) | < 0.001 | 0.0083 (0.0024) | < 0.001 |
| `SCZ15_Extreme_excitement15` | 0.0059 (0.0019) | 0.00175 | 0.0058 (0.0019) | 0.00198 |
| `SCZ15_Hear_voices15` | 0.0206 (0.0067) | 0.00201 | 0.0202 (0.0066) | 0.00242 |
| `SCZ15_headaches15` | 0.0079 (0.0027) | 0.00334 | 0.0079 (0.0027) | 0.00363 |
| `SCZ15_Special_messages15` | 0.0187 (0.0064) | 0.00354 | 0.0185 (0.0064) | 0.00384 |
| `SCZ15_Spied15` | 0.0051 (0.0026) | 0.0506 | 0.0052 (0.0026) | 0.0493 |
| `SCZ15_Irritable15` | 0.0044 (0.0027) | 0.0955 | 0.0044 (0.0027) | 0.0973 |
| `SCZ15_Special_powers15` | 0.0084 (0.0058) | 0.147 | 0.0081 (0.0058) | 0.160 |
| `SCZ15_PROD_seen_hallucinations9` | 0.0117 (0.0094) | 0.214 | 0.0122 (0.0094) | 0.194 |
| `SCZ15_Rage_attacks15` | 0.0048 (0.0039) | 0.220 | 0.0048 (0.0039) | 0.218 |
| `SCZ15_Sexual_inappropriate15` | 0.0114 (0.0107) | 0.283 | 0.0108 (0.0106) | 0.310 |
| `SCZ15_Unrealistic_abilities15` | 0.0071 (0.0073) | 0.330 | 0.0079 (0.0073) | 0.280 |
| `SCZ15_Racing_thoughts15` | 0.0022 (0.0027) | 0.425 | 0.0023 (0.0027) | 0.405 |
| `SCZ15_easily_scared15` | -0.0022 (0.0028) | 0.444 | -0.0019 (0.0028) | 0.510 |
| `SCZ15_unhappy15` | -0.0026 (0.0036) | 0.464 | -0.0028 (0.0036) | 0.438 |
| `SCZ15_Others_Read_thoughts15` | 0.0017 (0.0028) | 0.542 | 0.0016 (0.0028) | 0.577 |
| `SCZ15_worry15` | -0.0014 (0.0027) | 0.603 | -0.0015 (0.0027) | 0.576 |
| `SCZ15_Under_control_special_power15` | 0.0020 (0.0042) | 0.645 | 0.0021 (0.0042) | 0.624 |
| `SCZ15_Not_tired15` | -0.0008 (0.0022) | 0.715 | -0.0009 (0.0022) | 0.685 |
| `SCZ15_lose_confidence15` | 0.0009 (0.0025) | 0.719 | 0.0010 (0.0025) | 0.687 |
| **ADHD symptoms, age 9** | | | | |
| `ADHD9_talk_excess9` | 0.0067 (0.0036) | 0.0599 | 0.0068 (0.0036) | 0.0547 |
| `ADHD9_easily_bored9` | -0.0070 (0.0041) | 0.0867 | -0.0070 (0.0041) | 0.0863 |
| `ADHD9_difficulty_organizing_tasks9` | -0.0114 (0.0070) | 0.102 | -0.0116 (0.0070) | 0.0956 |
| `ADHD9_not_listening9` | -0.0058 (0.0049) | 0.238 | -0.0055 (0.0049) | 0.260 |
| `ADHD9_often_motor9` | 0.0066 (0.0072) | 0.355 | 0.0067 (0.0072) | 0.348 |
| `ADHD9_easily_distracted9` | 0.0041 (0.0044) | 0.359 | 0.0042 (0.0044) | 0.339 |
| `ADHD9_avoid_mental_effort_tasks9` | 0.0037 (0.0049) | 0.450 | 0.0035 (0.0049) | 0.477 |
| `ADHD9_difficulty_with_attention9` | 0.0041 (0.0054) | 0.456 | 0.0040 (0.0054) | 0.461 |
| `ADHD9_difficulty_waiting9` | 0.0035 (0.0047) | 0.459 | 0.0033 (0.0047) | 0.480 |
| `ADHD9_often_run9` | -0.0034 (0.0053) | 0.513 | -0.0033 (0.0053) | 0.529 |
| `ADHD9_often_interrupt9` | -0.0033 (0.0053) | 0.539 | -0.0029 (0.0053) | 0.581 |
| `ADHD9_often_lose_things9` | 0.0020 (0.0043) | 0.638 | 0.0018 (0.0043) | 0.676 |
| `ADHD9_difficulty_following_instructions9` | -0.0023 (0.0070) | 0.739 | -0.0024 (0.0070) | 0.734 |
| `ADHD9_often_forgetful9` | 0.0016 (0.0050) | 0.752 | 0.0016 (0.0050) | 0.748 |
| `ADHD9_often_move9` | 0.0013 (0.0060) | 0.830 | 0.0010 (0.0060) | 0.862 |
| `ADHD9_often_blurt_out9` | -0.0009 (0.0045) | 0.851 | -0.0014 (0.0045) | 0.764 |
| `ADHD9_difficulty_hold_still9` | 0.0007 (0.0049) | 0.884 | 0.0008 (0.0049) | 0.875 |
| `ADHD9_fail_close_attention9` | 0.0006 (0.0049) | 0.898 | 0.0006 (0.0049) | 0.909 |
| `ADHD9_difficulty_calm9` | -0.0009 (0.0086) | 0.918 | -0.0004 (0.0086) | 0.963 |
| **Autism-related traits, age 9** | | | | |
| `ASD9_difficulty_converse9` | -0.0324 (0.0124) | 0.00929 | -0.0323 (0.0124) | 0.00936 |
| `ASD9_easily_influenced9` | 0.0086 (0.0038) | 0.0222 | 0.0085 (0.0038) | 0.0242 |
| `ASD9_difficulty_socialize9` | 0.0224 (0.0099) | 0.0241 | 0.0212 (0.0099) | 0.0326 |
| `ASD9_absorbed_details9` | 0.0134 (0.0060) | 0.0243 | 0.0134 (0.0059) | 0.0239 |
| `ASD9_absorbed_problems9` | -0.0159 (0.0098) | 0.106 | -0.0157 (0.0098) | 0.109 |
| `ASD9_difficulty_pretend_play9` | -0.0154 (0.0096) | 0.110 | -0.0152 (0.0096) | 0.114 |
| `ASD9_delay_language9` | 0.0067 (0.0043) | 0.120 | 0.0067 (0.0043) | 0.121 |
| `ASD9_uninterested_sharing9` | 0.0165 (0.0130) | 0.206 | 0.0161 (0.0130) | 0.214 |
| `ASD9_difficulty_expect_behavior9` | -0.0120 (0.0096) | 0.215 | -0.0116 (0.0096) | 0.230 |
| `ASD9_absorbed_own9` | -0.0046 (0.0059) | 0.436 | -0.0044 (0.0059) | 0.461 |
| `ASD9_difficulty_on_track9` | 0.0045 (0.0062) | 0.469 | 0.0046 (0.0062) | 0.461 |
| `ASD9_talk_too_high_low9` | -0.0028 (0.0051) | 0.581 | -0.0029 (0.0051) | 0.573 |
| `ASD9_own_terms9` | 0.0028 (0.0071) | 0.697 | 0.0028 (0.0071) | 0.694 |
| `ASD9_dislike_change9` | 0.0017 (0.0050) | 0.736 | 0.0014 (0.0050) | 0.785 |
| `ASD9_strange_movements9` | -0.0020 (0.0069) | 0.774 | -0.0022 (0.0069) | 0.747 |
| `ASD9_difficulty_express9` | 0.0040 (0.0142) | 0.778 | 0.0041 (0.0142) | 0.775 |
| `ASD9_repeat_words9` | -0.0008 (0.0074) | 0.911 | -0.0004 (0.0074) | 0.954 |
| **Adverse experiences, age 15** | | | | |
| `ACE15_bullying_by_num15` | -0.0065 (0.0034) | 0.0554 | -0.0065 (0.0034) | 0.0526 |
| `ACE15_rumours_bullying15` | 0.0078 (0.0043) | 0.0695 | 0.0078 (0.0043) | 0.0691 |
| `ACE15_other_bullying15` | 0.0054 (0.0038) | 0.153 | 0.0053 (0.0038) | 0.161 |
| `ACE15_emotional_bullying15` | 0.0035 (0.0036) | 0.321 | 0.0033 (0.0036) | 0.353 |
| `ACE15_tease_bullying15` | 0.0015 (0.0040) | 0.706 | 0.0014 (0.0040) | 0.727 |
| `ACE15_bullying_time15` | 0.0008 (0.0023) | 0.734 | 0.0010 (0.0023) | 0.678 |
| `ACE15_bullied_often15` | -0.0001 (0.0041) | 0.988 | 0.0001 (0.0041) | 0.971 |
| **Adverse experiences, age 18** | | | | |
| `ACE18_emotional_abuse18` | -0.0353 (0.0039) | < 0.001 | -0.0355 (0.0039) | < 0.001 |
| `ACE18_witness_crime18` | -0.0216 (0.0060) | < 0.001 | -0.0215 (0.0060) | < 0.001 |
| `ACE18_hate_crime18` | -0.0197 (0.0103) | 0.0549 | -0.0198 (0.0103) | 0.0542 |
| `ACE18_other_abuse18` | -0.0476 (0.2136) | 0.824 | -0.0409 (0.2134) | 0.848 |
| **Substance use, age 18** | | | | |
| `SUD18_alcohol_often18` | 0.0094 (0.0024) | < 0.001 | 0.0093 (0.0024) | < 0.001 |
| `SUD18_drugs_often18` | 0.0165 (0.0049) | < 0.001 | 0.0163 (0.0048) | < 0.001 |
| `SUD18_cigarettes18` | 0.0037 (0.0012) | 0.00251 | 0.0037 (0.0012) | 0.00221 |
| `SUD18_snuff18` | 0.0008 (0.0010) | 0.419 | 0.0008 (0.0010) | 0.401 |
| **Parental socioeconomic indicators** | | | | |
| `SES_education_father` | -0.0026 (0.0017) | 0.117 | -0.0027 (0.0017) | 0.109 |
| `SES_birth_country_father` | -0.0035 (0.0041) | 0.398 | -0.0036 (0.0041) | 0.375 |
| `SES_birth_country_mother` | -0.0009 (0.0046) | 0.843 | -0.0009 (0.0046) | 0.851 |
| `SES_education_mother` | 0.0001 (0.0019) | 0.946 | 0.0002 (0.0019) | 0.926 |
| **Sex** | | | | |
| `SEX` | -0.0037 (0.0040) | 0.360 | -0.0037 (0.0040) | 0.355 |
| **Genotyping batch × principal component** | | | | |
| `batch_GSA3_x_PC1` | -0.3202 (0.2236) | 0.152 | -0.3106 (0.2232) | 0.164 |
| `batch_PSYC_x_PC2` | -0.1883 (0.1322) | 0.154 | -0.1796 (0.1320) | 0.174 |
| `batch_PSYC_x_PC1` | -0.0755 (0.1387) | 0.586 | -0.0635 (0.1385) | 0.647 |
| `batch_GSA3_x_PC2` | 0.0351 (0.3762) | 0.926 | 0.0352 (0.3753) | 0.925 |
| **Interactions with SUD15** | | | | |
| SUD15 score × ACE15 score | -0.0021 (0.0018) | 0.231 | -0.0021 (0.0018) | 0.232 |
| SUD15 score × ACE18 score | -0.0001 (0.0016) | 0.974 | -0.0001 (0.0016) | 0.941 |
| SUD15 score × ADHD9 score | 0.0022 (0.0016) | 0.178 | 0.0022 (0.0016) | 0.186 |
| SUD15 score × ASD9 score | -0.0021 (0.0018) | 0.251 | -0.0021 (0.0018) | 0.265 |
| SUD15 score × SCZ15 score | 0.0019 (0.0016) | 0.226 | 0.0020 (0.0016) | 0.201 |
| SUD15 score × SES score | -0.0014 (0.0019) | 0.450 | -0.0014 (0.0019) | 0.466 |
| SUD15 score × SEX | -0.0039 (0.0040) | 0.330 | -0.0042 (0.0040) | 0.293 |
| SUD15 score × SUD18 score | -0.0006 (0.0017) | 0.722 | -0.0007 (0.0017) | 0.703 |
| SUD15 score × PRS_Amygdala_excitatory (z) | 0.0038 (0.0027) | 0.151 | 0.0036 (0.0027) | 0.173 |
| SUD15 score × PRS_CGE_interneuron (z) | -0.0014 (0.0024) | 0.579 | -0.0015 (0.0024) | 0.542 |
| SUD15 score × PRS_Cerebellar_inhibitory (z) | -0.0013 (0.0022) | 0.549 | -0.0013 (0.0022) | 0.563 |
| SUD15 score × PRS_Deep_layer_corticothalamic (z) | 0.0011 (0.0024) | 0.630 | 0.0011 (0.0024) | 0.631 |
| SUD15 score × PRS_Deep_layer_intratelencepha (z) | 0.0006 (0.0028) | 0.839 | 0.0007 (0.0028) | 0.801 |
| SUD15 score × PRS_Eccentric_medium_spiny_neu (z) | -0.0019 (0.0025) | 0.442 | -0.0019 (0.0025) | 0.436 |
| SUD15 score × PRS_Hippocampal_CA1_3 (z) | -0.0011 (0.0028) | 0.710 | -0.0012 (0.0028) | 0.674 |
| SUD15 score × PRS_Hippocampal_CA4 (z) | -0.0048 (0.0025) | 0.055 | -0.0048 (0.0025) | 0.0554 |
| SUD15 score × PRS_Hippocampal_dentate_gyrus (z) | 0.0018 (0.0022) | 0.436 | 0.0018 (0.0022) | 0.426 |
| SUD15 score × PRS_LAMP5_LHX6_and_Chandelier (z) | 0.0002 (0.0028) | 0.944 | 0.0004 (0.0028) | 0.888 |
| SUD15 score × PRS_MGE_interneuron (z) | 0.0053 (0.0025) | 0.0348 | 0.0053 (0.0025) | 0.0341 |
| SUD15 score × PRS_Medium_spiny_neuron (z) | 0.0061 (0.0026) | 0.0184 | 0.0061 (0.0026) | 0.0183 |
| SUD15 score × PRS_Miscellaneous (z) | -0.0009 (0.0025) | 0.723 | -0.0007 (0.0025) | 0.776 |
| SUD15 score × PRS_Thalamic_excitatory (z) | -0.0007 (0.0019) | 0.730 | -0.0007 (0.0019) | 0.692 |
| SUD15 score × PRS_Upper_layer_intratelenceph (z) | -0.0054 (0.0025) | 0.0301 | -0.0052 (0.0025) | 0.0378 |
| SUD15 score × PRS_all (z) | 0.0009 (0.0022) | 0.677 | 0.0010 (0.0022) | 0.664 |

### Table S3b. DEP: fixed effects of the linear and additive mixed models

| Term | GLMM β (SE) | GLMM p | GAMM β (SE) | GAMM p |
|---|---|---|---|---|
| **Substance use, age 15 (anchor)** | | | | |
| `SUD15_Cigarettes15` | -0.0152 (0.0073) | 0.0368 | -0.0151 (0.0073) | 0.0389 |
| `SUD15_Snuff15` | 0.0065 (0.0083) | 0.438 | 0.0063 (0.0083) | 0.450 |
| `SUD15_Alcohol15` | 0.0021 (0.0050) | 0.668 | 0.0022 (0.0050) | 0.653 |
| `SUD15_OtherDrugs15` | -0.0060 (0.0140) | 0.671 | -0.0059 (0.0140) | 0.675 |
| `SUD15_Painkillers_opioids15` | -0.0010 (0.0099) | 0.923 | -0.0008 (0.0099) | 0.933 |
| `SUD15_Cannabis15` | -0.0009 (0.0157) | 0.953 | -0.0006 (0.0157) | 0.972 |
| **Polygenic risk scores** | | | | |
| `PRS_all` | 0.0238 (0.0097) | 0.0142 | smooth, Table S3c | – |
| `PRS_Hippocampal_CA4` | -0.0415 (0.0329) | 0.208 | smooth, Table S3c | – |
| `PRS_Cerebellar_inhibitory` | 0.0321 (0.0303) | 0.289 | smooth, Table S3c | – |
| `PRS_Eccentric_medium_spiny_neu` | 0.0310 (0.0309) | 0.315 | smooth, Table S3c | – |
| `PRS_Medium_spiny_neuron` | -0.0298 (0.0331) | 0.367 | smooth, Table S3c | – |
| `PRS_Deep_layer_intratelencepha` | -0.0288 (0.0352) | 0.414 | smooth, Table S3c | – |
| `PRS_Deep_layer_corticothalamic` | 0.0242 (0.0318) | 0.447 | smooth, Table S3c | – |
| `PRS_MGE_interneuron` | -0.0191 (0.0323) | 0.554 | smooth, Table S3c | – |
| `PRS_Amygdala_excitatory` | 0.0172 (0.0329) | 0.602 | smooth, Table S3c | – |
| `PRS_LAMP5_LHX6_and_Chandelier` | -0.0144 (0.0331) | 0.664 | smooth, Table S3c | – |
| `PRS_Miscellaneous` | 0.0080 (0.0326) | 0.807 | smooth, Table S3c | – |
| `PRS_Hippocampal_dentate_gyrus` | 0.0072 (0.0312) | 0.817 | smooth, Table S3c | – |
| `PRS_Thalamic_excitatory` | 0.0040 (0.0290) | 0.889 | smooth, Table S3c | – |
| `PRS_Upper_layer_intratelenceph` | 0.0030 (0.0330) | 0.927 | smooth, Table S3c | – |
| `PRS_Hippocampal_CA1_3` | 0.0019 (0.0339) | 0.955 | smooth, Table S3c | – |
| `PRS_CGE_interneuron` | 0.0001 (0.0320) | 0.997 | smooth, Table S3c | – |
| **Symptoms, age 15** | | | | |
| `SCZ15_unhappy15` | 0.0418 (0.0061) | < 0.001 | 0.0415 (0.0061) | < 0.001 |
| `SCZ15_headaches15` | 0.0286 (0.0046) | < 0.001 | 0.0283 (0.0046) | < 0.001 |
| `SCZ15_worry15` | 0.0218 (0.0047) | < 0.001 | 0.0219 (0.0047) | < 0.001 |
| `SCZ15_Extreme_excitement15` | -0.0111 (0.0032) | < 0.001 | -0.0114 (0.0032) | < 0.001 |
| `SCZ15_Irritable15` | 0.0145 (0.0045) | 0.00142 | 0.0145 (0.0045) | 0.00141 |
| `SCZ15_lose_confidence15` | 0.0112 (0.0044) | 0.0103 | 0.0113 (0.0044) | 0.00977 |
| `SCZ15_Too_much_energy15` | 0.0105 (0.0041) | 0.0107 | 0.0103 (0.0041) | 0.0124 |
| `SCZ15_Hear_voices15` | 0.0182 (0.0114) | 0.110 | 0.0177 (0.0114) | 0.120 |
| `SCZ15_Unrealistic_abilities15` | 0.0171 (0.0124) | 0.168 | 0.0171 (0.0124) | 0.167 |
| `SCZ15_Special_messages15` | 0.0149 (0.0109) | 0.172 | 0.0147 (0.0109) | 0.179 |
| `SCZ15_Rage_attacks15` | 0.0085 (0.0066) | 0.201 | 0.0086 (0.0066) | 0.193 |
| `SCZ15_Seen_hallucinations15` | 0.0080 (0.0069) | 0.245 | 0.0081 (0.0069) | 0.235 |
| `SCZ15_Read_others_minds15` | -0.0042 (0.0042) | 0.319 | -0.0042 (0.0042) | 0.310 |
| `SCZ15_Others_Read_thoughts15` | 0.0047 (0.0048) | 0.325 | 0.0044 (0.0048) | 0.361 |
| `SCZ15_Racing_thoughts15` | 0.0039 (0.0047) | 0.407 | 0.0041 (0.0047) | 0.381 |
| `SCZ15_PROD_seen_hallucinations9` | -0.0094 (0.0161) | 0.557 | -0.0090 (0.0161) | 0.576 |
| `SCZ15_Spied15` | -0.0026 (0.0045) | 0.567 | -0.0024 (0.0045) | 0.599 |
| `SCZ15_Special_powers15` | -0.0052 (0.0099) | 0.599 | -0.0053 (0.0099) | 0.593 |
| `SCZ15_Under_control_special_power15` | -0.0033 (0.0072) | 0.644 | -0.0033 (0.0072) | 0.644 |
| `SCZ15_Sexual_inappropriate15` | 0.0074 (0.0182) | 0.686 | 0.0078 (0.0182) | 0.669 |
| `SCZ15_Talking_fast15` | -0.0013 (0.0043) | 0.771 | -0.0012 (0.0043) | 0.787 |
| `SCZ15_easily_scared15` | 0.0014 (0.0048) | 0.773 | 0.0014 (0.0048) | 0.776 |
| `SCZ15_Not_tired15` | 0.0002 (0.0037) | 0.956 | 0.0001 (0.0037) | 0.985 |
| **ADHD symptoms, age 9** | | | | |
| `ADHD9_often_blurt_out9` | -0.0183 (0.0077) | 0.0184 | -0.0182 (0.0077) | 0.0187 |
| `ADHD9_fail_close_attention9` | -0.0160 (0.0083) | 0.0542 | -0.0159 (0.0083) | 0.0554 |
| `ADHD9_often_interrupt9` | 0.0148 (0.0091) | 0.103 | 0.0146 (0.0091) | 0.107 |
| `ADHD9_often_forgetful9` | 0.0126 (0.0086) | 0.141 | 0.0124 (0.0086) | 0.147 |
| `ADHD9_often_motor9` | -0.0166 (0.0123) | 0.177 | -0.0162 (0.0123) | 0.186 |
| `ADHD9_difficulty_with_attention9` | 0.0099 (0.0093) | 0.285 | 0.0099 (0.0093) | 0.287 |
| `ADHD9_often_run9` | -0.0083 (0.0090) | 0.354 | -0.0083 (0.0090) | 0.357 |
| `ADHD9_often_move9` | -0.0080 (0.0102) | 0.430 | -0.0081 (0.0102) | 0.428 |
| `ADHD9_not_listening9` | 0.0064 (0.0083) | 0.439 | 0.0066 (0.0083) | 0.429 |
| `ADHD9_avoid_mental_effort_tasks9` | -0.0062 (0.0084) | 0.463 | -0.0060 (0.0084) | 0.475 |
| `ADHD9_difficulty_calm9` | -0.0096 (0.0147) | 0.511 | -0.0092 (0.0147) | 0.528 |
| `ADHD9_difficulty_organizing_tasks9` | -0.0074 (0.0119) | 0.534 | -0.0077 (0.0119) | 0.521 |
| `ADHD9_difficulty_following_instructions9` | 0.0069 (0.0120) | 0.563 | 0.0068 (0.0120) | 0.571 |
| `ADHD9_easily_bored9` | -0.0038 (0.0070) | 0.583 | -0.0040 (0.0070) | 0.565 |
| `ADHD9_easily_distracted9` | -0.0039 (0.0076) | 0.604 | -0.0039 (0.0076) | 0.604 |
| `ADHD9_difficulty_hold_still9` | 0.0043 (0.0084) | 0.609 | 0.0044 (0.0084) | 0.605 |
| `ADHD9_talk_excess9` | 0.0031 (0.0061) | 0.611 | 0.0028 (0.0061) | 0.642 |
| `ADHD9_often_lose_things9` | -0.0014 (0.0073) | 0.847 | -0.0014 (0.0073) | 0.846 |
| `ADHD9_difficulty_waiting9` | -0.0001 (0.0081) | 0.993 | 0.0001 (0.0081) | 0.994 |
| **Autism-related traits, age 9** | | | | |
| `ASD9_difficulty_converse9` | -0.0491 (0.0212) | 0.0208 | -0.0485 (0.0212) | 0.0222 |
| `ASD9_dislike_change9` | 0.0167 (0.0085) | 0.0492 | 0.0166 (0.0085) | 0.0504 |
| `ASD9_difficulty_socialize9` | 0.0324 (0.0169) | 0.0554 | 0.0316 (0.0169) | 0.062 |
| `ASD9_absorbed_details9` | 0.0163 (0.0102) | 0.109 | 0.0161 (0.0102) | 0.112 |
| `ASD9_difficulty_pretend_play9` | -0.0201 (0.0164) | 0.221 | -0.0205 (0.0164) | 0.213 |
| `ASD9_talk_too_high_low9` | 0.0107 (0.0087) | 0.222 | 0.0111 (0.0087) | 0.203 |
| `ASD9_repeat_words9` | -0.0154 (0.0126) | 0.222 | -0.0153 (0.0126) | 0.227 |
| `ASD9_absorbed_problems9` | -0.0192 (0.0167) | 0.252 | -0.0191 (0.0167) | 0.254 |
| `ASD9_easily_influenced9` | 0.0060 (0.0064) | 0.350 | 0.0059 (0.0064) | 0.362 |
| `ASD9_difficulty_on_track9` | 0.0099 (0.0106) | 0.351 | 0.0101 (0.0106) | 0.339 |
| `ASD9_absorbed_own9` | 0.0087 (0.0101) | 0.391 | 0.0088 (0.0101) | 0.385 |
| `ASD9_delay_language9` | 0.0063 (0.0074) | 0.391 | 0.0061 (0.0074) | 0.405 |
| `ASD9_own_terms9` | 0.0084 (0.0122) | 0.492 | 0.0082 (0.0122) | 0.503 |
| `ASD9_difficulty_express9` | -0.0160 (0.0242) | 0.510 | -0.0156 (0.0242) | 0.520 |
| `ASD9_difficulty_expect_behavior9` | 0.0072 (0.0165) | 0.661 | 0.0075 (0.0165) | 0.647 |
| `ASD9_strange_movements9` | -0.0020 (0.0118) | 0.866 | -0.0017 (0.0118) | 0.886 |
| `ASD9_uninterested_sharing9` | -0.0013 (0.0222) | 0.952 | -0.0014 (0.0222) | 0.950 |
| **Adverse experiences, age 15** | | | | |
| `ACE15_emotional_bullying15` | 0.0085 (0.0061) | 0.164 | 0.0084 (0.0061) | 0.167 |
| `ACE15_bullying_time15` | 0.0055 (0.0040) | 0.166 | 0.0054 (0.0040) | 0.172 |
| `ACE15_bullying_by_num15` | -0.0051 (0.0058) | 0.380 | -0.0049 (0.0058) | 0.390 |
| `ACE15_rumours_bullying15` | 0.0061 (0.0073) | 0.406 | 0.0062 (0.0073) | 0.394 |
| `ACE15_tease_bullying15` | 0.0047 (0.0068) | 0.493 | 0.0047 (0.0068) | 0.486 |
| `ACE15_bullied_often15` | -0.0040 (0.0070) | 0.569 | -0.0038 (0.0070) | 0.593 |
| `ACE15_other_bullying15` | -0.0031 (0.0064) | 0.635 | -0.0033 (0.0064) | 0.613 |
| **Adverse experiences, age 18** | | | | |
| `ACE18_emotional_abuse18` | -0.0752 (0.0066) | < 0.001 | -0.0754 (0.0066) | < 0.001 |
| `ACE18_hate_crime18` | -0.0544 (0.0175) | 0.00197 | -0.0542 (0.0175) | 0.00203 |
| `ACE18_other_abuse18` | 0.2684 (0.3643) | 0.461 | 0.2695 (0.3641) | 0.459 |
| `ACE18_witness_crime18` | -0.0020 (0.0103) | 0.847 | -0.0017 (0.0103) | 0.871 |
| **Substance use, age 18** | | | | |
| `SUD18_cigarettes18` | 0.0107 (0.0021) | < 0.001 | 0.0107 (0.0021) | < 0.001 |
| `SUD18_drugs_often18` | 0.0216 (0.0083) | 0.00912 | 0.0216 (0.0083) | 0.00907 |
| `SUD18_alcohol_often18` | -0.0038 (0.0041) | 0.359 | -0.0037 (0.0041) | 0.372 |
| `SUD18_snuff18` | 0.0008 (0.0017) | 0.660 | 0.0007 (0.0017) | 0.667 |
| **Parental socioeconomic indicators** | | | | |
| `SES_birth_country_father` | -0.0187 (0.0071) | 0.00832 | -0.0186 (0.0071) | 0.00837 |
| `SES_education_mother` | 0.0035 (0.0034) | 0.306 | 0.0036 (0.0034) | 0.287 |
| `SES_birth_country_mother` | 0.0072 (0.0080) | 0.370 | 0.0073 (0.0080) | 0.358 |
| `SES_education_father` | 0.0012 (0.0029) | 0.674 | 0.0011 (0.0029) | 0.692 |
| **Sex** | | | | |
| `SEX` | 0.0417 (0.0068) | < 0.001 | 0.0416 (0.0068) | < 0.001 |
| **Genotyping batch × principal component** | | | | |
| `batch_PSYC_x_PC2` | -0.3105 (0.2287) | 0.175 | -0.2955 (0.2286) | 0.196 |
| `batch_GSA3_x_PC1` | -0.4600 (0.3846) | 0.232 | -0.4395 (0.3848) | 0.254 |
| `batch_PSYC_x_PC1` | 0.2577 (0.2392) | 0.281 | 0.2593 (0.2390) | 0.278 |
| `batch_GSA3_x_PC2` | 0.1085 (0.6516) | 0.868 | 0.0929 (0.6513) | 0.887 |
| **Interactions with SUD15** | | | | |
| SUD15 score × ACE15 score | -0.0039 (0.0030) | 0.192 | -0.0039 (0.0030) | 0.199 |
| SUD15 score × ACE18 score | -0.0025 (0.0028) | 0.373 | -0.0025 (0.0028) | 0.369 |
| SUD15 score × ADHD9 score | 0.0013 (0.0028) | 0.647 | 0.0013 (0.0028) | 0.645 |
| SUD15 score × ASD9 score | 0.0046 (0.0032) | 0.146 | 0.0046 (0.0032) | 0.143 |
| SUD15 score × SCZ15 score | -0.0064 (0.0027) | 0.0169 | -0.0063 (0.0027) | 0.0182 |
| SUD15 score × SES score | 0.0016 (0.0032) | 0.625 | 0.0016 (0.0032) | 0.611 |
| SUD15 score × SEX | 0.0043 (0.0068) | 0.527 | 0.0041 (0.0068) | 0.546 |
| SUD15 score × SUD18 score | 0.0014 (0.0029) | 0.626 | 0.0014 (0.0029) | 0.623 |
| SUD15 score × PRS_Amygdala_excitatory (z) | 0.0026 (0.0046) | 0.573 | 0.0026 (0.0046) | 0.577 |
| SUD15 score × PRS_CGE_interneuron (z) | 0.0012 (0.0042) | 0.781 | 0.0013 (0.0042) | 0.761 |
| SUD15 score × PRS_Cerebellar_inhibitory (z) | -0.0055 (0.0037) | 0.140 | -0.0056 (0.0037) | 0.132 |
| SUD15 score × PRS_Deep_layer_corticothalamic (z) | 0.0071 (0.0041) | 0.082 | 0.0072 (0.0041) | 0.0746 |
| SUD15 score × PRS_Deep_layer_intratelencepha (z) | 0.0056 (0.0048) | 0.243 | 0.0057 (0.0048) | 0.240 |
| SUD15 score × PRS_Eccentric_medium_spiny_neu (z) | -0.0016 (0.0043) | 0.708 | -0.0016 (0.0043) | 0.708 |
| SUD15 score × PRS_Hippocampal_CA1_3 (z) | 0.0068 (0.0048) | 0.161 | 0.0065 (0.0048) | 0.183 |
| SUD15 score × PRS_Hippocampal_CA4 (z) | -0.0081 (0.0043) | 0.058 | -0.0078 (0.0043) | 0.0663 |
| SUD15 score × PRS_Hippocampal_dentate_gyrus (z) | 0.0010 (0.0038) | 0.804 | 0.0008 (0.0038) | 0.831 |
| SUD15 score × PRS_LAMP5_LHX6_and_Chandelier (z) | 0.0035 (0.0048) | 0.468 | 0.0035 (0.0048) | 0.474 |
| SUD15 score × PRS_MGE_interneuron (z) | -0.0016 (0.0043) | 0.716 | -0.0017 (0.0043) | 0.692 |
| SUD15 score × PRS_Medium_spiny_neuron (z) | -0.0036 (0.0044) | 0.420 | -0.0035 (0.0044) | 0.424 |
| SUD15 score × PRS_Miscellaneous (z) | -0.0002 (0.0043) | 0.960 | -0.0003 (0.0043) | 0.950 |
| SUD15 score × PRS_Thalamic_excitatory (z) | 0.0021 (0.0032) | 0.515 | 0.0021 (0.0032) | 0.513 |
| SUD15 score × PRS_Upper_layer_intratelenceph (z) | -0.0004 (0.0042) | 0.931 | -0.0002 (0.0042) | 0.953 |
| SUD15 score × PRS_all (z) | 0.0018 (0.0038) | 0.631 | 0.0017 (0.0038) | 0.649 |

### Table S3c. PRS smooth terms of the additive mixed models

| Smooth | PLE-HM edf | PLE-HM F | PLE-HM p | DEP edf | DEP F | DEP p |
|---|---|---|---|---|---|---|
| `s(PRS_Amygdala_excitatory)` | 1.00 | 1.75 | 0.186 | 1.00 | 0.25 | 0.618 |
| `s(PRS_Cerebellar_inhibitory)` | 2.44 | 0.83 | 0.450 | 1.72 | 1.03 | 0.384 |
| `s(PRS_CGE_interneuron)` | 1.87 | 1.25 | 0.323 | 1.92 | 0.74 | 0.608 |
| `s(PRS_Deep_layer_corticothalamic)` | 1.00 | 0.20 | 0.655 | 1.00 | 0.53 | 0.468 |
| `s(PRS_Deep_layer_intratelencepha)` | 1.00 | 1.27 | 0.261 | 1.00 | 0.62 | 0.433 |
| `s(PRS_Eccentric_medium_spiny_neu)` | 2.17 | 1.03 | 0.337 | 1.13 | 0.99 | 0.385 |
| `s(PRS_Hippocampal_CA1_3)` | 2.54 | 2.56 | 0.0497 | 1.00 | 0.00 | 0.980 |
| `s(PRS_Hippocampal_CA4)` | 1.00 | 0.30 | 0.582 | 1.00 | 1.59 | 0.207 |
| `s(PRS_Hippocampal_dentate_gyrus)` | 1.00 | 0.02 | 0.899 | 1.64 | 0.49 | 0.606 |
| `s(PRS_LAMP5_LHX6_and_Chandelier)` | 1.28 | 0.14 | 0.890 | 1.00 | 0.14 | 0.707 |
| `s(PRS_Medium_spiny_neuron)` | 1.00 | 0.07 | 0.796 | 1.00 | 0.80 | 0.371 |
| `s(PRS_MGE_interneuron)` | 1.00 | 0.38 | 0.536 | 1.00 | 0.37 | 0.542 |
| `s(PRS_Miscellaneous)` | 1.00 | 3.87 | 0.0493 | 1.00 | 0.07 | 0.794 |
| `s(PRS_Thalamic_excitatory)` | 1.00 | 0.15 | 0.698 | 1.32 | 0.16 | 0.865 |
| `s(PRS_Upper_layer_intratelenceph)` | 1.00 | 0.23 | 0.632 | 1.00 | 0.01 | 0.927 |
| `s(PRS_all)` | 1.00 | 13.10 | < 0.001 | 1.00 | 6.28 | 0.0122 |
| Twin-pair random intercept (`s(cmpair)`, random-effect basis) | 353.96 | 0.22 | < 0.001 | 460.89 | 0.31 | < 0.001 |

edf, effective degrees of freedom (1 = linear). <!-- Results text checked against this table 2026-10-26: 11 of 16 smooths edf < 1.1 per outcome; max 2.54 / 1.92; p < 0.05 for all-cell, CA1–3 (0.0497) and miscellaneous (0.0493) for PLE-HM, all-cell for DEP; interactions with p < 0.05 match 04_results.md -->

## Table S4. Predictors by group

Variable names as in the analysis data. Items combined during screening (see Methods): `SUD15_OtherDrugs15` (11 illicit drugs), `SUD15_Alcohol15` (beer and other alcohol), `ACE15_other_bullying15` (six types of bullying) and `ACE18_other_abuse18` (five types of abuse or violence). `SCZ15_PROD_seen_hallucinations9` was measured at age 15; the suffix 9 in its name is a labelling error in the analysis data. <!-- D-24 resolved 2026-10-27: author confirmed no psychotic symptoms were measured at age 9 -->

| Group | n | Variables |
|---|---|---|
| Substance use, age 15 (anchor) (SUD15) | 6 | `SUD15_Alcohol15`, `SUD15_Cannabis15`, `SUD15_Cigarettes15`, `SUD15_OtherDrugs15`, `SUD15_Painkillers_opioids15`, `SUD15_Snuff15` |
| Polygenic risk scores (PRS) | 16 | `PRS_Amygdala_excitatory`, `PRS_CGE_interneuron`, `PRS_Cerebellar_inhibitory`, `PRS_Deep_layer_corticothalamic`, `PRS_Deep_layer_intratelencepha`, `PRS_Eccentric_medium_spiny_neu`, `PRS_Hippocampal_CA1_3`, `PRS_Hippocampal_CA4`, `PRS_Hippocampal_dentate_gyrus`, `PRS_LAMP5_LHX6_and_Chandelier`, `PRS_MGE_interneuron`, `PRS_Medium_spiny_neuron`, `PRS_Miscellaneous`, `PRS_Thalamic_excitatory`, `PRS_Upper_layer_intratelenceph`, `PRS_all` |
| Symptoms, age 15 (SCZ15) | 23 | `SCZ15_Extreme_excitement15`, `SCZ15_Hear_voices15`, `SCZ15_Irritable15`, `SCZ15_Not_tired15`, `SCZ15_Others_Read_thoughts15`, `SCZ15_PROD_seen_hallucinations9`, `SCZ15_Racing_thoughts15`, `SCZ15_Rage_attacks15`, `SCZ15_Read_others_minds15`, `SCZ15_Seen_hallucinations15`, `SCZ15_Sexual_inappropriate15`, `SCZ15_Special_messages15`, `SCZ15_Special_powers15`, `SCZ15_Spied15`, `SCZ15_Talking_fast15`, `SCZ15_Too_much_energy15`, `SCZ15_Under_control_special_power15`, `SCZ15_Unrealistic_abilities15`, `SCZ15_easily_scared15`, `SCZ15_headaches15`, `SCZ15_lose_confidence15`, `SCZ15_unhappy15`, `SCZ15_worry15` |
| ADHD symptoms, age 9 (ADHD9) | 19 | `ADHD9_avoid_mental_effort_tasks9`, `ADHD9_difficulty_calm9`, `ADHD9_difficulty_following_instructions9`, `ADHD9_difficulty_hold_still9`, `ADHD9_difficulty_organizing_tasks9`, `ADHD9_difficulty_waiting9`, `ADHD9_difficulty_with_attention9`, `ADHD9_easily_bored9`, `ADHD9_easily_distracted9`, `ADHD9_fail_close_attention9`, `ADHD9_not_listening9`, `ADHD9_often_blurt_out9`, `ADHD9_often_forgetful9`, `ADHD9_often_interrupt9`, `ADHD9_often_lose_things9`, `ADHD9_often_motor9`, `ADHD9_often_move9`, `ADHD9_often_run9`, `ADHD9_talk_excess9` |
| Autism-related traits, age 9 (ASD9) | 17 | `ASD9_absorbed_details9`, `ASD9_absorbed_own9`, `ASD9_absorbed_problems9`, `ASD9_delay_language9`, `ASD9_difficulty_converse9`, `ASD9_difficulty_expect_behavior9`, `ASD9_difficulty_express9`, `ASD9_difficulty_on_track9`, `ASD9_difficulty_pretend_play9`, `ASD9_difficulty_socialize9`, `ASD9_dislike_change9`, `ASD9_easily_influenced9`, `ASD9_own_terms9`, `ASD9_repeat_words9`, `ASD9_strange_movements9`, `ASD9_talk_too_high_low9`, `ASD9_uninterested_sharing9` |
| Adverse experiences, age 15 (ACE15) | 7 | `ACE15_bullied_often15`, `ACE15_bullying_by_num15`, `ACE15_bullying_time15`, `ACE15_emotional_bullying15`, `ACE15_other_bullying15`, `ACE15_rumours_bullying15`, `ACE15_tease_bullying15` |
| Adverse experiences, age 18 (ACE18) | 4 | `ACE18_emotional_abuse18`, `ACE18_hate_crime18`, `ACE18_other_abuse18`, `ACE18_witness_crime18` |
| Substance use, age 18 (SUD18) | 4 | `SUD18_alcohol_often18`, `SUD18_cigarettes18`, `SUD18_drugs_often18`, `SUD18_snuff18` |
| Parental socioeconomic indicators (SES) | 4 | `SES_birth_country_father`, `SES_birth_country_mother`, `SES_education_father`, `SES_education_mother` |
| Sex (SEX) | 1 | `SEX` |
| Genotyping batch × principal component (batch) | 4 | `batch_GSA3_x_PC1`, `batch_GSA3_x_PC2`, `batch_PSYC_x_PC1`, `batch_PSYC_x_PC2` |
| **Total** | **105** | |

## Table S5. Selection of the fusion depths

After the hyperparameter search (Table S1), the depth of each fusion block was varied from 1 to 5 while the other blocks were kept at the uniform depth from the search (3 for PLE-HM, 2 for DEP). All runs used the seed-42 training set, with the same 80/20 training/validation split as the search. The depth with the lowest validation RMSE was taken for each block. The resulting per-block configuration and the uniform configuration were then compared on the seed-42 test set. Per-block depths were adopted for PLE-HM, where they had the better test performance, and the uniform depth was kept for DEP. This choice used test-set results (see Methods). Bold: lowest validation RMSE per block.

**PLE-HM** (other blocks at the uniform depth 3; validation RMSE × 100)

| Fusion block | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Selected |
|---|---|---|---|---|---|---|
| PRS | 9.030 | 9.081 | 8.993 | 8.993 | **8.887** | 5 |
| SCZ15 | 8.930 | **8.875** | 8.993 | 8.906 | 8.971 | 2 |
| ADHD9 | **8.899** | 9.140 | 8.993 | 8.995 | 8.942 | 1 |
| ASD9 | 8.956 | 8.911 | 8.993 | 9.048 | **8.888** | 5 |
| ACE15 | 9.014 | 9.066 | 8.993 | **8.964** | 9.005 | 4 |
| ACE18 | **8.766** | 8.891 | 8.993 | 8.975 | 9.109 | 1 |
| SUD18 | **8.767** | 8.890 | 8.993 | 8.976 | 9.119 | 1 |
| SES | **8.766** | 8.893 | 8.993 | 8.982 | 9.115 | 1 |
| SEX | 8.953 | 8.929 | 8.993 | 9.050 | **8.870** | 5 |

| Configuration | Depths | Validation RMSE | Validation R² | Test RMSE (seed 42) | Test R² (seed 42) |
|---|---|---|---|---|---|
| Per-block | [5, 2, 1, 5, 4, 1, 1, 1, 5] | 0.0898 | 0.204 | 0.0863 | 0.262 |
| Uniform | 3 | 0.0899 | 0.201 | 0.0891 | 0.213 |

**DEP** (other blocks at the uniform depth 2; validation RMSE × 100)

| Fusion block | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Selected |
|---|---|---|---|---|---|---|
| PRS | **15.004** | 15.044 | 15.223 | 15.354 | 15.042 | 1 |
| SCZ15 | 15.312 | 15.044 | 15.333 | **14.961** | 15.308 | 4 |
| ADHD9 | 15.478 | **15.044** | 15.339 | 15.358 | 15.070 | 2 |
| ASD9 | **15.024** | 15.044 | 15.042 | 15.436 | 15.186 | 1 |
| ACE15 | **14.984** | 15.044 | 15.450 | 15.098 | 15.012 | 1 |
| ACE18 | 15.327 | **15.044** | 15.191 | 15.152 | 15.056 | 2 |
| SUD18 | 15.318 | **15.044** | 15.186 | 15.157 | 15.052 | 2 |
| SES | 15.322 | **15.044** | 15.189 | 15.160 | 15.048 | 2 |
| SEX | 15.036 | 15.044 | **14.942** | 15.512 | 15.186 | 3 |

| Configuration | Depths | Validation RMSE | Validation R² | Test RMSE (seed 42) | Test R² (seed 42) |
|---|---|---|---|---|---|
| Per-block | [1, 4, 2, 1, 1, 2, 2, 2, 3] | 0.1536 | 0.213 | 0.1551 | 0.201 |
| Uniform | 2 | 0.1504 | 0.245 | 0.1531 | 0.221 |

## Table S6. Test-set performance on all metrics

Mean ± SD over the five test sets, with all predictors. R², coefficient of determination (1 − SSE/SST); RMSE, root-mean-squared error; MAE, mean absolute error; Spearman ρ, rank correlation between predicted and observed scores. R² and RMSE are also given in Table 3 of the main text.

<!-- src: Models/Model Comparison/prediction_metrics_summary.csv (predictor_set = full); moved from main-text Table 3 on 2026-10-29 -->

| Model | R² | RMSE | MAE | Spearman ρ |
|---|---|---|---|---|
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
