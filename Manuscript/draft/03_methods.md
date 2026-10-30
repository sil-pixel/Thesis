<!--
STATUS 2026-10-23: Methods for BMC Psychiatry (replaces 03_materials_and_methods.md, written for AIIM).
- Describes the analysis as it is now run: shared splits, updated GLMM/GAMM (E-01, E-02 completed 2026-10-23),
  prediction-level evaluation, age-15 sensitivity analysis and attribution analyses (scripts written 2026-10-23;
  outputs pending from the author's runs).
- DCMFNet configuration selection is described as conducted, with the test-data involvement disclosed (D-04, D-09, D-21;
  E-03 not run). The ML benchmark is described as conducted.
- BMC allows topical subheadings in Methods. Section numbers are not used (BMC style).
- Outcome labels follow D-03 (descriptive names) pending author confirmation (M-01).
-->

# Methods

## Study population

We used data from the Child and Adolescent Twin Study in Sweden (CATSS), an ongoing population-based cohort of all twins born in Sweden since July 1992, identified through the Swedish Twin Registry. Parents are interviewed by telephone about their children's health around the twins' 9th or 12th birthday, and twins and parents complete questionnaires at ages 15 and 18 [@R-CATSS]. <!-- src: Thesis Report §3.1 p.7; Anckarsäter et al. 2011 abstract (PMID 22506305, read 2026-10-28): all twins born in Sweden since 1 July 1992; parental interviews at the 9th or 12th birthday since 2004; questionnaires to twins and parents at 15 and 18. Response rates not stated here (80% for the age-9/12 interviews up to 2010 only). -->
We used the neurodevelopmental measures from the age-9 parent interview, self-reports at ages 15 and 18, and genotype data available for a subset of participants. <!-- informant verified 2026-10-28 from Anckarsäter et al. 2011 (parental telephone interview at 9/12) --> The analysis data were prepared by the research team (files last updated in November 2023) and analysed on a secure server at Karolinska Institutet between November 2025 and October 2026. <!-- author, 2026-10-28: original data files last modified 2023-11-23; access started by 2025-11-06 at the latest; access ends 2026-10-31 --> `[AUTHOR INPUT: years in which the age-15 and age-18 questionnaires used here were collected, and birth years covered; from the research team or CATSS documentation]`

Individuals were eligible if they had substance-use data at age 15 and outcome data at age 18. The analysis sample comprised all eligible individuals with complete data on every predictor and both outcomes: 4,033 individuals from 2,690 twin pairs. The sample size was set by the available data; no formal sample size calculation was made. <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1-3 (shape (4033, 125); 2,017 + 673 = 2,690 cmpair IDs) -->
Of the 13,146 CATSS twins in the study data, 9,261 had data on substance use at age 15 and on both outcomes at age 18 and were eligible. Participant flow is shown in Fig. 1. <!-- src: thesis Figs 3.1 and 3.3 (13,146; 3,885 excluded for missing exposure or outcome), author confirmed 2026-10-28 (W-02); 9,261 = rows of catss_final_data.csv (analysis_sample_flow.csv). Thesis Fig. 3.1 reports 9,160 genotyped among the 13,146; 6,643 of the 9,261 eligible have PRS (analysis_sample_flow.csv), so about 2,500 genotyped twins lacked substance use at 15 or an outcome. The two numbers have different denominators and do not conflict. -->
We compared the analysis sample with excluded eligible participants on sex, outcomes, substance use at age 15 and summary scores of each predictor group, using standardised mean differences (Table 1). Counts below five are not reported. <!-- src: Data Preprocessing/Data Visualisation/analysis_sample_descriptives.R -->

## Outcomes

Two continuous outcomes were derived from self-report items at age 18. <!-- src: Data Preprocessing/Data Analysis/catss_analysis_final.html, CATSS18 section (ple_var_list18, mdq_cmrs_list18, cesd_var_list18 and their response labels), read 2026-10-28; outcome item lists in rename_catss_with_mapping.py:73-81 -->

- **Psychotic-like and hypomanic experiences (PLE-HM):** sum of 24 items, divided by 46: 7 items on psychotic-like experiences (for example feeling spied upon or seeing things that others cannot) from the Adolescent Psychotic-Like Symptom Screener [@R-LAURENS; @R-APSS], the 13 symptom items of the Mood Disorder Questionnaire [@R-MDQ] and 4 items from the Child Mania Rating Scale [@R-CMRS] (unrealistic abilities, talking fast, sexually inappropriate behaviour and hearing voices). <!-- src: catss_analysis_final.html (ple_var_list18: spied, read_thoughts, Special_messages, special_powers, under_control_special, read_others_mind, seen_hallucinations; mdq_cmrs_list18: 13 MDQ items + unrealistic_abilities, talk_fast, sexual_inappropriate, hear_voices) --> Psychotic-like and Child Mania Rating Scale items were scored from 0 to 3 and Mood Disorder Questionnaire items as no (0) or yes (1), so 46 is the maximum score. <!-- author confirmed 2026-10-28: 7×3 + 13×1 + 4×3 = 46; instruments confirmed (M-01) -->
- **Depressive symptoms (DEP):** sum of the 11 items of the short form of the Center for Epidemiologic Studies Depression Scale (CES-D) [@R-CESD; @R-CESD11], each scored from 0 ("not at all or almost not at all") to 3 ("all the time or almost all the time"), divided by 33, the maximum score. <!-- src: catss_analysis_final.html (cesd_var_list18, cesd_labels_list18 0–3); the 11 items match the Iowa short form (Kohout 1993) item for item --> The two positively worded items (being happy and enjoying life) were reverse-coded. <!-- author confirmed 2026-10-28 -->

Both scores lie between 0 and 1 and were analysed as continuous outcomes without clinical thresholds.

## Predictors

The 105 predictors were organised into 11 groups defined before analysis (number of variables in parentheses; full list in Additional file 1: Table S4): <!-- src: best_model_training.log:2 ; Models/DCMFNet/Method/legacy/train.py:74-87 -->

1. Substance use at age 15 (SUD15; 6): frequency of cigarette, snus, alcohol, cannabis and non-medical painkiller use, and the maximum across other drug items. <!-- src: Data Preprocessing/Feature Selection/feature_selection.py:76-97 -->
2. Polygenic risk scores (PRS; 16): a genome-wide schizophrenia PRS and cell-type-specific PRS restricted to genes specifically expressed in brain cell types of a human brain cell atlas [@R-SILETTI; @R-YAO2025], computed with PRS-CS [@R-PRSCS] (see *Genotype data and polygenic scores*).
3. Psychotic-like, hypomanic and emotional symptoms at age 15 (SCZ15; 23).
4. ADHD symptoms at age 9 (ADHD9; 19).
5. Autism-related traits at age 9 (ASD9; 17). Groups 4 and 5 are the ADHD and autism-spectrum modules of the Autism–Tics, ADHD and other Comorbidities inventory (A-TAC), a parent telephone interview [@R-ATAC]. <!-- author confirmed 2026-10-28 that the ADHD9 and ASD9 items are the A-TAC ADHD and autism modules; Larson et al. 2010 verified via Crossref/Europe PMC (PMC2823676) -->
6. Adverse experiences at age 15 (ACE15; 7), mainly bullying victimisation.
7. Adverse experiences at age 18 (ACE18; 4): emotional abuse, hate crime, witnessing crime and other abuse or violence. <!-- src: feature_selection.py:111-130 -->
8. Substance use at age 18 (SUD18; 4).
9. Parental socioeconomic indicators (SES; 4): parental education and country of birth.
10. Sex (1), as reported by parents in the opening questions of the age-9/12 interview. <!-- author, 2026-10-28 (M-06) -->
11. Genotyping batch × genetic principal component terms (4): products of batch indicators and the first two principal components, to adjust for population structure and batch effects. <!-- src: rename_catss_with_mapping.py (add_batch_pc_interactions, pc_cols=['PC1','PC2']) -->

ACE18 and SUD18 were measured at the same assessment as the outcomes, so models that include them combine prospective and concurrent information. We therefore repeated all analyses without these groups (see *Sensitivity analyses*).

**Genotype data and polygenic scores.** Genotype data for CATSS participants were obtained from the Swedish Twin Registry. Of the 13,146 twins in the study data, 9,271 had genotype records linked to their phenotype data, and 9,160 remained after genotype quality control. <!-- src: thesis §3.2 and Fig. 3.1 (13,146; 3,875 not linked; 111 excluded at QC; 9,160) --> Genotyping was done in three batches. <!-- src: batch dummies batch_GSA3 and batch_PSYC with a third reference level (rename_catss_with_mapping.py add_batch_pc_interactions, drop_first = True; Additional file 1: Table S4) --> `[VERIFY: genotyping arrays of the three batches (the batch labels GSA3 and PSYC suggest the Illumina Global Screening Array and PsychArray); sample and SNP quality-control criteria; imputation reference panel and software]` <!-- W-03 --> Polygenic scores were computed with PRS-CS [@R-PRSCS], which estimates SNP weights from genome-wide association study (GWAS) summary statistics under a continuous shrinkage prior, using schizophrenia GWAS summary statistics `[VERIFY: GWAS used, e.g. PGC3 [@R-PGC3], and whether a version without Swedish samples was used; LD reference panel; global shrinkage parameter (learned or fixed)]` <!-- W-03 -->. The genome-wide score used all SNPs. Each of the 15 cell-type-specific scores used only SNPs mapped to the 10% of genes most specifically expressed in one cell type of the human brain cell atlas [@R-SILETTI]; these cell types showed enrichment of schizophrenia SNP-heritability in earlier work [@R-YAO2025]. <!-- src: thesis §2.1 p.4 ("top 10% most specifically expressed genes for each cell type enriched for schizophrenia-associated genetic variants") and §3.2.1; pipeline adapted from Wang H, Master's thesis, Karolinska Institutet 2025 (unpublished, not cited) --> `[VERIFY: how SNPs were mapped to genes (window around each gene) and whether "all SNPs" is right for the genome-wide score]` <!-- W-03 --> Genetic principal components were computed to capture population structure, and individuals of non-European ancestry were identified as outliers `[VERIFY: were they excluded?]` <!-- W-03 -->. <!-- src: thesis §3.2.1 p.8 --> The models adjusted for population structure and batch through the products of the batch indicators with the first two principal components (group 11).

**Variable screening.** The 236 candidate variables were reduced in four steps, guided by pairwise Pearson correlations and by SHAP values from LightGBM models fitted to each outcome [@R-LGBM; @R-SHAP]. <!-- src: Thesis §3.4 p.10-11 and Fig. 3.3; Data Preprocessing/Feature Selection/feature_selection.py; Data Analysis/Feature Importance/feature_importance.py -->
1. Variables with more than 90% missing values were removed. <!-- src: feature_selection.py get_high_missingness_columns -->
2. Groups of highly correlated items were combined into one variable by taking the highest response: 11 illicit-drug items into "other drugs" and beer into alcohol at age 15; six types of bullying at age 15; and five types of abuse or violence at age 18. <!-- src: feature_selection.py group_columns (Drugs15, ACE15, ACE18) -->
3. Parent-reported symptom items at age 18 were removed. They were highly correlated with the self-reports and had more missing values, so the outcomes are based on self-reports. <!-- src: feature_selection.py group_columns (Symptoms18) -->
4. A further 41 variables with 50–90% missing values that could not be combined with correlated items were removed. Most were parent-reported items at age 15 or adverse experiences at age 9. <!-- src: feature_selection.py drop_mid_missing_cols (41 variables: 30 parent-reported at 15, 8 at age 9, 3 at age 18) -->

The LightGBM models used the outcomes and were fitted on random splits of the full sample (seeds 41–45) that do not correspond to the evaluation splits. <!-- src: feature_importance.py:31-45, 104-112; D-09, E-04 --> Although the reasons recorded in the code for each removal or grouping are missingness or correlation, the SHAP values informed these decisions, so we treat the screening as outcome-informed. Because all models received the same 105 predictors, this step could raise the performance of all models but does not favour any one of them. <!-- E-04 resolved 2026-10-26: author confirmed screening was inferred from Pearson correlations and LightGBM SHAP values (thesis §3.4) -->Responses of "don't know" or "don't want to answer" were set to missing. <!-- src: feature_selection.py:136-143 --> No questionnaire item in the final data contained −1 or other negative codes. <!-- src: missing_code_check.csv (no rows); A-03 resolved 2026-10-24 -->

## Data partitioning

The data were split at the twin-pair level, so that co-twins were always in the same partition, into 75% training and 25% test sets. This was repeated with five random seeds (42–46). The same five splits were used for every model: the split identifiers were exported once and read by the R scripts. <!-- src: Data Preprocessing/split_data.py; Models/GLMM/Method/GLMM_updated_split.R (read_split); legacy/train.py:61-68 --> <!-- checked 2026-10-25: Data Preprocessing/Data Analysis/split_summary.csv, matches_dcmfnet_log = True for seeds 42–46 --> Each test set contained 998–1,028 individuals. <!-- src: best_model_training.log:3, 361, 685, 1063, 1414 --> For DCMFNet, a further pair-level 80/20 split of each training set formed a validation set for early stopping and learning-rate scheduling. <!-- src: legacy/train.py:145-152 -->

## Deep cross-modal fusion network

We adapted DCMFNet [@R-DCMF], originally proposed for fusing image and text representations through iterative gated fusion (IGF), to grouped tabular predictors (Fig. 2). Substance use at age 15 is the anchor group. Each of groups 2–10 is fused with the anchor in its own IGF block, and a parallel independent stream receives all raw inputs, including the batch × PC terms. <!-- src: Models/DCMFNet/Method/dcmfnet/model.py:115-205; legacy/train.py:49, 74-87 -->

**Fusion block.** Let $\mathbf{x}_a$ be the anchor features ($d_a = 6$) and $\mathbf{x}_m$ the $d_m$ features of group $m$. Block $m$ has depth $L_m$ and starts from $\mathbf{g}_m^{(0)} = \mathbf{x}_m$. For $l = 1,\dots,L_m$:

$$\mathbf{f}_m^{(l)} = \tanh\big(\tanh(\mathbf{W}_1^{(l)}\mathbf{x}_a + \mathbf{b}_1^{(l)}) \odot \tanh(\mathbf{W}_2^{(l)}\mathbf{g}_m^{(l-1)} + \mathbf{b}_2^{(l)})\big), \tag{1}$$

$$\mathbf{z}_m^{(l)} = \sigma\big(\mathbf{W}_z^{(l)}(\mathbf{f}_m^{(l)} + \mathbf{g}_m^{(l-1)}) + \mathbf{b}_z^{(l)}\big), \tag{2}$$

$$\mathbf{g}_m^{(l)} = \tanh\big(\mathbf{W}_g^{(l)}[\mathbf{f}_m^{(l)};\, \mathbf{f}_m^{(l)} \odot \mathbf{z}_m^{(l)}] + \mathbf{b}_g^{(l)}\big), \tag{3}$$

where $\sigma$ is the logistic function, $\odot$ element-wise multiplication and $[\cdot;\cdot]$ concatenation. Equation 1 is a multiplicative interaction between the anchor and the current representation of group $m$; Equations 2–3 form the gate of the original architecture. The block output concatenates the fused representations from all layers, $\mathbf{u}_m = [\mathbf{f}_m^{(1)};\dots;\mathbf{f}_m^{(L_m)}]$. <!-- src: dcmfnet/model.py:11-47, 106-112; thesis-era commit 6173d10 DCMFNet/model.py:38-101 -->

**Feature gating.** Each block output, the concatenated fused stream, the independent stream and the final representation pass through a feature gate

$$\phi(\mathbf{u}) = \mathbf{u} \odot \sigma\big(\mathbf{V}_2\, \mathrm{Dropout}(\mathrm{ReLU}(\mathbf{V}_1\mathbf{u} + \mathbf{c}_1)) + \mathbf{c}_2\big), \tag{4}$$

with hidden width $\max(\lfloor D/r \rfloor, h_{\min})$ for input dimension $D$. This gate was motivated by squeeze-and-excitation attention [@R-SE], but it is computed for each individual from that individual's own features, without pooling. It is therefore an input-dependent weighting, not a population-level importance measure. <!-- src: dcmfnet/model.py:50-77 — D-06 -->

**Streams and output.** The fused stream is $\mathbf{h}_F = \phi_F([\phi_1(\mathbf{u}_1);\dots;\phi_9(\mathbf{u}_9)])$ and the independent stream is $\mathbf{h}_I = \phi_I(\mathbf{x})$, where $\mathbf{x}$ contains all 105 inputs. The prediction is $\hat{y} = \mathbf{w}^\top \phi_{\mathrm{out}}([\mathbf{h}_F;\mathbf{h}_I]) + b$, with a linear output. <!-- src: dcmfnet/model.py:168-205 --> Because the independent stream feeds gated raw inputs to a linear output layer, the network contains an approximately additive pathway alongside the interaction pathway. Inputs were not standardised. <!-- src: legacy/train.py:74-94 — D-13, E-07 -->

**Loss.** To give more weight to the sparse upper tail of the outcomes, we used a weighted Huber loss. Training outcomes were assigned to $K$ equal-width bins on [0, 1]; each bin received a weight proportional to $1/(n_k + 1)$, normalised to mean 1 and truncated at 20. Within each mini-batch, the weights were multiplied by a focal factor $(|y_i - \hat{y}_i| + 10^{-6})^{\gamma}$, computed without gradient, and renormalised to mean 1. <!-- src: dcmfnet/losses.py:15-76; thesis-era 6173d10:DCMFNet/loss.py:60-73 — D-07 -->

**Training and configuration selection.** Networks were trained with the Adam optimiser [@R-ADAM], weight decay, learning-rate reduction on plateau of the validation RMSE and early stopping with restoration of the best weights. <!-- src: legacy/train.py:371-476 --> Hyperparameters were selected with Optuna [@R-OPTUNA] (50 trials per outcome), minimising validation RMSE within the seed-42 training set (Additional file 1: Table S1, which also describes the tuning history). <!-- src: git c49eec1:DCMFNet/tuning.py:49-71, 204-229 — D-21 --> The search tuned one depth for all fusion blocks (best: 3 for PLE-HM, 2 for DEP). A subsequent sweep varied the depth of each block. For PLE-HM, the per-block depths from this sweep were adopted after per-block and uniform configurations had been compared on the test sets; for DEP, uniform depth 2 was kept (Additional file 1: Table S5). <!-- src: Models/DCMFNet/Analysis/complexity_analysis.py:130-250; Results/Complexity Analysis/Pos_complexity_comparison.csv; git 5c35f0e, 6173d10 — D-04, D-05 --> Several tuning runs were carried out, and the final choice among them was also informed by test-set results. <!-- src: discrepancies.md D-21 (git history c49eec1 … 29 Apr) --> In addition, the seed-42 training data used for tuning overlap the test sets of seeds 43–46. <!-- src: legacy/tuning.py:86-92, 216-217 — D-09 --> All of these steps can only make DCMFNet's test performance optimistic relative to the comparators, which were not tuned. The selected hyperparameters are given in Table 2. <!-- M-02 resolved 2026-10-26: author confirmed Table 2 matches hyperparameters.json (per-block PLE-HM configuration is the primary one) -->

**Table 2.** Selected DCMFNet hyperparameters. <!-- src: Models/DCMFNet/hyperparameters.json -->

| Hyperparameter | PLE-HM model | DEP model |
|---|---|---|
| Learning rate | 1.00 × 10⁻⁴ | 3.22 × 10⁻⁴ |
| Batch size | 4 | 6 |
| Maximum epochs | 23 | 20 |
| Weight decay | 2.95 × 10⁻⁵ | 1.79 × 10⁻⁴ |
| Fusion depth $L_m$ (PRS, SCZ15, ADHD9, ASD9, ACE15, ACE18, SUD18, SES, Sex) | 5, 2, 1, 5, 4, 1, 1, 1, 5 | 2 (all blocks) |
| Dropout | 0.234 | 0.326 |
| Gate reduction ratio $r$ / minimum width $h_{\min}$ | 6 / 13 | 5 / 5 |
| Huber δ / focal γ / bins $K$ | 0.086 / 1.32 / 13 | 0.116 / 2.62 / 10 |
| Scheduler patience / factor | 4 / 0.432 | 2 / 0.549 |
| Early-stopping patience | 5 | 7 |
| Trainable parameters | 105,501 | 71,623 |

<!-- src (parameter counts): recomputed 2026-10-26 by instantiating dcmfnet.model.DCMFNet with the hyperparameters.json values and sizes [6,16,23,19,17,7,4,4,4,1,4] (independent stream = all 105 inputs); 105,501 and 71,623, unchanged; no data used -->

## Mixed-effects comparators

**Generalised linear mixed model (GLMM).** The GLMM included all 105 predictors as linear main effects, a random intercept for twin pair and 24 interaction terms that mirror the network's fusion blocks: the SUD15 score multiplied by the score of each fused questionnaire group (SCZ15, ADHD9, ASD9, ACE15, ACE18, SUD18 and SES; 7 terms), by sex (1 term) and by each standardised PRS (16 terms). Group scores were the mean of the standardised items, standardised again; means and standard deviations were estimated in the training set and applied to the test set. <!-- src: Models/GLMM/Method/GLMM_updated_split.R build_model_spec(), add_interaction_scores():96-111 --> The model was fitted with `lme4::lmer` (Gaussian, identity link). <!-- src: GLMM_updated_split.R:291-313 --> With a Gaussian outcome and identity link, this GLMM is a linear mixed model. <!-- Q-01 decided 2026-10-27: the author uses "GLMM" throughout -->

**Generalised additive mixed model (GAMM).** The GAMM had the same specification, except that each PRS entered as a penalised smooth term and the twin-pair intercept as a random-effect smooth. It was fitted with `mgcv::gam` (Gaussian family, restricted maximum likelihood). Each smooth used mgcv's default thin-plate regression spline basis with dimension 10. <!-- src: Models/GAMM/Method/GAMM_updated_split.R:25-29, 336, 356 (s(PRS) with no bs or k argument, so mgcv defaults: bs = 'tp', k = 10 for one covariate); mgcv 1.9-4 from versions_R.txt (V-04, 2026-10-28) -->

For both models, test-set predictions were made at the population level, without the twin-pair random effect, because test pairs were not part of the training data. <!-- src: GLMM_updated_split.R (predict re.form = NA); GAMM_updated_split.R (exclude = "s(cmpair)") -->

**Machine-learning screening.** Forty regression algorithms from the LazyPredict list [@R-LAZY] were first fitted with scikit-learn default settings on the same splits and inputs (Additional file 1: Table S2). The estimators were called directly, without LazyPredict's own preprocessing, so inputs were not scaled. <!-- src: Models/ML/Method/lazy_predict.py:82-111 (model_class() fitted on raw X; LazyRegressor not used); Models/ML/Results/{Pos,Neg}_benchmark_summary.csv --> Defaults suit some algorithms poorly. For example, the lasso with its default penalty predicted a constant. Ranking the screened models by test performance would also select on test data. The screening was therefore used only to describe how default-setting models perform.

**Tuned machine-learning comparators.** After the screening, we chose three comparators by model class, not by their rank in the screening:
- **linear regression** without penalty, which has nothing to tune;
- **an elastic net**, a penalised linear model that includes the lasso and ridge regression as special cases [@R-ZOU];
- **gradient-boosted trees**, which learn non-linear effects and interactions without a prespecified structure and are a strong benchmark for tabular data [@R-GRIN].

For the elastic net, inputs were standardised, and the penalty strength (100 values) and L1 ratio (0.05–1) were chosen by cross-validation. For gradient boosting (scikit-learn `HistGradientBoostingRegressor`), 40 configurations were drawn at random over learning rate, number of trees, maximum depth, number of leaves, minimum leaf size and L2 penalty. Early stopping was not used, because its internal validation split is not grouped by twin pair. All tuning used the training split only, with five inner folds grouped by twin pair and mean squared error as the criterion. Each model was then refitted on the whole training split and evaluated once on the test split. The comparators used the same 105 inputs; in the age-15 analysis, ACE18 and SUD18 were removed. Like the network, they did not model twin dependence. <!-- src: Models/ML/Method/ml_comparators_test_predictions.py (ELASTIC_NET_L1_RATIOS, BOOSTING_SPACE, N_BOOSTING_CONFIGS = 40, GroupKFold n_splits = 5); scikit-learn 1.8.0 from versions_python.txt -->

## Evaluation and statistical comparison

All models were evaluated on the same five test sets with the same metric code: the coefficient of determination, $R^2 = 1 - \sum(y_i-\hat{y}_i)^2 / \sum(y_i-\bar{y})^2$, root-mean-squared error (RMSE), mean absolute error (MAE) and Spearman's rank correlation. Results are given as mean and standard deviation over the five splits. <!-- src: Models/Model Comparison/prediction_evaluation.py core_metrics(), summarise() (ddof = 0); GLMM_updated_split.R header items 4–5 -->

Models were compared split by split with the corrected resampled t-test, which accounts for the overlap between training sets across splits [@R-NADEAU] (test-to-training ratio 1/3, 4 degrees of freedom), and 95% confidence intervals for the mean differences were derived from the same corrected variance. P-values were adjusted with the Holm method within each metric, across all 18 comparisons among the network and the mixed models (three model pairs with each of two predictor sets, and full versus age-15-only predictors for each model, for both outcomes). Comparisons of the network with the three machine-learning comparators, and of each comparator's full and age-15-only predictor sets, formed a second family of 18 comparisons with its own Holm correction. <!-- src: prediction_evaluation.py corrected_t_test(), holm() --> With five splits these tests have low power, so a non-significant difference is not evidence of equivalence.

For the mixed models, interaction terms were assessed with their Wald p-values, Holm-adjusted across the 24 terms, and PRS smooth terms by their effective degrees of freedom. <!-- src: Models/GLMM/Results/{Pos,Neg}_GLMM_updated_all_coefs.csv; Models/GAMM/Results/GAMM_updated_{Pos,Neg}_smooth_terms.csv -->

**Interaction ablation.** To test whether the SUD15 interactions improved prediction as a whole, each model was also fitted without them on the same five splits. The mixed models were refitted without the 24 interaction terms, keeping all main effects and the random intercept. DCMFNet was retrained without the fused stream, so that all inputs passed only through the independent stream and the output layers; it used the same hyperparameters and training procedure, which were not re-tuned for the reduced network. Because the feature gates multiply inputs by input-dependent weights, the reduced network can still represent some interactions; the ablation therefore removes the dedicated SUD15 fusion, not every interaction. Test R² with and without the interactions was compared split by split with the corrected resampled t-test, with a Holm correction across these six comparisons (three models, two outcomes). On each split, we also tested the 24 interaction coefficients of the mixed models jointly with a Wald test, and in the GLMM with a likelihood-ratio test of the two fits refitted by maximum likelihood. <!-- src: Models/Model Comparison/interaction_ablation_mixed.R; Models/DCMFNet/Analysis/dcmfnet_fusion_ablation.py (IndependentStreamModel; corrected_t_test, holm); authorised by the author 2026-10-29 -->

## Calibration, subgroup and sensitivity analyses

The comparison was repeated with participant-level test predictions from DCMFNet retrained with the same seeds, splits and hyperparameters, so that all models were scored with identical code. <!-- src: Models/DCMFNet/Analysis/dcmfnet_test_predictions.py; Models/Model Comparison/glmm_gamm_test_predictions.R; prediction_evaluation.py -->

- **Calibration** [@R-VANCAL]: calibration-in-the-large (mean observed minus mean predicted), calibration slope and intercept from regressing observed on predicted values, and calibration curves over deciles of the predictions.
- **Upper tail:** within the 10% of test participants with the highest observed scores, we report the precision of the model's top 10% of predictions, the area under the ROC curve for identifying this group, and RMSE and mean bias.
- **Sex:** all metrics computed separately for males and females.
- **Predictors measured by age 15:** all models were refitted without ACE18 and SUD18. The mixed models dropped these variables and their two interaction terms (97 main effects, 22 interactions). For DCMFNet, whose architecture expects every group, both inputs were set to their training-set mean in training and test data, which removes their information while keeping the tuned architecture. <!-- src: glmm_gamm_test_predictions.R PREDICTOR_SETS; dcmfnet_test_predictions.py neutralise_age18() -->

## Attribution analyses

We asked whether DCMFNet's gate values reflect what the network relies on for its predictions, and whether DCMFNet and the mixed-effects models rely on the same predictor groups. For each split, the following quantities were computed for the 11 predictor groups on the test set, from the same retrained models: <!-- src: Models/DCMFNet/Analysis/shap_analysis.py; Models/GLMM/Analysis/GLMM_shap_updated.R; Models/GAMM/Analysis/GAMM_shap_updated.R; Models/Model Comparison/attribution_agreement.py -->

1. **Gate values:** the mean gate value per group in DCMFNet's fused and independent streams. <!-- src: Models/DCMFNet/Analysis/importance_analysis_avg.py:43-114 -->
2. **SHAP values for DCMFNet** [@R-SHAP]: expected-gradients SHAP values (100 training rows as background, 200 samples per explained row), summed within each group; group importance was the mean absolute group SHAP value.
3. **Exact SHAP values for the GLMM and GAMM:** for the population-level prediction, with the training data as reference. Main effects contribute $\beta_j(x_j - \bar{x}_j)$. Each product term $\gamma S P$ was split between SUD15 and its partner according to its two-player Shapley values, $\tfrac{\gamma}{2}(S-\bar{S})(P+\bar{P})$ and $\tfrac{\gamma}{2}(P-\bar{P})(S+\bar{S})$. In the GAMM, each smooth PRS term contributes $f_k(x_k) - \bar{f}_k$, where $\bar{f}_k$ is its mean over the training data.
4. **Grouped permutation importance** for all three models: the drop in test $R^2$ when all variables of a group are permuted jointly, averaged over 10 permutations.

Agreement between sources was summarised by Spearman's rank correlation across groups and the overlap of the three highest-ranked groups, and stability by the correlation between splits. All attributions describe associations used by the fitted models, not causal effects.

## Software and reporting

Analyses used Python 3.11.5 with PyTorch 2.11.0, scikit-learn 1.8.0 [@R-SKLEARN], LightGBM 4.6.0, Optuna 4.8.0, SHAP 0.50.0 and LazyPredict 0.3.0, and R 4.6.1 with lme4 2.0-6 and mgcv 1.9-4. <!-- src: versions_python.txt and versions_R.txt, exported from the analysis server 2026-10-28 (V-01/V-04). These record the server environments on that date; they are assumed to be the environments the analyses ran in (author to confirm, V-01). --> The analysis code is archived on Zenodo (https://doi.org/10.5281/zenodo.23755819) and maintained on GitHub (https://github.com/sil-pixel/Thesis). <!-- C-05, 2026-10-29: concept DOI (resolves to the latest release; v1.0-manuscript = record 23755820, commit 7d2eb05) --> Reporting follows the TRIPOD+AI statement [@R-TRIPODAI] (Additional file 2). No protocol for this analysis was published or registered. Patients and members of the public were not involved in the design, conduct, reporting or interpretation of the study. <!-- author, 2026-10-27 (Q-08) -->

**Use of AI tools.** Claude Code (Anthropic) was used to help draft and revise the manuscript text from the authors' own write-up and results, and to review and write analysis, figure and table scripts, which the authors checked and ran. ChatGPT (OpenAI) was used to check the first author's original write-up. The authors take full responsibility for the content. <!-- author approved the shorter wording 2026-10-29. "Original write-up" stands for the author's thesis, written by the author and checked with ChatGPT; the paper does not name the thesis. Springer Nature asks that LLM use beyond copy-editing be documented in the Methods [VERIFY: current policy page before submission]. -->

## Ethics

`[AUTHOR INPUT: approving committee(s), reference number(s) and date(s) for CATSS and the genotype linkage; consent procedure (parents for minors, participants at age 18); statement that the study was carried out in accordance with relevant guidelines and regulations / the Declaration of Helsinki]` <!-- C-02 --> The same information is given under *Declarations*.
