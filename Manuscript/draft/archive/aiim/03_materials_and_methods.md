<!--
DRAFT STATUS (2026-10-14): Describes the protocol AS CONDUCTED for the thesis, reconstructed from code and logs.
Several elements are flagged for change pending author-approved re-analysis (TODO E-01–E-07). When the
re-analysis is run, §2.6–2.8 must be rewritten to describe the corrected protocol, and the as-conducted
protocol should move to the Supplement only if those results are still reported.
Outcome labels follow D-03 (neutral names) pending author confirmation (TODO M-01).
-->

# 2. Materials and methods

## 2.1. Study population and data source

We used data from the Child and Adolescent Twin Study in Sweden (CATSS), a population-based cohort of twins born in Sweden and drawn from the Swedish Twin Registry. Twins and their parents were invited to complete questionnaires at ages 9 (or 12), 15 and 18 [R-CATSS]. <!-- src: Thesis Report §3.1 p.7; cohort design per Anckarsäter et al. 2011 [VERIFY: invitation ages and response rates from cohort paper] -->
For the present analysis, we used age-9 parent-reported neurodevelopmental measures, age-15 and age-18 self-reports, and genotype data available for a subset of participants. `[AUTHOR INPUT: birth cohorts included, data freeze/extraction date, who reported age-9 measures]`

Individuals were eligible if they had substance-use data at age 15 and outcome data at age 18 (n = 9,261 of 13,146 twins with any exposure/outcome record). Of these, 9,160 had genotype data that passed quality control. <!-- src: Thesis Figs 3.1, 3.3 p.8, p.11 [VERIFY: the two flow charts report 9,271→9,160 and 13,146→9,261; reconcile into one flow diagram, TODO W-02] -->
The analysis sample comprised all individuals with complete data on every predictor and outcome (complete-case analysis): **n = 4,033 individuals from 2,690 twin pairs**. <!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1-3 (shape (4033, 125); 2,017 + 673 = 2,690 cmpair IDs) -->
`[TODO: Table 1 — characteristics of the analysis sample and comparison with excluded participants (TODO A-02)]`

## 2.2. Outcomes

Two continuous outcomes were derived from age-18 self-report items. `[AUTHOR INPUT: instrument names, item response scales and references — TODO M-01]`

- **Psychotic-like and hypomanic experiences score (PLE-HM).** Sum of 24 items: 7 psychotic-like experiences (e.g., hearing voices, feeling spied upon, thought reading, special messages) and 17 hypomanic/manic items (e.g., racing thoughts, increased energy, decreased need for sleep). The sum was divided by 46. <!-- src: Data Preprocessing/Data Visualisation/rename_catss_with_mapping.py:73-81, 91 [VERIFY: maximum attainable score; why 46 for 24 items] -->
- **Depressive symptom score (DEP).** Sum of 11 items (e.g., feeling sad, unhappy, lonely, not enjoying life, poor appetite). The sum was divided by 33. <!-- src: rename_catss_with_mapping.py:78, 92 [VERIFY: instrument (SMFQ-derived?) and item range] -->

Both scores therefore lie in [0, 1]. They were analysed as continuous outcomes, without clinical thresholds. In the thesis these outcomes were labelled "positive" and "negative" schizophrenia symptoms. We use construct-descriptive labels because the depressive-symptom items do not measure negative symptoms of schizophrenia (see §4.x Limitations). `[AUTHOR INPUT: confirm final outcome naming]`

## 2.3. Predictors

The predictors (105 variables) were organized into 11 a priori groups ("modalities"). Group sizes are given in parentheses: <!-- src: best_model_training.log:2 ; Models/DCMFNet/Method/legacy/train.py:74-87 -->

1. **Substance use at age 15 (SUD15; 6)** — frequency of cigarette, snus (moist snuff), alcohol, cannabis, non-medical painkiller/opioid use, and an "other drugs" variable. The "other drugs" variable is the maximum across highly correlated items (stimulants, opioids, psychedelics, inhalants, sedatives, performance enhancers). <!-- src: Data Preprocessing/Feature Selection/feature_selection.py:76-97 -->
2. **Polygenic risk scores (PRS; 16)** — a genome-wide schizophrenia PRS and cell-type-specific PRS restricted to the 10% most specifically expressed genes of brain cell types from a human brain cell atlas [R-SILETTI, R-YAO2025], computed with PRS-CS [R-PRSCS]. `[AUTHOR INPUT: GWAS summary statistics (e.g., PGC3 [R-PGC3]), LD reference, PRS-CS φ setting, genotyping platforms/batches, imputation, QC thresholds — TODO W-03]`
3. **Psychotic-like, hypomanic and emotional symptoms at age 15 (SCZ15; 23)**.
4. **ADHD symptoms at age 9 (ADHD9; 19)** and 5. **autism-related traits at age 9 (ASD9; 17)** `[VERIFY: instrument (A-TAC?) and informant]`.
6. **Adverse experiences at age 15 (ACE15; 7)** — bullying victimization items (six highly correlated bullying types combined into one item).
7. **Adverse experiences at age 18 (ACE18; 4)** — emotional abuse, hate crime, witnessing crime, and a combined item for other abuse/violence. <!-- src: feature_selection.py:111-130 -->
8. **Substance use at age 18 (SUD18; 4)**.
9. **Parental socioeconomic indicators (SES; 4)** — parental education and country of birth.
10. **Sex (1)** `[AUTHOR INPUT: register-based sex assigned at birth vs self-report; definition per SAGER (Guide p.8–9)]`.
11. **Genotyping batch × principal component terms (4)** — products of genotyping-batch indicators and the first two genetic principal components, which adjust for population structure and batch effects. <!-- src: Data Preprocessing/Data Visualisation/rename_catss_with_mapping.py (add_batch_pc_interactions, pc_cols=['PC1','PC2']) -->

Because ACE18 and SUD18 were measured at the same assessment as the outcomes, models that include them estimate concurrent rather than prospective associations for these groups (see TODO E-05).

**Variable screening.** Starting from 236 candidate variables, we removed variables with > 90% missingness. We combined or removed variables with pairwise Pearson |r| > 0.9. We removed variables with 50–90% missingness that were not judged informative, and we grouped moderately correlated items guided by SHAP-value clustering from a LightGBM model [R-LGBM, R-SHAP]. <!-- src: Thesis §3.4 p.10-11; feature_selection.py; Data Analysis/Feature Importance/feature_importance.py -->
`[VERIFY: whether any outcome-informed (SHAP-based) step removed or retained variables; that step used a separate 70/30 split of the full cohort (feature_importance.py:31-45), which overlaps the evaluation test sets — D-09, TODO E-04]`
Responses of "don't know/don't want to answer" were set to missing. <!-- src: feature_selection.py:136-143 --> `[VERIFY: grouped bullying/abuse items encode missing as −1 (feature_selection.py:116-130) — TODO A-03]`

## 2.4. Deep cross-modal fusion network

### 2.4.1. Overview
We adapted the Deep Cross-Modal Fusion Network (DCMFNet) [R-DCMF], originally proposed for fusing image and text representations through iterative gated fusion (IGF). We applied it to grouped tabular predictors. Substance use at age 15 serves as the **anchor** (guided) group. Each of the nine groups 2–10 in §2.3 is fused with the anchor in its own IGF block. The batch × PC group enters only through a parallel *independent stream* that receives all raw inputs. <!-- src: Models/DCMFNet/Method/dcmfnet/model.py:115-205; legacy/train.py:49, 74-87 --> Fig. `[TODO: architecture figure regenerated from code — existing Models/DCMFNet/Architecture/*.png must be checked against the implementation]`

### 2.4.2. Iterative gated fusion block
Let $\mathbf{x}_a \in \mathbb{R}^{d_a}$ denote the anchor features ($d_a = 6$) and $\mathbf{x}_m \in \mathbb{R}^{d_m}$ the features of group $m \in \{1,\dots,M\}$, with $M = 9$. Block $m$ has depth $L_m$ and is initialised with $\mathbf{g}_m^{(0)} = \mathbf{x}_m$. For $l = 1,\dots,L_m$, with layer-specific weights,

$$\mathbf{f}_m^{(l)} = \tanh\!\big(\tanh(\mathbf{W}_1^{(l)}\mathbf{x}_a + \mathbf{b}_1^{(l)}) \odot \tanh(\mathbf{W}_2^{(l)}\mathbf{g}_m^{(l-1)} + \mathbf{b}_2^{(l)})\big), \tag{1}$$

$$\mathbf{z}_m^{(l)} = \sigma\!\big(\mathbf{W}_z^{(l)}(\mathbf{f}_m^{(l)} + \mathbf{g}_m^{(l-1)}) + \mathbf{b}_z^{(l)}\big), \tag{2}$$

$$\mathbf{g}_m^{(l)} = \tanh\!\big(\mathbf{W}_g^{(l)}[\mathbf{f}_m^{(l)};\, \mathbf{f}_m^{(l)} \odot \mathbf{z}_m^{(l)}] + \mathbf{b}_g^{(l)}\big), \tag{3}$$

where $\mathbf{W}_1^{(l)} \in \mathbb{R}^{d_m \times d_a}$, $\mathbf{W}_2^{(l)}, \mathbf{W}_z^{(l)} \in \mathbb{R}^{d_m \times d_m}$, $\mathbf{W}_g^{(l)} \in \mathbb{R}^{d_m \times 2d_m}$, $\sigma$ is the logistic sigmoid, $\odot$ is element-wise multiplication and $[\cdot;\cdot]$ is concatenation. Eq. (1) is a bilinear-type interaction between the anchor and the current group representation. Eqs. (2)–(3) form the adaptive selection gate of the original architecture. <!-- src: dcmfnet/model.py:11-47 (FusionModule, GatedModule, GatedFusionLayer); identical in thesis-era commit 6173d10 DCMFNet/model.py:38-101 -->
The block output concatenates the fused representations from all layers, $\mathbf{u}_m = [\mathbf{f}_m^{(1)};\dots;\mathbf{f}_m^{(L_m)}] \in \mathbb{R}^{L_m d_m}$, which is then passed through a feature-gating module $\phi_m$ (Eq. 4). <!-- src: dcmfnet/model.py:106-112 -->

### 2.4.3. Feature gating
The thesis motivated this component with squeeze-and-excitation (SE) attention [R-SE]. In the implementation, however, the gate is computed **per instance from the instance's own features**, without a pooling ("squeeze") step:

$$\phi(\mathbf{u}) = \mathbf{u} \odot \sigma\!\big(\mathbf{V}_2\, \mathrm{Dropout}(\mathrm{ReLU}(\mathbf{V}_1\mathbf{u} + \mathbf{c}_1)) + \mathbf{c}_2\big), \tag{4}$$

with hidden width $\max(\lfloor D/r \rfloor, h_{\min})$ for input dimension $D$, reduction ratio $r$ and minimum width $h_{\min}$. <!-- src: dcmfnet/model.py:50-77 (SEAttention); thesis-era 6173d10 DCMFNet/model.py:139-164 — discrepancy D-06 -->
This is input-conditioned multiplicative feature gating. It assigns instance-specific weights, so it does not provide a population-level feature importance.

### 2.4.4. Streams and output
The fused stream gates the concatenated block outputs, $\mathbf{h}_F = \phi_F([\mathbf{u}'_1;\dots;\mathbf{u}'_M])$ with $\mathbf{u}'_m = \phi_m(\mathbf{u}_m)$. The independent stream gates all raw inputs, $\mathbf{h}_I = \phi_I([\mathbf{x}_a;\mathbf{x}_1;\dots;\mathbf{x}_M;\mathbf{x}_c])$, where $\mathbf{x}_c$ is the batch × PC group. The prediction is

$$\hat{y} = \mathbf{w}^\top \phi_{\mathrm{out}}([\mathbf{h}_F;\mathbf{h}_I]) + b. \tag{5}$$

The output layer is linear (predictions are not constrained to [0,1]). <!-- src: dcmfnet/model.py:168-205 -->
Because $\mathbf{h}_I$ contains gated raw inputs that feed a linear output layer, the network includes an approximately additive (gated linear) pathway in addition to the interaction pathway.
Input features were not standardised before entering the network. <!-- src: legacy/train.py:74-94 (raw .to_numpy()) — D-13, TODO E-07 -->

### 2.4.5. Loss function
To up-weight the sparse upper tail of the outcome distributions, we used a weighted Huber loss. Let $\ell_\delta(y_i, \hat{y}_i)$ be the Huber loss with threshold $\delta$. Training outcomes were assigned to $K$ equal-width bins on [0, 1]. Bin $k$ with $n_k$ training observations received the weight $\tilde{w}_k = (1/(n_k + 1)) / \overline{(1/(n_\cdot + 1))}$, truncated at 20. For each mini-batch $B$, a focal factor $q_i = (|y_i - \hat{y}_i| + 10^{-6})^{\gamma}$ was computed without gradient and normalised by its batch mean. The combined weights $\omega_i = q_i\,\tilde{w}_{k(i)}$ were renormalised to batch mean 1. The batch loss was

$$\mathcal{L}_B = \frac{1}{|B|}\sum_{i \in B} \omega_i\, \ell_\delta(y_i, \hat{y}_i). \tag{6}$$

<!-- src: dcmfnet/losses.py:15-76 (ImbalancedRegressionLoss); bin weighting identical in thesis-era git 6173d10:DCMFNet/loss.py:60-73 — D-07 -->

## 2.5. Comparator models

**Linear mixed model (LMM).** A Gaussian mixed model with identity link included all predictor groups as main effects, SUD15 × covariate product terms, batch-by-PC terms, and a random intercept per twin pair. Models were fitted with `lme4::glmer` without a family argument, which defaults to a Gaussian linear mixed model. <!-- src: Models/GLMM/Method/GLMM.R:7, 16-65, 220, 239 [VERIFY: lme4 version; prediction for test pairs (new levels) uses population-level fixed effects?] -->
`[TODO: The as-conducted interaction set contains 97 of the 546 intended SUD15 × covariate terms because of element-wise vector recycling in model-formula construction (D-01). Describe the re-specified model after TODO E-01.]`

**Generalised additive mixed model (GAMM).** Same specification, with each PRS entered as a penalised smooth term and the twin-pair random intercept entered as a random-effect smooth. The model was fitted with `mgcv::gam` (Gaussian family, REML smoothing-parameter estimation). <!-- src: Models/GAMM/Method/GAMM.R:7, 257, 277 [VERIFY: basis dimension k and mgcv version] -->

**Machine-learning benchmarks.** Forty regression algorithms (linear, regularised, kernel, nearest-neighbour, tree-ensemble, boosting and neural) were fitted with default settings using LazyPredict [R-LAZY] on the same 105 inputs and the **same partitions** as DCMFNet. <!-- src: Models/ML/Method/lazy_predict.py:29-47, 82-111; 40 models in Models/ML/Results/Pos_benchmark_summary.csv --> `[VERIFY: LazyPredict version and its internal preprocessing (imputation/scaling) — TODO A-05]` Some estimators select their regularisation strength by internal cross-validation (e.g., LassoCV). That internal cross-validation was not grouped by twin pair `[VERIFY]`.

## 2.6. Data partitioning

Data were split at the twin-pair level (both co-twins always in the same partition) into 75% training and 25% test sets. This was repeated for five random seeds (42–46). The split was not stratified by outcome. <!-- src: legacy/train.py:61-68, 503-510; D-08 --> Within each training set, a further pair-level 80/20 split defined a validation set for early stopping and learning-rate scheduling (≈2,410 training, ≈600 validation, ≈1,010 test individuals per seed). <!-- src: legacy/train.py:145-152; best_model_training.log lines 3, 9, 361, 367 -->
`[TODO: as conducted, LMM/GAMM partitions were drawn in R with the same seed integers and therefore differ from the Python partitions (D-02). Re-run on shared partitions — TODO E-02.]`

## 2.7. Training and hyperparameter selection

DCMFNet was trained with Adam [CITATION NEEDED: Kingma & Ba 2015], weight decay, and ReduceLROnPlateau scheduling on validation RMSE (minimum learning rate 10⁻⁶). We used early stopping on validation RMSE and restored the best-epoch weights. <!-- src: legacy/train.py:371-476 -->
Hyperparameters were selected with Optuna [R-OPTUNA] (50 trials per outcome; median pruning after 5 startup trials and 5 warm-up epochs), minimising validation RMSE on the seed-42 training partition. The search space is given in Table S1. <!-- src: git c49eec1:DCMFNet/tuning.py:49-71, 204-229 — D-21 --> The selected values are in Table 2.

**Table 2.** Selected DCMFNet hyperparameters. <!-- src: Models/DCMFNet/hyperparameters.json -->

| Hyperparameter | PLE-HM model | DEP model |
|---|---|---|
| Learning rate | 1.00 × 10⁻⁴ | 3.22 × 10⁻⁴ |
| Batch size | 4 | 6 |
| Maximum epochs | 23 | 20 |
| Weight decay | 2.95 × 10⁻⁵ | 1.79 × 10⁻⁴ |
| IGF depth $L_m$ (PRS, SCZ15, ADHD9, ASD9, ACE15, ACE18, SUD18, SES, Sex) | 5, 2, 1, 5, 4, 1, 1, 1, 5 `[AUTHOR INPUT/VERIFY: D-05]` | 2 (all blocks) |
| Dropout | 0.234 | 0.326 |
| Gate reduction ratio $r$ | 6 | 5 |
| Minimum gate width $h_{\min}$ | 13 | 5 |
| Huber δ | 0.086 | 0.116 |
| Focal γ | 1.32 | 2.62 |
| Number of bins $K$ | 13 | 10 |
| Scheduler patience / factor | 4 / 0.432 | 2 / 0.549 |
| Early-stopping patience | 5 | 7 |
| Trainable parameters | 105,501 | 71,623 |

Table note: Parameter counts were computed from the model definition with the input group sizes in §2.3. <!-- src: computed 2026-10-14 by instantiating dcmfnet.model.DCMFNet with sizes [6,16,23,19,17,7,4,4,4,1,4]; no data used -->

**Depth selection (as conducted).** The Optuna search tuned a single depth shared by all blocks (best: 3 for PLE-HM, 2 for DEP). A subsequent sweep varied each block's depth from 1 to 5 while holding the others fixed, selecting the depth that minimised validation RMSE. <!-- src: Models/DCMFNet/Analysis/complexity_analysis.py:130-200 --> For the PLE-HM model, the per-block depths from the seed-42 sweep were adopted after comparing per-block and uniform configurations on the test sets. For the DEP model, uniform depth 2 was retained. <!-- src: complexity_analysis.py:204-250; Results/Complexity Analysis/Pos_complexity_comparison.csv; git 5c35f0e, 6173d10 — D-04, D-05 -->
`[TODO: This selection used test-set performance. Replace with nested selection (TODO E-03) and rewrite this paragraph.]`

## 2.8. Evaluation

Performance was assessed on each test partition with the coefficient of determination ($R^2 = 1 - \sum(y_i-\hat{y}_i)^2 / \sum(y_i-\bar{y})^2$, computed on the test set), root-mean-squared error (RMSE), mean absolute error (MAE), Pearson's $r$ and Spearman's $\rho$. Results are summarised as mean ± standard deviation over the five partitions. <!-- src: legacy/train.py:157-203, 541-563 (np.std, ddof=0); GLMM.R:269 and GAMM.R:308 use sd() (ddof=1) — SDs are not computed on the same basis; harmonise when re-running -->
`[TODO: paired comparisons with corrected resampled t-tests [R-NADEAU] and Holm adjustment (TODO A-01); calibration slope/intercept and smoothed calibration curves [R-VANCAL] (TODO E-09); pre-specified upper-tail metrics (TODO E-10).]`

## 2.9. Model interpretation

`[TODO: Rewrite after E-06. As conducted, "importance" was the mean gate value per input group from the fused, independent and output gating modules, averaged over test individuals and five partitions (importance_analysis_avg.py:43-114). These values are not comparable across groups with unstandardised inputs and do not measure contribution to predictions (D-12). The manuscript should use grouped permutation importance and stream ablation instead [R-JAIN].]`

## 2.10. Software

Analyses used Python `[VERIFY version]` with PyTorch `[VERIFY]`, scikit-learn `[VERIFY]`, Optuna `[VERIFY]` and LazyPredict `[VERIFY]`, and R `[VERIFY]` with `[VERIFY: lme4/mgcv versions]`. Code is available at `[TODO: tagged release + DOI — TODO C-05]`.

## 2.11. Ethics

`[AUTHOR INPUT: ethics approval body, reference number(s) and date(s) for CATSS and genotype linkage; consent procedure for minors and at age 18; data protection basis — TODO C-02]`. The study followed the TRIPOD+AI reporting guideline [R-TRIPODAI] `[TODO: only after checklist completion — TODO C-06]`.
