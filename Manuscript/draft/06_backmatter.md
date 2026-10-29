<!--
STATUS 2026-10-23: Back matter for BMC Psychiatry (replaces 07_declarations.md, written for AIIM).
BMC research articles end with: List of abbreviations; Declarations (the headings below, "Not applicable" where relevant);
References; figure legends; tables; additional files. [VERIFY: JOURNAL REQUIREMENT — headings taken from the BMC
series declarations format; the BMC Psychiatry page itself could not be fetched. "Clinical trial number" was added to
BMC declarations in 2025 — confirm it is required for observational studies.]
Nothing here may be invented: every [AUTHOR INPUT] needs the author's own information.
-->

# Abbreviations

ACE: adverse childhood experiences; ADHD: attention-deficit/hyperactivity disorder; ASD: autism spectrum disorder; CATSS: Child and Adolescent Twin Study in Sweden; DCMFNet: Deep Cross-Modal Fusion Network; DEP: depressive symptoms; GAMM: generalised additive mixed model; GLMM: generalised linear mixed model; GWAS: genome-wide association study; IGF: iterative gated fusion; MAE: mean absolute error; PC: principal component; PLE-HM: psychotic-like and hypomanic experiences; PRS: polygenic risk score; RMSE: root-mean-squared error; SES: socioeconomic status; SHAP: Shapley additive explanations; SUD: substance use; TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, updated for artificial intelligence.
<!-- Update if the comparator is called "GLMM" in the final text (Methods AUTHOR INPUT). -->

# Supplementary information

- **Additional file 1** (PDF): Table S1, DCMFNet hyperparameter search and tuning history; Table S2, default-settings screening of 40 machine-learning algorithms; Table S2b, hyperparameters selected for the tuned comparators; Table S3, fixed effects and smooth terms of the linear and additive mixed models; Table S4, predictors by group; Table S5, selection of the fusion depths.
- **Additional file 2** (PDF): TRIPOD+AI checklist.

# Declarations

## Ethics approval and consent to participate
`[AUTHOR INPUT: "CATSS and the genotype linkage were approved by [committee, e.g., the Regional Ethical Review Board in Stockholm / Swedish Ethical Review Authority], reference number(s) [xxx], date(s) [xxx]. [Consent procedure: parents for participants under 15/18; participants themselves at age 18 — VERIFY]. The study was conducted in accordance with the Declaration of Helsinki."]` — TODO C-02

## Consent for publication
Not applicable. <!-- No individual-level data, images or case details are reported. -->

## Availability of data and materials
Draft: "The CATSS data contain sensitive personal information and cannot be shared publicly under Swedish data-protection law. Researchers can apply for access through the Swedish Twin Registry, Karolinska Institutet `[AUTHOR INPUT: exact procedure/URL]`. The analysis code is available at https://doi.org/10.5281/zenodo.23755819."

## Competing interests
`[AUTHOR INPUT]` or: "The authors declare that they have no competing interests."

## Funding
`[AUTHOR INPUT: funders and grant numbers for CATSS, genotyping and this work, and the role of the funders]` or: "This research received no specific funding."

## Authors' contributions
`[AUTHOR INPUT]` Example structure only: "XX designed the study, wrote the analysis code, analysed the data and drafted the manuscript. YY supervised the study and contributed to the design and interpretation. All authors read and approved the final manuscript."

## Acknowledgements
`[AUTHOR INPUT: CATSS participants and staff; colleagues (names only with their consent)]`

## Clinical trial number
Not applicable. `[VERIFY: JOURNAL REQUIREMENT]`

# Figure legends

**Fig. 1** Participant flow from the CATSS twins in the study data to the complete-case analysis sample and the five evaluation splits. Each excluded individual is counted at the first missing predictor group, in the order shown; steps that excluded fewer than five individuals are combined with the preceding step. <!-- src: figures/make_figures.py from analysis_sample_flow.csv; top step from thesis Figs 3.1 and 3.3 (W-02, author confirmed 2026-10-28) -->

**Fig. 2** Architecture of the deep cross-modal fusion network (DCMFNet). Left: substance use at age 15 (SUD15, the anchor) is fused with each of nine predictor groups in an iterative gated fusion (IGF) block of depth *L* (fused stream); each block output passes through a feature gate $\phi$, and the gated outputs are concatenated and gated again ($\phi_F$). The independent stream gates all 105 inputs, including the genotyping batch × principal component terms ($\phi_I$). Both streams are concatenated, gated ($\phi_{\mathrm{out}}$) and passed to a linear output. Numbers of inputs per group in parentheses. Right: one layer of an IGF block (Eqs 1–3 in Methods). <!-- src: figures/make_figures.py; Models/DCMFNet/Method/dcmfnet/model.py; hyperparameters.json -->

**Fig. 3** Test-set R² of each model on each of the five splits, with all predictors. Each symbol is one split; grey lines join the results of the same split, so that within-split differences between models can be seen. Black bars and the numbers above each panel give the mean over the five splits. The dashed line separates the fusion network and the mixed models from the comparators tuned within the training data. <!-- src: figures/make_figures.py fig3() from Models/Model Comparison/prediction_metrics_per_seed.csv (predictor_set = full) -->

**Fig. 4** Calibration of each model, by decile of predicted score: mean predicted against mean observed score per decile, averaged over the five test sets, with all predictors. Each panel highlights one model; the other models are shown in light grey for comparison. The dashed line shows perfect calibration. <!-- src: figures/make_figures.py fig4() from Models/Model Comparison/calibration_bins.csv (same data as calibration_curves_full.pdf) -->

**Fig. 5** Importance of each predictor group, by model and attribution method, for psychotic-like and hypomanic experiences (left) and depressive symptoms (right). Each cell gives the group's share of the total importance for that method, as a percentage averaged over the five test sets. The two left-hand columns of each panel show DCMFNet's mean gate values, also expressed as shares. The fused stream has no gate for substance use at age 15 (the anchor group) or for the batch × PC terms, so these cells are empty. Negative permutation importances were set to zero. Groups are ordered by their mean GLMM SHAP share across the two outcomes. Importance describes what each fitted model uses for prediction, not causal effects. <!-- src: Models/Model Comparison/attribution_importance_by_group.pdf (attribution_importance_plot.py, from attribution_modality_summary.csv). Figure labels say "GLMM"; relabel if the model is called LMM in the final text. -->
