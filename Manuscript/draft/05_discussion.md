<!--
STATUS 2026-10-23: First prose draft for BMC Psychiatry (Discussion with strengths and limitations, then Conclusions).
- Written from the verified results in 04_results.md (performance, paired tests, interactions, smooths, ML baselines).
- Paragraphs marked TODO depend on the pending runs (attribution E-06; calibration/sex/top decile/age-15 E-09, A-04, E-10, E-05).
  Write them to the evidence; if the gates agree with SHAP and permutation, say so and soften the related limitation.
- No causal language; "no detectable gain", not "equivalent"; no clinical-utility claims.
-->

# Discussion

In a population-based twin cohort, a deep cross-modal fusion network built around adolescent substance use did not predict psychotic-like and hypomanic experiences or depressive symptoms at age 18 better than linear and additive mixed models that were given the same predictors and interaction terms mirroring the network's fusion design. Mean test R² was about 0.02 lower for the network than for the mixed models for PLE-HM and similar for DEP, and no paired difference was statistically significant. Penalised linear models fitted with default settings did best. The mixed models also showed little evidence of the substance-use interactions that motivate the fusion design: the number of nominally significant interaction terms was close to the number expected by chance, and most PRS effects were close to linear. `[TODO: one or two sentences on calibration and the age-15-only analysis, from prediction_evaluation.py]` `[TODO: one sentence on whether gates agreed with SHAP and permutation importance, from attribution_agreement.py]`

These findings are consistent with comparisons in clinical prediction, where flexible machine-learning models rarely outperform regression when the comparison is at low risk of bias [@R-CHRIST], and with benchmarks showing that deep networks are seldom the best choice for medium-sized tabular data [@R-GRIN]. They also agree with earlier machine-learning work in this cohort [@R-TATE] `[VERIFY: content]`. Two features of our design make the result conservative for the network. First, only the network was tuned, and part of its configuration was chosen with test-set information; both can only favour it. Second, the mixed models were given the same interaction structure as the network, so the comparison isolates how the interactions are modelled rather than whether they are included. `[TODO: L-02 — compare with direct competitors (ML/DL with PRS and environmental predictors for psychotic or depressive outcomes) once identified]`

Several explanations are possible. The training sets contained about 3,000 individuals, while the networks had 72,000 to 106,000 trainable parameters (Table 2). At this sample size, flexible models are likely to fit noise rather than signal [@R-RILEY] `[VERIFY: content]`. The predictors and outcomes are questionnaire sum scores and PRS, which explain a limited share of variance, and this limits the attainable R² for any model. Finally, if interactions between substance use and other domains are weak at the population level, a model designed to capture them has little to gain. Our results cannot distinguish between these explanations. They also do not show that such interactions are absent: prediction accuracy is a different question from testing moderation, and the interaction tests here had limited power after correction for multiple testing.

`[TODO — attribution paragraph (E-06). Report: (a) agreement between DCMFNet gate values and SHAP / permutation importance on the same models (Spearman, top-3 overlap); (b) agreement between DCMFNet and the LMM; (c) seed stability. If gates disagree with SHAP/permutation, conclude that gate values should not be read as predictor importance [@R-JAIN], as was done in the thesis. If they agree, say that in this setting gate rankings were consistent with model-agnostic attributions, but that this does not make them causal.]`

`[TODO — calibration, sex and age-15 paragraph (E-09, A-04, E-05, E-10). Report calibration slopes and calibration-in-the-large for each model; whether conclusions differ by sex; how much performance drops without ACE18/SUD18 and whether the model ranking changes; upper-tail performance.]`

## Strengths and limitations

Strengths of this study include a population-based twin cohort with genetic and multi-informant questionnaire data; evaluation of all models on identical twin-pair-grouped splits with the same metric code; comparators that matched the network's inputs and interaction structure; paired inference that accounts for overlapping training sets; and analyses of calibration, sex, predictors measured before the outcome and attribution.

The study also has limitations.

- **Configuration selection.** The network's depth configuration, and the choice among tuning runs, used test-set results, and the tuning data overlapped later test sets. The network's own performance estimates are therefore likely optimistic, although this bias works against our main conclusion.
- **Feature screening.** The variable-screening step used the outcomes on data that overlap the test sets [@R-KAPOOR]. This affects all models equally but may inflate absolute performance.
- **Complete-case analysis.** Only individuals with complete data were analysed, about 44% of the eligible sample `[VERIFY: against analysis_sample_flow.csv]`. `[TODO: summarise Table 1 differences between included and excluded participants]`. Results may not generalise to the full cohort.
- **Outcome measures.** The outcomes are continuous questionnaire scores, not diagnoses, and the PLE-HM score combines psychotic-like and hypomanic items. `[AUTHOR INPUT: M-01 instrument details]`
- **Concurrent predictors.** In the main analysis, adverse experiences and substance use at age 18 were measured at the same time as the outcomes. `[TODO: state how results changed when they were removed]`
- **Twin dependence.** The mixed models included a twin-pair random intercept, but the network and the machine-learning baselines did not model twin dependence during training; splits were grouped by twin pair for all models.
- **Generalisability.** A single cohort with no external validation; PRS derived from European-ancestry genome-wide association studies `[VERIFY: GWAS ancestry — W-03]`.
- **Power.** Five splits give limited power to detect small differences between models.

# Conclusions

In this Swedish twin cohort, a deep fusion network designed around interactions between adolescent substance use and other risk domains did not predict symptoms at age 18 better than mixed-effects models with matched inputs and interactions, and penalised linear models did best. The data gave little evidence of the interactions that motivated the network. `[TODO: one sentence on attribution agreement once available]` Deep models for psychiatric risk prediction in cohort data should be compared with well-specified regression models on identical, family-aware splits, with model selection kept away from the test data, before their added complexity or their attention-based explanations are relied on.
