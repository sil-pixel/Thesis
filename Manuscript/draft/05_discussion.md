<!--
STATUS 2026-10-27: Full prose draft for BMC Psychiatry (Discussion with strengths and limitations, then Conclusions), written
from the verified results in 04_results.md. Numbers re-checked against the CSVs 2026-10-27 (W-10). Remaining: the L-02 PubMed/Scopus confirmation
and the [VERIFY] markers. No causal language; "no detectable gain", not "equivalent"; no clinical-utility claims.
src for the numbers: prediction_metrics_summary.csv, prediction_metrics_by_sex_summary.csv, paired_comparisons.csv,
attribution_modality_summary.csv, attribution_agreement_summary.csv, attribution_stability.csv, ML_comparators_tuning.csv.
-->

# Discussion

In a population-based twin cohort, a deep cross-modal fusion network built around adolescent substance use did not predict psychotic-like and hypomanic experiences or depressive symptoms at age 18 better than linear and additive mixed models that were given the same predictors and interaction terms mirroring the network's fusion design. Mean test R² was about 0.02 lower for the network than for the mixed models for PLE-HM and similar for DEP, and no paired difference was statistically significant. Machine-learning comparators tuned within the training data did better than all three models, and an elastic net did best; DCMFNet was behind each comparator on almost every split, although no difference remained significant after correction. The mixed models also showed little evidence of the substance-use interactions that motivate the fusion design: the number of nominally significant interaction terms was close to the number expected by chance, and most PRS effects were close to linear. All models were reasonably well calibrated, and removing the predictors measured at age 18 lowered R² by 0.05–0.07 in every model without changing their ranking. The network's gate values were nearly uniform across predictor groups and agreed poorly with SHAP values and permutation importance computed on the same models.

These findings are consistent with comparisons in clinical prediction, where flexible machine-learning models rarely outperform regression when the comparison is at low risk of bias [@R-CHRIST], and with benchmarks showing that deep networks are seldom the best choice for medium-sized tabular data [@R-GRIN]. They also agree with earlier machine-learning work in this cohort: random forests, support vector machines, gradient boosting and a neural network did not predict parent-rated mental health problems at age 15 significantly better than logistic regression [@R-TATE]. Two features of our design make the result conservative for the network. First, part of the network's configuration was chosen with test-set information, which can only favour it, whereas the machine-learning comparators were tuned on training data alone. Second, the mixed models were given the same interaction structure as the network, so the comparison isolates how the interactions are modelled rather than whether they are included. 

Few studies have compared flexible models with regression for these outcomes in population cohorts, and we found none that used a deep fusion architecture with polygenic and environmental predictors. <!-- VERIFIED 2026-10-28 (L-02): PubMed search, title/abstract blocks for deep/fusion models (deep learning, neural network*, multimodal/multi-modal fusion, multimodal deep, cross-modal, attention mechanism*) × psychotic/hypomanic/depressive outcomes × PRS/substance use/cannabis × adolescent/youth/cohort/twin: 9 hits; adding a regression/mixed-model block gave 2 (both among the 9). Closest records read in full by the author (PMIDs 41399528, 36411277, 34899225): none tested this comparison. Non-systematic search. --> The studies we found point in the same direction as ours:
- **Depression, Add Health cohort:** gradient boosting improved discrimination of depressive symptoms by about 0.02 AUC over logistic regression. Adding polygenic scores to environmental predictors did not improve prediction [@R-GEURGAS].
- **Psychotic experiences, Brazilian youth cohort:** polygenic and polyenvironmental risk scores, alone or combined, did not predict psychotic experiences in adolescents [@R-NAVARRO].
- **Mental health problems, ABCD study:** earlier subclinical symptoms predicted later problems, and neuroimaging added little [@R-HOU].
- **Psychosis, clinical high-risk patients:** multimodal prediction performed well where clinical, polygenic and imaging information was combined. There, the risk estimates of separate models were integrated step by step, rather than fused in a single network [@R-KOUTS].

Our attribution results fit this pattern. Symptoms measured earlier carried most of the predictive information, and polygenic scores contributed little.

Several explanations are possible. The training sets contained about 3,000 individuals, while the networks had 72,000 to 106,000 trainable parameters (Table 2). Flexible models need far more data than regression to reach stable performance [@R-PLOEG], so at this sample size they are likely to fit noise rather than signal. The predictors and outcomes are questionnaire sum scores and PRS, which explain a limited share of variance, and this limits the attainable R² for any model. Finally, if interactions between substance use and other domains are weak at the population level, a model designed to capture them has little to gain. Some observations fit this last explanation:
- gradient boosting, which could learn interactions freely, selected shallow trees in most splits;
- the elastic net selected strong, ridge-like shrinkage;
- removing the 24 interaction terms from the mixed models raised test R² on every split, to the level of linear regression, so the interactions slightly reduced out-of-sample accuracy, as overfitting would (Table 5);
- removing the fusion blocks from the network changed its R² inconsistently across splits, by about 0.01–0.02 on average in favour of fusion, with confidence intervals that included 0. <!-- src: interaction_ablation_summary.csv -->

Our results cannot distinguish between these explanations. They also do not show that such interactions are absent: prediction accuracy is a different question from testing moderation, and the interaction tests here had limited power after correction for multiple testing. The network without fusion blocks also kept the hyperparameters tuned for the full network, which, if anything, favours the full network in this comparison.

The network's gate values should not be read as measures of predictor importance.
- **The gates hardly separated the groups.** In the fused stream they were close to the midpoint of the logistic gate (0.50) for every predictor group.
- **They were unstable.** Their ranking of the groups changed from split to split.
- **They disagreed with the model-agnostic attributions.** They agreed poorly with SHAP values and permutation importance computed on the same trained networks: mean Spearman ρ 0.04 for PLE-HM and 0.36 for DEP for the fused gates.

This agrees with work showing that attention weights need not reflect what a model relies on [@R-JAIN]. In this network, the gates scale each individual's features. A value near 0.50 for every group means that this scaling did little to distinguish between groups.

The model-agnostic attributions, in contrast, were stable across splits and broadly consistent between the network and the mixed models.
- **In all models,** symptoms at age 15 contributed most, followed by adverse experiences and substance use at age 18.
- **Substance use at age 15,** the anchor of the fusion design, contributed only about 5% of SHAP importance (4.7–6.1% across models and outcomes).

The network thus relied on much the same predictors as the linear models, not mainly on its anchor domain, which fits the absence of a performance gain. These attributions describe what the fitted models used for prediction. They do not identify causal risk factors, and the age-18 predictors were measured at the same time as the outcomes.

The secondary analyses did not change the main conclusion.
- **Calibration.** All models predicted the mean outcome well, although the network slightly over-predicted depressive symptoms. The network, the mixed models and linear regression produced predictions that were spread somewhat too widely (calibration slopes 0.82–0.89). The tuned elastic net and gradient boosting had slopes close to 1.
- **The highest scores.** Every model under-predicted the scores of the 10% of participants with the highest scores. The network's weighted loss, designed to give more weight to the upper tail, produced the smallest under-prediction. It did not, however, identify this group better (AUROC 0.76, against 0.76–0.78 for the other models).
- **Predictors measured by age 15.** Without the age-18 predictors, R² fell by 0.05–0.07 in every model, and the ranking of the models did not change. A substantial part of the performance in the main analysis therefore came from measures taken at the same time as the outcomes. With predictors available by age 15, R² was 0.13–0.20.
- **Sex.** All models explained less variance in depressive symptoms in males than in females. With age-15 predictors only, the network explained almost none of the variance in males (R² 0.01), whereas the elastic net reached 0.10. In the network the age-18 inputs were held at their training mean, not removed, which may contribute to this difference. These sex differences were not formally tested.

## Strengths and limitations

Strengths of this study include a population-based twin cohort with genetic and multi-informant questionnaire data; evaluation of all models on identical twin-pair-grouped splits with the same metric code; comparators that matched the network's inputs and interaction structure; paired inference that accounts for overlapping training sets; and analyses of calibration, sex, predictors measured before the outcome and attribution.

The study also has limitations.

- **Configuration selection.** The network's depth configuration, and the choice among tuning runs, used test-set results, and the tuning data overlapped later test sets. The network's own performance estimates are therefore likely optimistic, although this bias works against our main conclusion.
- **Feature screening.** The variable-screening step used the outcomes on data that overlap the test sets [@R-KAPOOR]. This affects all models equally but may inflate absolute performance.
- **Complete-case analysis.** Only individuals with complete data were analysed, 43.5% of the eligible sample, mainly because genotype data were missing. Included participants had slightly lower outcome and age-15 symptom scores than excluded participants (standardised differences up to 0.22), so the sample may under-represent adolescents with more symptoms, and results may not generalise to the full cohort.
- **Outcome measures.** The outcomes are continuous questionnaire scores, not diagnoses, and the PLE-HM score combines psychotic-like and hypomanic items.
- **Concurrent predictors.** In the main analysis, adverse experiences and substance use at age 18 were measured at the same time as the outcomes. Removing them lowered R² by 0.05–0.07 in every model without changing the ranking of the models, so the main-analysis estimates overstate what can be predicted from information available at age 15.
- **Twin dependence.** The mixed models included a twin-pair random intercept, but the network and the machine-learning comparators did not model twin dependence during training; splits, and the comparators' inner tuning folds, were grouped by twin pair.
- **Generalisability.** A single cohort with no external validation; PRS derived from European-ancestry genome-wide association studies `[VERIFY: GWAS ancestry]` <!-- W-03 -->.
- **Power.** Five splits give limited power to detect small differences between models.

# Conclusions

In this Swedish twin cohort, a deep fusion network designed around interactions between adolescent substance use and other risk domains did not predict symptoms at age 18 better than mixed-effects models with matched inputs and interactions, and an elastic net and gradient boosting tuned on training data alone did better than all three models. The data gave little evidence of the interactions that motivated the network. The network's gate values did not reflect what it relied on for prediction and should not be interpreted as predictor importance. The models were developed to compare modelling approaches, not for clinical use. Deep models for psychiatric risk prediction in cohort data should be compared with well-specified regression models on identical, family-aware splits, with model selection kept away from the test data, before their added complexity or their attention-based explanations are relied on.
