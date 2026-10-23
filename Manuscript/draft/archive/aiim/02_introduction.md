<!--
STATUS 2026-10-19: First draft under framing Option B (author decision X-00, 2026-10-19).
- Outcome labels are WORKING labels pending M-01 ("psychotic-like and hypomanic symptoms"; "depressive symptoms"). The thesis labels "positive/negative symptoms" are not used (D-03).
- Contribution statements (final paragraph) are conditional on the approved experiments. Any contribution whose experiment is not run must be deleted, not softened.
- Citation rule: a reference is cited for a specific claim only if its content access in literature/reference_registry.md covers that claim. Otherwise it is marked [VERIFY: content].
- No numerical results appear here on purpose: the current comparative numbers are not valid for claims (E-01–E-03, A-01).
-->

# 1. Introduction

Psychotic-like experiences, hypomanic symptoms and depressive symptoms are common in late adolescence. Although most are transient, they are linked to later psychiatric morbidity and to help-seeking `[CITATION NEEDED: adolescent PLEs/subclinical symptoms and later psychopathology; candidate Cederlöf/Kelleher et al. 2017, Schizophr Res, doi 10.1016/j.schres.2016.08.029 — content unverified]`. Their established correlates span several domains:
- adolescent substance use, particularly cannabis [R-DIFORTI, R-HALL] `[VERIFY: content of R-DIFORTI beyond the association with psychotic disorder incidence]`;
- childhood adversity [R-VARESE];
- socioeconomic circumstances [R-AGERBO] `[VERIFY: content]`;
- inherited liability captured by polygenic risk scores (PRS) derived from large genome-wide association studies [R-PGC3] `[VERIFY: content]`.

Work linking psychiatric GWAS signal to specific brain cell types [R-YAO2025] `[VERIFY: content]` now makes it possible to build cell-type-specific PRS. These scores could carry more biologically targeted information than a genome-wide score. A long-standing hypothesis holds that these domains act jointly rather than additively, for example that genetic liability or early adversity modifies the effect of substance use [R-ROBINSON] `[VERIFY: content]`.

<!-- src: synthetic_data/agents/researcher.md (outcome naming rule via D-03); discrepancies.md D-03 -->

This hypothesis has a direct implication for prediction modelling. If cross-domain interactions carry substantial signal, models that represent them explicitly should predict symptom outcomes better than additive models. Several model families can capture such interactions:
- mixed models with pre-specified product terms;
- tree ensembles that learn interactions implicitly;
- deep architectures that fuse groups of inputs.

The last family has developed quickly. Multimodal and group-wise fusion networks are increasingly applied to heterogeneous health data, including tabular clinical variables combined with imaging [R-BORSOS]. Architectures built for cross-modal fusion in other domains, such as iterative gated fusion [R-DCMF], can be adapted so that one exposure domain conditions the representation of every other domain. The result is a model whose inductive bias matches the interaction hypothesis.

A matching inductive bias does not guarantee better predictions. In clinical prediction, a systematic review of comparisons between machine learning and logistic regression found no performance benefit for machine learning when comparisons were at low risk of bias, and it found that calibration was rarely assessed [R-CHRIST]. On medium-sized tabular datasets, benchmark studies report that tree-based ensembles generally outperform deep learning models [R-GRIN], and similar conclusions have been reported elsewhere [R-SHWARTZ] `[VERIFY: content]`. Prior machine-learning work on the cohort used here, which predicted adolescent mental-health problems, found performance comparable to regression [R-TATE] `[VERIFY: read abstract/full text before citing this finding]`.

Comparisons between model classes are also vulnerable to methodological artefacts that favour the more flexible model:
- hyperparameters or architectures selected on data later used for testing;
- feature screening outside the training folds [R-KAPOOR] `[VERIFY: content]`;
- related individuals, such as co-twins, placed on both sides of a split `[CITATION NEEDED: family/twin dependence and grouped cross-validation]`;
- unequal tuning effort across model families.

The number of parameters that can be estimated reliably is also limited by the sample size and the expected signal [R-RILEY] `[VERIFY: content]`. Deep fusion models can exceed this limit by orders of magnitude in cohorts of a few thousand participants.

<!-- Gap statement: derived from research/contribution_matrix.md §3. Must be re-checked after targeted competitor search L-02. -->
It therefore remains unclear whether explicitly modelling cross-domain interactions improves the prediction of adolescent symptom outcomes from cohort questionnaire and polygenic data. It is also unclear why it fails when it does. A null result could have two different causes:
1. the model class cannot learn interactions of realistic size from the available sample;
2. the data contain little interaction signal beyond the additive effects.

Standard benchmarks cannot tell these explanations apart, yet they have different consequences for model choice and for the substantive interaction hypothesis. `[CITATION NEEDED: confirm via L-02 that no prior study evaluates interaction-focused deep fusion against additive models for these outcomes with cell-type PRS — gap statement depends on this]`

In this study, we use data from a Swedish twin cohort [R-CATSS] to address both questions. The data include:
- adolescent questionnaire measures of substance use, adversity, neurodevelopmental and psychotic-like symptoms;
- socioeconomic indicators;
- cell-type-specific PRS.

<!-- src: Models/DCMFNet/Results/Final Best Model/best_model_training.log:1-3 (n, features) -->

Our contributions are as follows. `[TODO: keep only items whose experiments are approved and completed; delete the rest]`
1. **A twin-aware, leakage-controlled comparison** of an exposure-anchored deep fusion network, adapted from iterative gated fusion, against additive, penalised, interaction-specified mixed-effects and tree-based models. All models are evaluated on identical twin-pair-grouped partitions, model selection uses validation data only, and model differences are tested with paired inference. `[TODO: pending E-01, E-02, E-03, E-04, A-01/E-08]`
2. **Diagnostic analyses that explain the comparison:**
   - an ablation separating the fused interaction stream from the model's direct-input stream `[TODO: pending E-06]`;
   - learning curves relating performance to training-set size `[TODO: pending E-14]`;
   - a planted-interaction simulation that preserves the empirical predictor distribution and estimates how strong cross-domain interactions must be before the fusion model outperforms additive models `[TODO: pending E-16]`.
3. **Calibration, upper-tail performance and prospective-predictor analyses.** These sensitivity analyses restrict predictors to those measured before the outcome assessment `[TODO: pending E-05, E-09, E-10]`.
4. **A reusable evaluation protocol** for multimodal tabular models on twin or family cohort data, reported according to TRIPOD+AI [R-TRIPODAI]. `[TODO: Supplement; C-06]`

<!-- Do NOT add: any superiority claim; "interpretable" claims based on gate values (D-12); causal or "amplifier" language (C7). -->
