<!--
STATUS 2026-10-25: Background for BMC Psychiatry (replaces 02_introduction.md, written for AIIM). Gap statement rewritten with the L-02 studies (reference_registry.md §G).
- BMC research articles open with "Background"; the template asks for no subheadings in this section.
- Framing: neutral benchmark (DCMFNet vs matched mixed models) plus a check of attention-based explanations.
  The planted-interaction simulation, learning curves and stream ablation (E-16, E-14, E-06 ablation part) were
  not run and are not promised here.
- Citations use pandoc keys [@R-...] from literature/reference_registry.md (and latex/references.bib).
  A key may support a claim only if its "Content access" covers it; otherwise [VERIFY: content].
- Outcome labels are working labels pending M-01.
-->

# Background

Psychotic-like experiences, hypomanic symptoms and depressive symptoms are common in late adolescence. Most are transient, but they are not benign: in the cohort studied here, psychotic experiences in adolescence were associated with later substance use disorder and suicide attempts [@R-CEDERLOF]. Risk factors for these symptoms and for the psychotic disorders they may precede span several domains: substance use, particularly cannabis [@R-DIFORTI; @R-HALL]; childhood adversity [@R-VARESE]; socioeconomic circumstances [@R-AGERBO]; and inherited liability captured by polygenic risk scores (PRS) [@R-PGC3]. <!-- content checked 2026-10-26 (abstracts): R-DIFORTI and R-AGERBO concern psychotic disorder/schizophrenia, hence 'and for the psychotic disorders'; R-PGC3 is the SCZ GWAS underlying PRS --> Mapping psychiatric genetic signal onto specific brain cell types [@R-YAO2025] now allows cell-type-specific PRS, which may carry more targeted information than a genome-wide score.

A long-standing hypothesis holds that genetic and environmental risk factors act jointly rather than additively, although such interactions remain poorly understood [@R-ROBINSON]. Applied to the domains above, genetic liability or early adversity would change the effect of substance use. <!-- R-ROBINSON abstract checked 2026-10-26: review of environmental risk factors (incl. cannabis, childhood adversity) and GxE for SCZ/BD --> If such interactions are strong, models that represent them should predict symptoms better than additive models. Deep multimodal fusion networks are one way to do this: they combine groups of inputs ("modalities") through learned non-linear interactions, and they are increasingly applied to heterogeneous health data [@R-BORSOS]. A fusion architecture in which one exposure domain conditions the representation of every other domain, such as the iterative gated fusion of the Deep Cross-Modal Fusion Network (DCMFNet) [@R-DCMF], matches the interaction hypothesis directly. Such networks also contain attention-like gates, which are often read as indicators of which inputs matter.

Matching the hypothesis does not guarantee better predictions. In clinical prediction, a systematic review found no performance benefit of machine learning over logistic regression in comparisons at low risk of bias, and calibration was rarely assessed [@R-CHRIST]. On medium-sized tabular data, tree ensembles usually outperform deep networks [@R-GRIN]. Studies of adolescent mental health point the same way:
- **In the cohort used here,** random forests, support vector machines, gradient boosting and a neural network predicted mental-health problems at age 15 no better than logistic regression [@R-TATE].
- **In a US cohort,** gradient boosting improved the prediction of depressive symptoms over logistic regression only slightly, and polygenic scores added nothing to environmental predictors [@R-GEURGAS].
- **For psychotic experiences in adolescents,** polygenic and polyenvironmental risk scores did not predict the outcome [@R-NAVARRO].
- **In a large child cohort,** earlier symptoms were the strongest predictors of later mental-health problems, and neuroimaging added little [@R-HOU].

Where multimodal prediction of psychosis has worked well, in patients at clinical high risk, separate clinical, polygenic and imaging models were combined step by step rather than fused within one network [@R-KOUTS].

To our knowledge, no study has tested whether a deep fusion network built around substance-use interactions, using polygenic and environmental predictors in a population cohort, predicts better than regression models given the same inputs and interactions. <!-- VERIFIED 2026-10-28 (L-02): PubMed search, title/abstract blocks for deep/fusion models (deep learning, neural network*, multimodal/multi-modal fusion, multimodal deep, cross-modal, attention mechanism*) × psychotic/hypomanic/depressive outcomes × PRS/substance use/cannabis × adolescent/youth/cohort/twin: 9 hits; adding a regression/mixed-model block gave 2 (both among the 9). Closest records read in full by the author (PMIDs 41399528, 36411277, 34899225): none tested this comparison. Non-systematic search. --> Such a test also has to avoid artefacts that favour the more flexible model:
- model selection on test data;
- feature screening outside the training data [@R-KAPOOR];
- related individuals on both sides of a split [@R-ROBERTS];
- unequal tuning effort;
- comparators that lack the interactions the flexible model is credited with.

Finally, attention weights are not necessarily faithful explanations of a model's predictions [@R-JAIN]. Gate values therefore need to be checked against attribution methods that measure what the model relies on.

We therefore compared DCMFNet, adapted so that substance use at age 15 is fused with each other risk domain, against two kinds of comparator:
- linear and generalised additive mixed models given the same predictors, a twin-pair random intercept and interaction terms that mirror the network's fusion design;
- three standard machine-learning models (linear regression, an elastic net and gradient boosting) tuned within the training data.

All models were evaluated on identical twin-pair-grouped splits in the Child and Adolescent Twin Study in Sweden [@R-CATSS], with outcomes at age 18. We asked:
1. whether the fusion network predicted better than these comparators, including calibration, performance by sex and models restricted to predictors measured by age 15;
2. whether the mixed models showed evidence of the substance-use interactions that motivate the fusion design;
3. whether the network's attention gates agreed with SHAP values and permutation importance computed on the same models, and with exact SHAP values from the mixed models.

<!-- Do NOT add: superiority claims; "interpretable" claims based on gate values (D-12); causal or "amplifier" language. -->
