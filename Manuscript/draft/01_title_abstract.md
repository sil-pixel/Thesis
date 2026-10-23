# Title page, abstract, keywords

<!-- STATUS 2026-10-23: restructured for BMC Psychiatry (Springer Nature sn-jnl LaTeX template in Manuscript/sn-article-template).
     BMC research-article format: structured abstract (Background, Methods, Results, Conclusions), up to 350 words [VERIFY: JOURNAL REQUIREMENT — limit seen on BMC series pages, the BMC Psychiatry page itself was not reachable]; 3–10 keywords.
     Outcome labels are working labels pending M-01. Numbers come from 04_results.md. -->

## Title
Does deep cross-modal fusion improve prediction of late-adolescent psychopathology? A comparison with mixed-effects models in a Swedish twin cohort
<!-- Alternative: "Deep multimodal fusion versus mixed-effects models for predicting psychotic-like, hypomanic and depressive symptoms at age 18: a twin cohort study". Avoid "DCMFNet" in the title (not an established abbreviation). [AUTHOR INPUT: choose] -->

## Authors and affiliations
`[AUTHOR INPUT: full names in submission order; affiliations (department, organisation, city, postcode, country); corresponding author and email]`
<!-- Thesis lists: student author; supervisor; co-supervisor; examiner (examiners are normally not authors). Every author must meet the authorship criteria in the Springer Nature editorial policies. -->

## Abstract
<!-- src: 04_results.md (GLMM_updated_summary.csv, GAMM_updated_summary.csv, Final Best Model/test_results_summary.csv, per-seed CSVs; ML: Models/ML/Results/{Pos,Neg}_benchmark_summary.csv). Word count without headings and TODO markers: ~329 (2026-10-23); leaves ~20 words for the pending results under a 350-word limit, so merge the two TODO sentences into one or trim elsewhere. -->

**Background:** Deep multimodal fusion networks can model interactions between psychiatric risk domains, but whether they predict better than well-specified regression models is rarely tested on identical data. We compared a deep cross-modal fusion network, in which adolescent substance use is fused with each other risk domain, against mixed-effects models with matched inputs and interactions.

**Methods:** We analysed 4,033 twins (2,690 pairs) from the Child and Adolescent Twin Study in Sweden with complete data on 105 predictors in 11 groups, including substance use at age 15, neurodevelopmental traits, adverse experiences and 16 cell-type-specific polygenic scores. Outcomes were continuous scores of psychotic-like and hypomanic experiences and of depressive symptoms at age 18. Comparators were a linear and a generalised additive mixed model, each with a twin-pair random intercept and 24 substance-use interaction terms mirroring the network's fusion design. All models were evaluated on the same five twin-pair-grouped splits, with corrected resampled t-tests. We also assessed calibration, performance by sex, models without age-18 predictors, and whether the network's attention gates agreed with SHAP values and permutation importance.

**Results:** For psychotic-like and hypomanic experiences, mean test R² was 0.206 (SD 0.019) for the network, 0.228 (0.016) for the linear mixed model and 0.227 (0.015) for the additive mixed model. For depressive symptoms, it was 0.198 (0.021), 0.202 (0.022) and 0.200 (0.023). No paired difference was statistically significant. A cross-validated lasso with default settings reached 0.261 and 0.234. Of the 24 interaction terms, three and one had unadjusted p < 0.05, about the number expected by chance, and none remained significant after correction. `[TODO: one sentence on calibration, sex and age-15-only results — prediction_evaluation.py]` `[TODO: one sentence on gate–SHAP–permutation agreement — attribution_agreement.py]`

**Conclusions:** In this twin cohort, the fusion network did not predict better than mixed-effects models with matched inputs and interactions, even though only the network was tuned and part of its configuration was chosen using test data. The data gave little evidence of interactions between adolescent substance use and other risk domains. Deep psychiatric prediction models should be benchmarked against well-specified regression models on identical splits before their complexity or attention-based explanations are relied on.

## Keywords
Psychotic-like experiences; Depressive symptoms; Adolescent substance use; Polygenic risk scores; Deep learning; Multimodal fusion; Mixed-effects models; Prediction model; Twin study
<!-- BMC: 3–10 keywords [VERIFY: JOURNAL REQUIREMENT]. [AUTHOR INPUT: confirm] -->
