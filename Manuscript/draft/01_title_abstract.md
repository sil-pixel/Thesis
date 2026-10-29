# Title page, abstract, keywords

<!-- STATUS 2026-10-23: restructured for BMC Psychiatry (Springer Nature sn-jnl LaTeX template in Manuscript/sn-article-template).
     BMC research-article format: structured abstract (Background, Methods, Results, Conclusions), up to 350 words [VERIFY: JOURNAL REQUIREMENT — limit seen on BMC series pages, the BMC Psychiatry page itself was not reachable]; 3–10 keywords.
     Outcome labels are working labels pending M-01. Numbers come from 04_results.md. -->

## Title
Does deep cross-modal fusion improve prediction of late-adolescent psychopathology? A comparison with mixed-effects models in a Swedish twin cohort
<!-- Alternative: "Deep multimodal fusion versus mixed-effects models for predicting psychotic-like, hypomanic and depressive symptoms at age 18: a twin cohort study". Avoid "DCMFNet" in the title (not an established abbreviation). [AUTHOR INPUT: choose] -->

## Authors and affiliations
<!-- Author list and order given by the author, 2026-10-28; to be confirmed with the supervisors (supervisor_request.md, item 4). Format per line: given name(s) | family name | email | affiliation number(s), comma-separated. The build reads this list and the numbered "Affiliation N:" lines. -->
- Silpa Soni | Nallacheruvu | silpa.soni.nallacheruvu@stud.ki.se | 1
- Shuyang | Yao | shuyang.yao@ki.se | 1
- Sam | Andersson | sam.andersson@ki.se | 2
- Tong | Gong | tong.gong@ki.se | 1
- Lu | Yi | lu.yi@ki.se | 1

Affiliation 1: Department of Medical Epidemiology and Biostatistics, Karolinska Institutet, Stockholm, Sweden
Affiliation 2: Department of Clinical Neuroscience, Karolinska Institutet, Stockholm, Sweden
<!-- author, 2026-10-28 (affiliation 1, for the author; others to confirm) and 2026-10-29 (Sam Andersson: affiliation 2); supervisor_request.md, item 4 -->

`[AUTHOR INPUT: confirm the affiliation for each author, postcode, corresponding author, and the given/family-name split above]`
<!-- Every author must meet the authorship criteria in the Springer Nature editorial policies; contributions go in Declarations (CRediT). -->

## Abstract
<!-- src: 04_results.md (prediction_metrics_summary.csv, paired_comparisons.csv, ML_comparators_tuning.csv). Word count without headings: 348 (2026-10-25). -->

**Background:** Deep multimodal fusion networks can model interactions between psychiatric risk domains, but whether they predict better than well-specified regression models is rarely tested on identical data. We compared a deep cross-modal fusion network, which fuses adolescent substance use with each other risk domain, against regression and machine-learning models.

**Methods:** We analysed 4,033 twins (2,690 pairs) from the Child and Adolescent Twin Study in Sweden with complete data on 105 predictors in 11 groups, including substance use at age 15, neurodevelopmental traits, adverse experiences and 16 cell-type-specific polygenic scores. Outcomes were continuous scores of psychotic-like and hypomanic experiences and of depressive symptoms at age 18. Comparators were generalised linear and additive mixed models (GLMM, GAMM) with a twin-pair random intercept and 24 substance-use interactions mirroring the fusion design, and linear regression, an elastic net and gradient boosting tuned within the training data. All models were evaluated on the same five twin-pair-grouped splits, with corrected resampled t-tests. Secondary analyses covered calibration, sex, predictors measured by age 15, and the network's gate values.

**Results:** For psychotic-like and hypomanic experiences, mean test R² was 0.206 for the network, 0.228 for the GLMM and 0.227 for the GAMM; for depressive symptoms, it was 0.198, 0.202 and 0.200. No paired difference was statistically significant. An elastic net did best (0.260 and 0.237) and was ahead of the network on every split, but no difference survived correction. No interaction term survived correction. Without age-18 predictors, R² fell by 0.05–0.07 in every model. The network's gate values were nearly uniform and agreed poorly with SHAP values and permutation importance.

**Conclusions:** In this twin cohort, the fusion network did not predict better than mixed-effects models with matched inputs and interactions, even though part of its configuration was chosen using test data, and simpler models tuned on training data alone did better. The data gave little evidence of interactions between adolescent substance use and other risk domains, and the network's gates did not indicate predictor importance. Deep psychiatric prediction models should be benchmarked against well-specified regression models on identical splits before their complexity or attention-based explanations are relied on.

## Keywords
Psychotic-like experiences; Depressive symptoms; Adolescent substance use; Polygenic risk scores; Deep learning; Multimodal fusion; Mixed-effects models; Prediction model; Twin study
<!-- BMC: 3–10 keywords [VERIFY: JOURNAL REQUIREMENT]. [AUTHOR INPUT: confirm] -->
