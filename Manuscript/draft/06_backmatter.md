<!--
STATUS 2026-10-23: Back matter for BMC Psychiatry (replaces 07_declarations.md, written for AIIM).
BMC research articles end with: List of abbreviations; Declarations (the headings below, "Not applicable" where relevant);
References; figure legends; tables; additional files. [VERIFY: JOURNAL REQUIREMENT — headings taken from the BMC
series declarations format; the BMC Psychiatry page itself could not be fetched. "Clinical trial number" was added to
BMC declarations in 2025 — confirm it is required for observational studies.]
Nothing here may be invented: every [AUTHOR INPUT] needs the author's own information.
-->

# Abbreviations

ACE: adverse childhood experiences; ADHD: attention-deficit/hyperactivity disorder; ASD: autism spectrum disorder; CATSS: Child and Adolescent Twin Study in Sweden; DCMFNet: Deep Cross-Modal Fusion Network; DEP: depressive symptoms; GAMM: generalised additive mixed model; IGF: iterative gated fusion; LMM: linear mixed model; MAE: mean absolute error; PC: principal component; PLE-HM: psychotic-like and hypomanic experiences; PRS: polygenic risk score; RMSE: root-mean-squared error; SES: socioeconomic status; SHAP: Shapley additive explanations; SUD: substance use; TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis, updated for artificial intelligence.
<!-- Update if the comparator is called "GLMM" in the final text (Methods AUTHOR INPUT). -->

# Supplementary information

- **Additional file 1** (PDF): Table S1, Optuna search space; Table S2, all 40 machine-learning baselines; Table S3, full LMM and GAMM coefficients; Table S4, variable list by predictor group; Table S5, depth-configuration results (5 seeds); tuning history (D-21). `[TODO]`
- **Additional file 2** (PDF): TRIPOD+AI checklist. `[TODO: C-06]`

# Declarations

## Ethics approval and consent to participate
`[AUTHOR INPUT: "CATSS and the genotype linkage were approved by [committee, e.g., the Regional Ethical Review Board in Stockholm / Swedish Ethical Review Authority], reference number(s) [xxx], date(s) [xxx]. [Consent procedure: parents for participants under 15/18; participants themselves at age 18 — VERIFY]. The study was conducted in accordance with the Declaration of Helsinki."]` — TODO C-02

## Consent for publication
Not applicable. <!-- No individual-level data, images or case details are reported. -->

## Availability of data and materials
Draft: "The CATSS data contain sensitive personal information and cannot be shared publicly under Swedish data-protection law. Researchers can apply for access through the Swedish Twin Registry, Karolinska Institutet `[AUTHOR INPUT: exact procedure/URL]`. The analysis code is available at `[TODO: repository release + DOI — C-05]`."

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

**Fig. 1** Participant flow from the CATSS cohort to the complete-case analysis sample. `[TODO: from analysis_sample_flow.csv — W-02]`

**Fig. 2** Architecture of the deep cross-modal fusion network. Substance use at age 15 (anchor) is fused with each of nine predictor groups in iterative gated fusion blocks (fused stream); an independent stream receives all inputs. `[TODO: diagram regenerated from code — C-04]`

**Fig. 3** Calibration curves for each model and outcome, by decile of predicted score, pooled over the five test sets. `[TODO: calibration_curves_full.pdf from prediction_evaluation.py]`

**Fig. 4** Predictor-group importance by source: DCMFNet gate values, DCMFNet SHAP, DCMFNet permutation importance, LMM SHAP and LMM permutation importance. `[TODO: from attribution_agreement.py]`
