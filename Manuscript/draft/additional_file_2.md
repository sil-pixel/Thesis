# Additional file 2. TRIPOD+AI checklist

<!--
STATUS 2026-10-27: first complete pass (C-06). Item list from Collins et al., BMJ 2024;385:e078378, Table 2
(PMC11019967, read 2026-10-27); item descriptions are shortened paraphrases, not the official wording. Before
submission, transfer the "Reported in" column to the official checklist form (tripod-statement.org) with page numbers.
Status: Reported / Partly (gap noted) / Author input (placeholder in the draft) / Not applicable (reason given).
This study develops and internally evaluates several models on the same data and compares them; it does not propose
a model for use, which makes several implementation items not applicable.
-->

Reporting checklist for the TRIPOD+AI statement [@R-TRIPODAI]. D, development; E, evaluation. Sections refer to the main manuscript.

| Section | Item | D/E | Topic (shortened) | Reported in | Status |
|--------------|----|----|--------------------|--------------------|----------------------------------------|
| Title | 1 | D;E | Study type, population, outcome | Title | Reported |
| Abstract | 2 | D;E | TRIPOD+AI for Abstracts | Abstract | Reported (structured abstract; check against the abstract checklist before submission) |
| Background | 3a | D;E | Context and rationale; existing models | Background | Reported |
| | 3b | D;E | Target population, intended purpose and users | Background (aims); Conclusions | Reported: a methodological comparison in a population cohort; the Conclusions state that the models are not intended for clinical use |
| | 3c | D;E | Known health inequalities | – | Not reported: sex differences are analysed (Results), but known inequalities are not discussed in the Background |
| Objectives | 4 | D;E | Objectives; development and/or validation | Background (final paragraph) | Reported |
| Data | 5a | D;E | Data sources, rationale, representativeness | Methods: Study population; Results: Table 1 | Reported (included vs excluded comparison) |
| | 5b | D;E | Dates of data collection and follow-up | Methods: Study population | Partly: access period reported; collection years and birth cohorts are author input (M-06) |
| Participants | 6a | D;E | Setting, centres | Methods: Study population | Reported (population-based national twin cohort) |
| | 6b | D;E | Eligibility criteria | Methods: Study population; Fig. 1 | Reported |
| | 6c | D;E | Treatments | – | Not applicable: treatment data were not available in the analysis data and not used |
| Data preparation | 7 | D;E | Pre-processing and quality checks | Methods: Predictors (variable screening, missing codes) | Reported; similarity across sociodemographic groups not assessed |
| Outcome | 8a | D;E | Outcome definition, timing, rationale | Methods: Outcomes | Reported (instruments, response scales and scoring) |
| | 8b | D;E | Qualifications of outcome assessors | – | Not applicable: self-reported questionnaires |
| | 8c | D;E | Blinding of outcome assessment | – | Not applicable: self-report |
| Predictors | 9a | D | Choice and pre-selection of predictors | Methods: Predictors (variable screening); Additional file 1: Table S4 | Reported (outcome-informed screening disclosed as a limitation) |
| | 9b | D;E | Definition and timing of predictors | Methods: Predictors; Additional file 1: Table S4 | Partly: age-9 instruments and informants are author input (M-06) |
| | 9c | D;E | Qualifications of predictor assessors | – | Not applicable: questionnaire and genotype data |
| Sample size | 10 | D;E | How study size was arrived at, and adequacy | Methods: Study population; Discussion | Reported: set by the available complete cases, no formal calculation; the Discussion relates sample size to model complexity |
| Missing data | 11 | D;E | Handling of missing data | Methods: Study population; Fig. 1; Discussion (limitations) | Reported (complete-case analysis) |
| Analytical methods | 12a | D | Use and partitioning of data | Methods: Data partitioning | Reported |
| | 12b | D | Handling of predictors (form, scaling) | Methods: DCMFNet, mixed-effects and machine-learning comparators | Reported |
| | 12c | D | Model type, building steps, tuning, internal validation | Methods; Table 2; Additional file 1: Tables S1, S2b, S5 | Reported (test-set involvement in DCMFNet configuration disclosed) |
| | 12d | D;E | Heterogeneity across clusters | Methods: Data partitioning, mixed-effects comparators | Reported (twin pairs as clusters: grouped splits; random intercept in the mixed models) |
| | 12e | D;E | Performance measures and model comparison | Methods: Evaluation and statistical comparison (incl. interaction ablation); Calibration, subgroup and sensitivity analyses | Reported |
| | 12f | E | Model updating | – | Not applicable: no recalibration or updating |
| | 12g | E | How predictions were calculated | Methods: Software and reporting | Reported: code archived with a DOI |
| Class imbalance | 13 | D;E | Class-imbalance methods | Methods: Loss | Reported (continuous outcomes; tail-weighted loss in DCMFNet) |
| Fairness | 14 | D;E | Approaches to fairness | Methods: Calibration, subgroup and sensitivity analyses | Partly: performance by sex reported; no fairness intervention |
| Model output | 15 | D | Output and any thresholds | Methods: Outcomes | Reported (continuous scores, no thresholds) |
| Training vs evaluation | 16 | D;E | Differences between development and evaluation data | Methods: Data partitioning | Reported (random splits of one cohort; no differences by design) |
| Ethical approval | 17 | D;E | Ethics committee and consent | Methods: Ethics; Declarations | Author input (C-02) |
| Funding | 18a | D;E | Funding and role of funders | Declarations | Author input (C-03) |
| Conflicts of interest | 18b | D;E | Competing interests | Declarations | Author input (C-03) |
| Protocol | 18c | D;E | Protocol access | Methods: Software and reporting | Reported: no protocol was published |
| Registration | 18d | D;E | Study registration | Methods: Software and reporting | Reported: not registered |
| Data sharing | 18e | D;E | Data availability | Declarations: Availability of data and materials | Partly: access procedure is author input (C-07) |
| Code sharing | 18f | D;E | Code availability | Methods: Software and reporting; Declarations | Reported: Zenodo DOI and GitHub repository |
| Patient and public involvement | 19 | D;E | Patient and public involvement | Methods: Software and reporting | Reported: no involvement |
| Participants | 20a | D;E | Participant flow | Results: Analysis sample; Fig. 1 | Reported |
| | 20b | D;E | Characteristics and missing data | Results: Table 1 | Reported |
| | 20c | E | Comparison with development data | – | Not applicable: internal evaluation on random splits of the same cohort |
| Model development | 21 | D;E | Numbers in each analysis | Methods: Data partitioning; Fig. 1 | Reported (training, validation and test sizes) |
| Model specification | 22 | D | Full model available for new predictions | Methods; Table 2; Additional file 1: Table S3 | Partly: mixed-model coefficients and all code given; network weights trained on CATSS are not shared |
| Model performance | 23a | D;E | Performance with confidence intervals, incl. subgroups | Results: Tables 3–6 | Reported: mean and SD over splits for each model; paired differences with 95% confidence intervals and corrected p values (Tables 4–5); results by sex (Table 6) |
| | 23b | D;E | Heterogeneity across clusters | – | Not applicable: twin pairs are too small for cluster-level performance |
| Model updating | 24 | E | Results of updating | – | Not applicable |
| Interpretation | 25 | D;E | Overall interpretation, incl. fairness | Discussion | Reported |
| Limitations | 26 | D;E | Limitations and their effects | Discussion: Strengths and limitations | Reported |
| Usability | 27a | D | Handling poor or missing input data in use | – | Not applicable: no model is proposed for use |
| | 27b | D | User interaction and expertise | – | Not applicable: no model is proposed for use |
| | 27c | D;E | Next steps for research | Discussion; Conclusions | Reported |
