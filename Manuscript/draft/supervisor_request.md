Subject: DCMFNet paper: draft coming, and a few inputs I need from you

Hi [supervisor names],

I'm turning my thesis on DCMFNet into a paper for BMC Psychiatry, with Diagnostic and Prognostic Research as the fallback if it is transferred. My access to the CATSS data ends on 28 October. All analyses that need the data have now been run, and the retrained DCMFNet reproduces the thesis results. I plan to send you a full draft by [date] so you can review it and suggest edits. **If you would like any further analysis that needs the data, please tell me by [date before 28 October]**, since I can't run anything after that.

**Main finding.** On the same five twin-pair-grouped splits, DCMFNet did not predict better than the linear or additive mixed models (mean test R²):

| | PLE-HM | Depressive symptoms |
|---|---|---|
| DCMFNet | 0.206 | 0.198 |
| GLMM (linear mixed model) | 0.228 | 0.202 |
| GAMM (additive mixed model) | 0.227 | 0.200 |

No paired difference was significant.

Simpler machine-learning models, tuned only on the training data with folds grouped by twin pair, did better than all three:

| | PLE-HM | Depressive symptoms |
|---|---|---|
| Elastic net | 0.260 | 0.237 |
| Gradient boosting | 0.244 | 0.229 |
| Linear regression | 0.239 | 0.215 |

DCMFNet was behind the elastic net and boosting on every split, although with five splits no difference survived correction. I chose these three comparators by model class, after a default-settings screening of 40 algorithms (which goes in the supplement), and not by their rank in it. So the paper is framed as a careful benchmark of deep multimodal fusion against regression and standard machine-learning models, not as a new-method paper.

**Other results you should know about before reading the draft:**
- **Interactions.** None of the 24 substance-use interaction terms in the mixed models survived correction.
- **Predictors measured at age 18.** Leaving out adverse experiences and substance use at age 18 lowered R² by 0.05–0.07 in every model. The ranking of the models stayed the same.
- **Attention gates, checked against SHAP.** We had wanted to validate the gate values with SHAP and permutation importance on the same trained networks, but there wasn't time before. I've now done this. The two do not match: the gate values were almost identical across predictor groups (about 0.50), changed between splits, and agreed poorly with SHAP and permutation importance (mean Spearman ρ 0.04 for PLE-HM and 0.36 for depressive symptoms). So the paper reports the gates as not indicating predictor importance.
- **What drives the predictions.** SHAP and permutation importance agree across all models: symptoms at age 15 matter most, then the age-18 groups. Substance use at age 15 contributes little.
- **Calibration.** All models were reasonably well calibrated. DCMFNet over-predicted depressive symptoms slightly on average.

**What I need from you.** The draft keeps a marked placeholder for each of these, so you can fill them in directly in the draft if that's easier.

1. **Outcome instruments.** Which questionnaires the age-18 outcomes come from:
   - the 24 psychotic-like and hypomanic items (score divided by 46);
   - the 11 depressive items (divided by 33).

   Also what the constants 46 and 33 are. Reviewers will check this closely.
2. **Age-9 and age-15 measures.** The instruments and informants for the ADHD and autism items at age 9 (A-TAC, parent-reported?), and whether sex comes from the register or from self-report.
3. **Participant flow before the analysis file.** How we get from the full CATSS cohort to the 9,261 individuals with both outcomes and substance use at 15. My thesis gives 13,146 at the start and 9,160 genotyped, but only 6,643 of the 9,261 have PRS. I need the correct numbers for the flow figure.
4. **PRS pipeline.** The GWAS summary statistics, LD reference, PRS-CS settings, genotyping platforms, imputation and QC. Should this cite Wang (2025), or another published description?
5. **Ethics.** The ethical approval(s) for CATSS and the genotype linkage, with committee name, reference number and date, and how consent/assent was obtained.
6. **Authorship and declarations.** Author order and affiliations, the corresponding author, who did what (CRediT), funding and grant numbers, and competing interests. Also whether an institutional agreement covers the article processing charge.
7. **Decisions:**
   - whether the tuned comparators (linear regression, elastic net, gradient boosting) stay in the main tables, as in the draft, or move to the supplement;
   - the title and keywords, once you see the draft.
8. **Disclosure of model selection.** DCMFNet's depth configuration was partly chosen using test-set results, and its tuning data overlapped later test sets. The paper discloses this. Since the bias favours DCMFNet, the main conclusion is conservative. Please check that you are comfortable with how this is worded.
9. **Revisions after 28 October.** I won't be able to run anything once my access ends. If reviewers ask for new analyses, could one of you run them on the server? I'm leaving a note there with the folder layout, the run order and the exact scripts used, all of which are in the GitHub repository.

Thank you, and I'll send the full draft by [date].

Best,
[your name]
