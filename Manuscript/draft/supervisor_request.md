Subject: DCMFNet paper: draft coming, and a few inputs I need from you

Hi Shuyang and Sam,

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

1. **Data collection dates.** The paper should give the birth years of the twins and the years in which the age-15 and age-18 questionnaires were collected. Our analysis data has no birth-year column. Is it enough to give these for the CATSS cohort as a whole (for example, from the cohort documentation), or do we need them for the twins in our final sample? If the latter, could someone with access look them up after 28 October?
2. **Genotype data and PRS.** I've written the paragraph "Genotype data and polygenic scores" in Methods with everything I could take from my thesis and code: the linkage and QC counts (9,271 linked, 9,160 after QC), three genotyping batches, PRS-CS, 15 cell-type scores based on the top 10% most specifically expressed genes per cell type in the Siletti atlas, and the batch × PC adjustment. Could you confirm it is correct, and tell me whether it is enough or whether we should add more detail on the points below? If a published paper already describes the same CATSS genotype data and pipeline, a short summary and that citation may be enough. The pipeline I adapted (Han Wang's master's thesis, 2025) is not published, so I haven't cited it.
   1. Genotyping arrays for the three batches (the labels in the data are GSA3 and PSYC, so presumably the Global Screening Array and the PsychArray).
   2. Sample and SNP quality-control criteria.
   3. Imputation reference panel and software.
   4. The schizophrenia GWAS used (PGC3?), and whether it excluded Swedish samples to avoid overlap with CATSS.
   5. PRS-CS settings: the LD reference panel and whether the global shrinkage parameter was learned or fixed.
   6. How SNPs were mapped to the cell-type gene sets (for example a window around each gene), and whether the genome-wide score used all SNPs.
   7. Whether individuals of non-European ancestry were excluded or only flagged.
3. **Ethics.** The ethical approval(s) for CATSS and the genotype linkage, with committee name, reference number and date, and how consent/assent was obtained.
4. **Authorship.** The draft currently lists the authors in this order:
   1. Silpa Soni Nallacheruvu
   2. Shuyang Yao
   3. Sam Andersson
   4. Tong Gong
   5. Lu Yi

   Is this list and order fine, or should we change it? In particular:
   - should anyone else be included, for example someone responsible for the CATSS data or the genotyping and PRS;
   - who should be the corresponding author;
   - affiliations: the draft gives all of us the Department of Medical Epidemiology and Biostatistics, Karolinska Institutet, Stockholm, Sweden. Is that right for everyone, or does anyone need a different or second affiliation?
   - should I check with Tong and Lu directly, or will you?
5. **Declarations.** Who did what (CRediT), funding and grant numbers, and competing interests. Also whether an institutional agreement covers the article processing charge.
6. **Decisions:**
   - whether the tuned comparators (linear regression, elastic net, gradient boosting) stay in the main tables, as in the draft, or move to the supplement;
   - the title and keywords, once you see the draft.
7. **Disclosure of model selection.** DCMFNet's depth configuration was partly chosen using test-set results, and its tuning data overlapped later test sets. The paper discloses this. Since the bias favours DCMFNet, the main conclusion is conservative. Please check that you are comfortable with how this is worded.
8. **Revisions after 28 October.** I won't be able to run anything once my access ends. If reviewers ask for new analyses, could one of you run them on the server? I'm leaving a note there with the folder layout, the run order and the exact scripts used, all of which are in the GitHub repository.

Thank you, and I'll send the full draft by [date].

Best,
Silpa
