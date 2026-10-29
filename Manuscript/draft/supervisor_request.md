Subject: DCMFNet paper: first draft for your input

Hi Shuyang and Sam,

I'm turning my thesis on DCMFNet into a paper for BMC Psychiatry, with Diagnostic and Prognostic Research as the fallback if it is transferred. Attached is a first draft. It is a working draft that I'll keep editing, but I'd like your input now on **which analyses and results should be included, moved to the supplement or left out** (see "What should be in the paper" below), before I polish the text.

**My access to the CATSS data ends on 31 October.** All analyses in the draft have been run. If you would like any further analysis that needs the data, please tell me by [date before 31 October], since I can't run anything after that. The code is in the GitHub repository and archived on Zenodo (https://doi.org/10.5281/zenodo.23755819). Placeholders that need your input are marked in red in the PDF.

**What changed since the thesis: shared train/test splits.** In the thesis, the R models (GLMM, GAMM) and the Python models (DCMFNet and the machine-learning models) each drew their own train/test splits. The same seed numbers give different partitions in R and Python, so the test sets differed between the two groups of models, and the thesis comparison was not made on the same test data. To fix this, I saved the twin-pair IDs of each training and test set (seeds 42–46) from the Python splits and re-ran all models on exactly those splits. In the same re-run, the R scripts now compute R² as 1 − SSE/SST, as Python does (the thesis R scripts used the squared correlation), and the mixed-model interaction terms now mirror DCMFNet's fusion blocks. This reversed the thesis comparison: the GLMM's test R² went from 0.181 to 0.228 for PLE-HM and from 0.179 to 0.202 for depressive symptoms, which puts it level with or above DCMFNet. Even on the old squared-correlation definition, the corrected GLMM gives 0.236 and 0.216, so the gain comes from the corrected setup, not only the change of definition. The paired comparisons below are all on identical splits.

**Attached:** `manuscript.pdf` (figures on the last pages), `additional_file_1.pdf` (supplementary tables) and `additional_file_2.pdf` (TRIPOD+AI checklist).

It would help to have your comments on the rest of the draft by [date].

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

**What should be in the paper.** These are the analyses in the draft now. Please tell me which to keep in the main text, move to the supplement or drop, and whether anything is missing. Any new analysis would need to be requested before 31 October, or run later by someone with server access.

| Analysis | Where it is now |
|---|---|
| Main comparison: DCMFNet vs GLMM vs GAMM, paired tests with 95% confidence intervals | Main text (Tables 3–4, Fig. 3) |
| Tuned comparators: linear regression, elastic net, gradient boosting | Main text (Tables 3–4) |
| MAE and Spearman's ρ for all models (Table 3 now shows R² and RMSE only) | Supplement (Table S6) |
| Default-settings screening of 40 algorithms | Supplement (Table S2) |
| Age-15 analysis (without the age-18 predictors) | Main text |
| Calibration | Main text (Fig. 4, Table 5) |
| Results by sex | Main text (Table 5) |
| Top 10% of scores (under-prediction, AUROC) | Main text (Table 5) |
| Gate values vs SHAP and permutation importance | Main text |
| Predictor-group importance across models | Main text (Fig. 5) |
| Mixed-model coefficients and interactions | Supplement (Table S3) |
| DCMFNet tuning and depth search | Supplement (Tables S1, S5) |

**What else I need from you.** The draft keeps a marked placeholder for each of these, so you can fill them in directly in the draft if that's easier.

1. **Data collection dates.** The paper should give the birth years of the twins and the years in which the age-15 and age-18 questionnaires were collected. Our analysis data has no birth-year column. Is it enough to give these for the CATSS cohort as a whole (for example, from the cohort documentation), or do we need them for the twins in our final sample? If the latter, I can look them up before 31 October if someone can tell me where to find them; otherwise someone with access would need to do it later.
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
   - affiliations: the draft gives Sam the Department of Clinical Neuroscience and the rest of us the Department of Medical Epidemiology and Biostatistics, both Karolinska Institutet, Stockholm, Sweden. Is that right for everyone, or does anyone need a different or second affiliation?
   - should I check with Tong and Lu directly, or will you?
5. **Declarations.** Who did what (CRediT), funding and grant numbers, and competing interests. Also whether an institutional agreement covers the article processing charge.
6. **Title and keywords.** Any changes you'd suggest.
7. **Disclosure of model selection.** DCMFNet's depth configuration was partly chosen using test-set results, and its tuning data overlapped later test sets. The paper discloses this. Since the bias favours DCMFNet, the main conclusion is conservative. Please check that you are comfortable with how this is worded.
8. **Revisions after 31 October.** I won't be able to run anything once my access ends. If reviewers ask for new analyses, could one of you run them on the server? The repository README lists the run order and the exact scripts used, and `Manuscript/draft/server_handover_README.md` has the full details and the values a rerun should reproduce.

Thank you!

Best,
Silpa
