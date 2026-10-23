# Simulated AIIM Peer Review (round 0: thesis evidence as it stands)

Date: 2026-10-14. Reviewer stance: AIIM methodological reviewer plus psychiatric-epidemiology reviewer. This evaluates the *evidence currently available*. It does not evaluate prose, because the main text is not yet drafted.

**Recommendation if submitted today: Reject (desk or after review).** Main reasons: novelty is not established, the main performance claim is contradicted by the authors' own baselines, and the protocol has leakage and specification errors.

| # | Concern | Evidence | Consequence | Priority | Recommended action | Author approval |
|---|---|---|---|---|---|---|
| 1 | **Scope/novelty:** DCMFNet is a published architecture applied to new data. AIIM excludes "mere application" (Guide p.3). | contribution_matrix C2 | Desk rejection risk | Critical | Decide framing (X-00). If Option A, run E-06/E-10 and show a mechanism-level contribution. | Required |
| 2 | **Central claim contradicted:** OLS and penalized linear models outperform DCMFNet on identical splits. | D-11; 04_results Table R1 | Any superiority claim would be refuted by the paper's own benchmark | Critical | Report honestly; reframe (Option B) or obtain new evidence | Required |
| 3 | **Mis-specified interaction baselines (97/546 terms).** | D-01 | Interaction conclusions and LMM/GAMM metrics are invalid | Critical | E-01 | Required |
| 4 | **Non-identical partitions across model families.** | D-02 | Comparison not paired | Critical | E-02 | Required |
| 5 | **Test-informed selection and overlapping tuning data.** | D-04, D-09, D-21 | Optimistic bias favouring the proposed model | Critical | E-03 nested protocol | Required |
| 6 | **Outcome constructs:** "negative symptoms" are depressive items; "positive" includes hypomanic items. | D-03 | Psychiatric validity; title and abstract misleading | Critical | M-01 rename + justify | Required (input) |
| 7 | **No inferential comparison;** 5 splits with SD only. | A-01 | Cannot assess "consistent improvement" | Critical | E-08 | Required |
| 8 | **Concurrent predictors (age 18)** mislabelled as risk prediction. | D-14 | Reverse causation; not prospective | High | E-05 | Required |
| 9 | **Interpretability claims rest on gate means** of unstandardised inputs. | D-12 | Not interpretable as importance | High | E-06, E-07 | Required |
| 10 | **Complete-case analysis drops ~56%;** descriptives describe a different sample. | D-10 | Selection bias; TRIPOD+AI non-compliance | High | A-02, E-12 | Required |
| 11 | **Model capacity vs n:** 105k parameters, ~2,400 training rows. | Methods Table 2 | Overfitting risk; motivates learning curves | Medium | E-14; report parameter count (done) | Required |
| 12 | **Methods misdescribe implementation** (SE squeeze; loss). | D-06, D-07 | Reproducibility | High | Fixed in draft (M-03, M-04) | — |
| 13 | **Calibration not assessed.** | E-09 | TRIPOD+AI; clinical relevance | High | E-09 | Required |
| 14 | **Twin dependence** is ignored in DCMFNet and ML model fitting (only splits are grouped). Internal CV of LassoCV etc. is not grouped. | D-13 | Mild optimism in baselines' internal tuning; standard errors | Medium | Document; grouped inner CV in E-03 | Required |
| 15 | **No external validation;** Swedish twins, European-ancestry PRS. | — | Generalizability | Medium | Limitation; E-15 | Input |
| 16 | **Clinical usefulness** not assessed (AIIM strongly recommends). | Guide p.2 | Weak medical impact case | Medium | Discuss honestly. For population cohort risk models, a decision-analytic evaluation would require a defined use case, so do not invent one. | Input |
| 17 | **Reproducibility:** dataset-construction script missing; tuning artifacts overwritten. | D-16, D-21 | Cannot reproduce | Medium | R-01, R-02, C-05 | Required |
| 18 | **Sex/gender (SAGER)** not addressed. | Guide p.8–9 | Compliance | Medium | A-04; define SEX | Input |
| 19 | **Literature positioning** lacks direct competitors (PRS + environment ML for psychosis/depression). | L-02 | Novelty unclear | High | Targeted search | — |

## What reviewers would view positively
- Twin-pair-grouped splitting (avoids co-twin leakage) in all pipelines.
- Repeated random splits rather than a single split.
- A broad baseline sweep on identical partitions (the ML benchmark), which is rarer than it should be.
- Use of cell-type-specific PRS, a recent and biologically motivated input.
- The thesis already acknowledges the predictive ceiling and the twin-dependence limitation.

All actionable items are transferred to `TODO.md`.
