# Submission Readiness

**Assessment date:** 2026-10-19 (updated)
**Status: NOT READY.** There are 7 open Critical TODOs plus 1 in progress (see `TODO.md`). Submission readiness cannot be declared while any Critical item is open.

| Area | Status | Blocking items |
|---|---|---|
| Scientific framing / novelty | ⚠️ Option B chosen (2026-10-19); the methodological core depends on E-06/E-14/E-16 | E-16 |
| Validity of comparisons | ❌ | E-01, E-02, E-03, A-01 |
| Outcome definitions | ❌ | M-01 |
| Methods text | ⚠️ Drafted from code (as conducted); needs update after re-analysis | E-01–E-07 |
| Background | ⚠️ Rewritten for BMC Psychiatry (02_background.md) | L-02, VERIFY content tags |
| Results / Discussion | ❌ Discussion outlined (05_discussion.md); Results await re-analysis | E-01–E-16 |
| Abstract, highlights, keywords | ❌ Skeleton | after results |
| Ethics, authorship, CRediT, funding, COI | ❌ Author input | C-02, C-03 |
| GenAI declaration | ⚠️ Template drafted | author confirmation |
| Figures (format/resolution) | ❌ | C-04 |
| References | ⚠️ 25+ verified in registry; content verification partial | L-01, L-02 |
| Reporting checklist (TRIPOD+AI) | ❌ | C-06 |
| Code/data availability | ⚠️ | C-05, R-01, R-02 |
| File format (.tex) | ⚠️ Springer Nature template; built by latex/build.py; compiles | author block, figures |

**Manuscript word count:** not meaningful yet. Only Methods is drafted (~2,000 words incl. comments). The Guide states no word limit.

## Target journal (decision 2026-10-23): **BMC Psychiatry**, Springer Nature LaTeX template; fallback by transfer: Diagnostic and Prognostic Research. Requirements: `00_requirements_checklist.md`.

## Journal comparison (2026-10-28)
How the four candidates compare for this paper as it now stands: a benchmark in which the new network did not beat regression, with mixed models, polygenic scores and a twin cohort. Fees and metrics come from third-party aggregators unless noted; **check them on the publisher's pages and in Clarivate JCR before relying on them.** Fees change every year, and Karolinska Institutet's agreements (Bibsam) may cover some or all of a fee `[VERIFY: KI library]`.

| | Artificial Intelligence in Medicine (AIIM) | BMC Psychiatry | Diagnostic and Prognostic Research (DPR) | Translational Psychiatry |
|---|---|---|---|---|
| Publisher, access | Elsevier; hybrid (subscription route without a fee, or open access) | BMC / Springer Nature; fully open access | BMC / Springer Nature; fully open access | Springer Nature (Nature Portfolio); fully open access |
| Readership | AI and informatics in medicine | General psychiatry, all methods | Prediction-model methodologists and clinical epidemiologists | Biological and translational psychiatry |
| Impact factor (aggregator) | about 7.8 (another source 9.45) | 3.4 (2023, via Wikipedia/JCR) | about 3.9 | listed range about 5–10; CiteScore 8.5 |
| Fee (open access) | none on the subscription route; OA fee not found | about €2,490 / US$2,890 / £2,090 | about €2,290 / US$2,790 / £1,990 | about €3,590 / US$4,790 / £3,190 |
| Fit with a "no gain over regression" result | **Weak.** The journal looks for methodological novelty, and the paper's new method did not win | **Good.** Broad scope; the paper is framed as a careful benchmark for psychiatric prediction | **Very good.** Scope covers the evaluation and comparison of prediction models, and it states it publishes sound research regardless of outcome | **Uncertain.** The genetic and multimodal angle fits, but the interactions were null and the paper is not about biology or translation |
| Reporting demands | No reporting guideline required | TRIPOD+AI recommended `[VERIFY]` | Populated TRIPOD+AI checklist **mandatory** (already done, Additional file 2) | Nature Portfolio reporting summary and data/code statements `[VERIFY]` |
| Pros | Highest impact among the four; AI audience understands the architecture; no fee on the subscription route | Reaches the psychiatric researchers who build these models; the current draft and template are set up for it; transfer to DPR is possible within BMC | Ideal referees for the corrected t-tests, grouped splits and TRIPOD+AI; low risk of rejection for a negative result; checklist already done | High visibility in psychiatric genetics; values the cell-type PRS and twin cohort |
| Cons | High risk of desk rejection for "insufficient methodological novelty"; earlier AIIM draft would need its framing reversed | Lower impact factor; generalist referees may ask for clinical relevance | Small readership in psychiatry; lowest visibility for the psychiatric findings | Highest fee; scope risk of a desk rejection; may expect biological follow-up or replication that can't be run after 28 October |
| Draft work needed | Reformat to Elsevier; restore the Option B framing (archive/aiim) | None beyond the open TODOs | Minimal (same template and structure) | Reformat to Nature Portfolio style; rewrite the Background around translational relevance |

**Recommendation.** Keep BMC Psychiatry first and DPR as the fallback. Together they fit a careful negative benchmark best, the draft is already in their format, and a transfer between them is straightforward. Translational Psychiatry is worth considering only if the supervisors think the polygenic-score angle is strong enough for its scope. AIIM is no longer a good fit, because the paper no longer claims a better method.

Sources (accessed 2026-10-28): Translational Psychiatry fee and metrics from [Researcher.Life](https://researcher.life/journal/translational-psychiatry/20054) and [Editage](https://www.editage.com/journal/translational-psychiatry/); BMC Psychiatry from [Wikipedia](https://en.wikipedia.org/wiki/BMC_Psychiatry) and [Researcher.Life](https://researcher.life/journal/bmc-psychiatry/120); AIIM from [journalmetrics.org](https://www.journalmetrics.org/journal/artificial-intelligence-in-medicine) and [LetPub](https://www.letpub.com/journal-selector/journal/893); DPR as recorded below (2026-10-19 and 2026-10-22).

## Earlier target journals (decision 2026-10-19, superseded)
**Primary: Artificial Intelligence in Medicine (Elsevier), Option B framing.** Main risk: a desk rejection as "insufficient methodological novelty" (Guide p.3). Mitigation: the diagnostic experiments (E-06, E-14, E-16) must form the methodological core.

Fallbacks, in order. Scope was checked by web search on 2026-10-19. **Re-verify fees and scope on the official pages before any transfer.**

| # | Journal | Fit for Option B | Fee (OA) | Verification |
|---|---|---|---|---|
| 1 | ~~Journal of Clinical Epidemiology (Elsevier)~~ **Dropped 2026-10-23:** its exclusions cover prognostic research without a methods focus and single-country studies. | Published the ML-vs-LR review [R-CHRIST]; prediction-methodology readership | Hybrid (subscription route available) `[VERIFY: on the official Elsevier page]` | Aggregator listings only |
| 2 | Diagnostic and Prognostic Research (BMC) | Scope covers evaluation of prediction models and methodology articles; states it publishes well-conducted research regardless of outcome | APC ≈ £1990 / $2790 / €2290 `[VERIFY: mirror of the BMC About page; official page not reached]` | Journal aims-and-scope page (search snippet) |
| 3 | BMC Medical Research Methodology | Studies of how methodological choices affect results; suits E-16 | APC `[VERIFY]` | Mirror/aggregator |
| 4 | PLOS Digital Health | Rigour-based criteria; no explicit negative-results policy found | Flat-fee model `[VERIFY amount; check institutional agreement]` | PLOS journal-information page (search snippet) |

Also possible: Journal of Biomedical Informatics (scope risk similar to AIIM), JAMIA Open, and ML4H / CHIL proceedings. Psychiatry journals only after M-01, and with a different framing.
Elsevier's Article Transfer Service can propose sister journals after a rejection; whether AIIM participates is `[VERIFY]`.

### Journal metrics (third-party aggregators, 2026-10-22; verify in Clarivate JCR before relying on them)
No official acceptance rates were found for any of these journals.
- **AIIM:** IF 7.8 (journalmetrics.org; another aggregator gives 9.45).
- **J Clin Epidemiol:** IF 5.2.
- **Diagn Progn Res:** IF 3.9; median 25 days to first decision and 174 days to acceptance (BMC page, 2023).
- **BMC Med Res Methodol:** IF 3.7.
- **PLOS Digit Health:** IF 7.7; APC about USD 2,575 (Researcher.Life).
