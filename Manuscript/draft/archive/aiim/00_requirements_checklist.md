# AIIM Submission Requirements Checklist

**Target journal:** *Artificial Intelligence in Medicine* (Elsevier, ISSN 0933-3657)
**Primary source:** `Manuscript/Guide for authors - Artificial Intelligence in Medicine - ISSN 0933-3657 _ ScienceDirect.com by Elsevier.pdf` (22 pp.; PDF footer shows it was printed from the live ScienceDirect page on 14/10/2026 18:44)
**Verification date:** 2026-10-14
**Online/PDF discrepancy check:** The PDF is a same-day capture of the live page, so no discrepancy is expected. The live page was not re-fetched because ScienceDirect returns HTTP 403 to automated access. Re-check it manually before submission (TODO C-01).

Status legend: ✅ met · ⚠️ partially met / draft placeholder · ❌ not met · ➖ not applicable

| # | Requirement | Required / Recommended | Source (PDF page) | Current status | Action needed |
|---|---|---|---|---|---|
| 1 | **Article type:** "Original research contributions: Theoretical and/or methodological papers about novel approaches" | Required (choice) | p.3 | ⚠️ | This is the only type that fits. AIIM states that "mere application of well-known or already published algorithms… to medical data is not regarded as original research" (p.3). The novelty case is currently weak; see `research/contribution_matrix.md`. |
| 2 | **Scope:** both (a) potential high impact in a medical domain AND (b) strong methodological novelty in AI/CS | Required | p.2 | ❌ | (b) is not established. DCMFNet is an existing architecture (Huang et al., 2024) adapted to tabular data, and it underperforms simple linear baselines. See the framing decision. |
| 3 | Compare with other proposals and explicitly discuss elements of novelty | Required for methodological papers | p.2 | ⚠️ | ML benchmark exists. Comparisons with tabular-DL methods (TabNet, FT-Transformer, MLP) and tuned GBDT are missing. |
| 4 | Real-world medical domain discussed "at the proper depth, from both the technical and the medical points of view" | Required | p.2 | ⚠️ | Psychiatric content must be rebuilt around correct outcome constructs (see D-03). |
| 5 | Clinical assessment of usefulness/potential impact | Strongly recommended | p.2 | ❌ | No clinical-utility evaluation (e.g., decision curve, high-risk identification). |
| 6 | Clear English, revised with care (fluent co-authors or editing service) | Required in practice ("poor English… likely rejected") | p.3, p.21 | ⚠️ | Final language edit needed. Use American or British English consistently, not both (p.21). |
| 7 | **Ethics statement** confirming approval by committee(s), with **date and reference number**, plus informed consent; or a justification for exemption | Required at submission | p.4, p.10 | ❌ | `[AUTHOR INPUT: Swedish Ethical Review Authority approval number(s) and date(s) for CATSS + genotype linkage]` |
| 8 | Declaration of Helsinki / ICMJE compliance; consideration of vulnerable populations (minors, mental-health conditions) | Required (studies in humans) | p.10 | ❌ | CATSS participants were children at ages 9/15. Consent/assent procedures must be described. `[AUTHOR INPUT]` |
| 9 | Submission declaration (not published elsewhere; the thesis is allowed as prior "academic thesis") | Required (implied) | p.5 | ⚠️ | Disclose the master's thesis (Karolinska Institutet, June 2026) in the cover letter. Avoid text recycling from the thesis. |
| 10 | Authorship criteria (ICMJE-type: all three conditions) and definitive author list at submission | Required | p.5–6 | ❌ | `[AUTHOR INPUT: author list/order; supervisors' roles]`. No changes are allowed after acceptance. |
| 11 | Single corresponding author; email on the title page | Required | p.5, p.12 | ❌ | `[AUTHOR INPUT]` |
| 12 | **Title page:** concise title without abbreviations where possible; full names; affiliations with lowercase superscripts and full postal address with country; corresponding author | Required | p.12 | ⚠️ | Draft in `01_title_abstract.md`. Avoid "DCMFNet" in the title (non-established abbreviation). |
| 13 | Competing-interest declaration via the **declarations tool** (.docx uploaded) | Required | p.6 | ❌ | `[AUTHOR INPUT]` |
| 14 | Funding statement in the standard format, or the no-funding sentence; role of sponsors | Required | p.6–7 | ❌ | `[AUTHOR INPUT: funding, e.g., KI/MEB grants for CATSS, genotyping]` |
| 15 | **Generative AI declaration**: new section before references titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process", using the given template | Required if AI tools were used | p.7 | ⚠️ | The thesis already declares AI use, and this draft is AI-assisted, so the statement is needed. Template in `07_declarations.md`. |
| 16 | AI-generated figures: disclose in each caption and in the GenAI statement; no AI alteration of primary data images | Required if applicable | p.9, p.15 | ➖/⚠️ | Applies only if any schematic (e.g., architecture figure) was AI-generated. `[AUTHOR INPUT]` |
| 17 | Inclusive language; gender-neutral writing | Required | p.8 | ⚠️ | Avoid stigmatizing terms ("schizophrenic", "addicts"). Use person-first language. |
| 18 | **Sex/gender reporting (SAGER):** define sex vs gender; address sex/gender dimensions or declare as a limitation | Required | p.8–9 | ❌ | The dataset variable `SEX` (coded 1/2) needs a definition (register sex vs. self-report?). `[VERIFY]` Consider SAGER checklist. |
| 19 | **File format:** editable source (.docx single-column, or .tex); PDF not acceptable as source | Required | p.12 | ❌ | The draft is Markdown. Convert to .docx or Elsevier LaTeX (elsarticle) at the end. |
| 20 | No strikethrough/underline unless meaningful | Required | p.12 | ✅ | — |
| 21 | **Abstract ≤ 250 words**; purpose, procedures, main findings with statistical significance where applicable, principal conclusions; stand-alone; no references; define abbreviations | Required | p.12–13 | ⚠️ | Skeleton only, because it is framing-dependent. Word count must be checked when written. |
| 22 | **Keywords: 1–7**, English, avoid multi-word "and/of" phrases; abbreviations only if established | Required | p.13 | ⚠️ | Candidate list in `01_title_abstract.md` |
| 23 | **Highlights: 3–5 bullets, ≤ 85 characters each including spaces**, separate editable file with "highlights" in the filename | Encouraged (optional) | p.13 | ⚠️ | Draft after framing |
| 24 | Graphical abstract | **Not mentioned** in this guide | — | ➖ | No requirement. `[VERIFY: JOURNAL REQUIREMENT]` on the live submission system |
| 25 | SI units | Required | p.13 | ✅ | No physical units are used, apart from time if runtime is reported. |
| 26 | **Math:** editable text, not images; variables italic; `exp` for powers of e; solidus for small fractions; display equations numbered consecutively | Required | p.13 | ⚠️ | Methods draft uses LaTeX-style math; renumber when converting |
| 27 | **Tables:** editable text; cited in text; numbered consecutively; captions provided; notes below table; **no vertical rules or shading**; avoid duplicating results | Required | p.13–14 | ⚠️ | — |
| 28 | **Figures:** separate files (Figure_1…); cited and numbered; caption = brief title + description, all symbols/abbreviations explained; vector EPS/PDF or ≥300 dpi photos/≥1000 dpi line art; no multiple graphs in one file (except intended comparisons); colour-blind accessible | Required | p.14–15 | ❌ | Existing thesis figures are 150-dpi PNG (`dpi=150` in `legacy/train.py`). They must be regenerated as vector PDF. |
| 29 | Image manipulation policy | Required | p.9 | ✅ | No primary image data |
| 30 | Supplementary material: cited in text, captioned, submitted with the manuscript | Recommended | p.15 | ⚠️ | Plan: Appendix/Supplement for full hyperparameter tables, variable list, sensitivity analyses |
| 31 | **Research data:** Option B, meaning deposit and link data **or** state why data cannot be shared | Encouraged (statement) | p.16 | ⚠️ | CATSS/STR individual data cannot be shared publicly. State the access route via the Swedish Twin Registry. `[AUTHOR INPUT: exact access procedure]` |
| 32 | Data statement at submission | Encouraged | p.16 | ⚠️ | Draft in `07_declarations.md` |
| 33 | **Code availability** | **No explicit requirement** in guide (code counts as "research data", p.16) | p.16 | ⚠️ | Thesis states code at github.com/sil-pixel/Thesis. Recommend a tagged release with a Zenodo DOI and removal of the synthetic/deployment components from the paper's provenance. |
| 34 | **Article structure:** clearly defined numbered sections (1, 1.1, 1.1.1); abstract not numbered; cross-reference by section number | Required | p.17 | ⚠️ | Applied in `manuscript.md` |
| 35 | Acknowledgements: separate section directly before references; not on the title page | Required | p.17 | ⚠️ | — |
| 36 | **CRediT author statement** | Required (corresponding author) | p.17–18 | ❌ | `[AUTHOR INPUT]` |
| 37 | Appendices labelled A, B…; equations (A.1); tables/figures Table A.1, Fig. A.1 | Required if appendices are used | p.18 | ⚠️ | — |
| 38 | **References:** every cited item in the list and vice versa; must be real; include DOIs; no "unpublished"/personal communication in the list (recommended); "in press" = accepted | Required | p.18 | ⚠️ | Registry in `literature/reference_registry.md` |
| 39 | **Reference style:** numbered in square brackets in order of appearance; >6 authors → first 6 + "et al."; LTWA journal abbreviations; consistent style accepted at submission | Style is flexible at submission, but the numbered system is described | p.18–19 | ⚠️ | Use a Vancouver-like numbered style |
| 40 | Data references tagged `[dataset]`; software references tagged `[software]` with version | Encouraged | p.19 | ⚠️ | Cite PyTorch, lme4, mgcv, LazyPredict, Optuna with versions `[VERIFY versions]` |
| 41 | Preprints marked as such with DOI; use the formal publication if available | Required | p.20 | ✅ | DCMFNet is a GI'24 proceedings paper, not a preprint |
| 42 | Clinical trial registration | Required for trials only | p.11 | ➖ | Observational cohort; registration not required |
| 43 | **Reporting guideline (TRIPOD+AI)** | **Not mandated by AIIM**; the thesis claims adherence | — | ⚠️ | Strongly recommended for a prediction-model paper. Complete the TRIPOD+AI checklist as a supplement (TODO C-06). |
| 44 | Submission checklist (corresponding author details, all files, spell check, references cross-checked, permissions, APC awareness) | Required | p.20 | ❌ | Final stage |
| 45 | Word / page limit | **None stated** in this guide | — | ➖ | `[VERIFY: JOURNAL REQUIREMENT]`. No explicit limit was found. Typical AIIM research articles are long-form, but keep the paper focused. |
| 46 | Peer review: single anonymized (no blinding of the manuscript needed) | Info | p.4 | ✅ | — |
| 47 | Conference papers: only if thoroughly revised and extended | Info | p.4 | ➖ | Not a conference paper. A master's thesis is allowed prior dissemination. |

## Items that could not be confirmed from the guide
- Graphical abstract requirement: `[VERIFY: JOURNAL REQUIREMENT]` (not mentioned)
- Word/figure/table limits: `[VERIFY: JOURNAL REQUIREMENT]` (not mentioned)
- Mandatory reporting guideline: none stated. TRIPOD+AI is recommended on scientific grounds, not as a journal mandate.
