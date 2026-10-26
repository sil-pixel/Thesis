# BMC Psychiatry Submission Requirements Checklist

**Target journal:** *BMC Psychiatry* (BMC / Springer Nature, open access). Fallback by transfer: *Diagnostic and Prognostic Research* (same publisher and article structure, but a populated TRIPOD+AI checklist is mandatory).
**Template:** Springer Nature LaTeX template v3.1 (Dec 2024), `Manuscript/sn-article-template/` (`sn-jnl.cls`). The template asks for a single .tex file, with no `\input`.
**Verification date:** 2026-10-23
**Sources and limits of verification:**
- The official BMC Psychiatry and Springer pages block automated access (redirects, HTTP 403, JavaScript challenge), so the journal's own "Preparing your manuscript" page was **not read**.
- Requirements below come from search-result snippets of BMC series pages and from the template itself.
- Every item marked `[VERIFY]` must be checked manually on bmcpsychiatry.biomedcentral.com before submission (TODO C-01).

Status: ✅ met · ⚠️ partial or placeholder · ❌ not met · ➖ not applicable

| # | Requirement | Source | Status | Action |
|---|---|---|---|---|
| 1 | **Editorial criterion:** scientific validity, not interest or impact: sound question, suitable methods and analysis, adherence to community standards | BMC series editorial-board page (search snippet) | ⚠️ | Methods now describe the shared splits, the matched comparators and paired tests. Configuration selection (D-04, D-09, D-21) is disclosed, not fixed (E-03). |
| 2 | Article type: Research article | BMC series | ✅ | — |
| 3 | **Title page:** title, all authors with affiliations, corresponding author email | Template; BMC series | ❌ | `[AUTHOR INPUT]` (C-03) |
| 4 | **Abstract:** structured as Background, Methods, Results, Conclusions; **≤ 350 words** `[VERIFY]` | BMC series pages (limit seen on BMC Women's Health and other BMC pages, not the BMC Psychiatry page) | ⚠️ | Draft at about 315 words with two TODO sentences. Recount after the TODOs are filled. |
| 5 | **Keywords:** 3–10 | BMC series pages | ✅ | 9 candidates; author to confirm |
| 6 | **Main text headings:** Background, Methods, Results, Discussion, Conclusions | BMC research-article format; template | ✅ | Restructured 2026-10-23. Background has no subheadings, as the template asks. |
| 7 | **List of abbreviations** | BMC format | ⚠️ | Drafted in `06_backmatter.md` |
| 8 | **Declarations**, all headings present ("Not applicable" where relevant): Ethics approval and consent to participate; Consent for publication; Availability of data and materials; Competing interests; Funding; Authors' contributions; Acknowledgements; Clinical trial number `[VERIFY]` | BMC series pages; template | ⚠️ | Headings in place; content is `[AUTHOR INPUT]` (C-02, C-03) |
| 9 | **Ethics statement** naming the approving committee and reference number; consent (minors); accordance with guidelines and regulations | Template (Methods guidance); BMC | ❌ | `[AUTHOR INPUT]` (C-02). Short version in Methods, full version in Declarations. |
| 10 | **Data availability statement** (mandatory) | BMC series pages | ⚠️ | Restricted-access statement drafted; author to add the access route |
| 11 | **Reporting guideline** checklist (EQUATOR); for prediction models TRIPOD+AI | BMC editorial policy `[VERIFY: whether BMC Psychiatry requires the checklist at submission]`; mandatory at DPR | ❌ | Complete TRIPOD+AI as Additional file 2 (C-06) |
| 12 | **Use of AI tools:** document LLM use in the Methods; AI tools cannot be authors | Springer Nature AI policy `[VERIFY]` | ⚠️ | Draft statement in Methods (*Software and reporting*) for the author to confirm |
| 13 | Sex and gender reporting (SAGER) | Springer Nature policy `[VERIFY]` | ❌ | Define the SEX variable (register vs self-report) |
| 14 | **References:** numbered (Vancouver) | BMC style | ⚠️ | LaTeX build uses `sn-vancouver-num`. Registry keys map to `latex/references.bib`. Some claims still `[VERIFY: content]`. |
| 15 | **Figures:** separate files at submission; legends in the manuscript | Template; BMC | ❌ | Regenerate as vector PDF (C-04); legends drafted in `06_backmatter.md` |
| 16 | **Tables:** editable, in the manuscript, cited in order | BMC | ⚠️ | Tables 1–4 placed; Table 1 drafted 2026-10-24 (group rows need scale notes) |
| 17 | **Additional files:** title and description for each | BMC | ⚠️ | Listed in `06_backmatter.md` |
| 18 | LaTeX submission: compiled PDF first; .tex source later `[VERIFY]` | Sibling BMC journal guidelines (search snippet) | ⚠️ | `latex/build.sh` produces `manuscript.tex` and the PDF |
| 19 | Journal-specific formatting needed only after the first round of review `[VERIFY for BMC Psychiatry]` | Sibling BMC journal (search snippet) | ➖ | — |
| 20 | Prior dissemination: master's thesis | Springer Nature preprint/thesis policy `[VERIFY]` | ⚠️ | Mention the thesis in the cover letter; avoid text recycling |
| 21 | Cover letter | BMC | ❌ | Draft after the results are final |
| 22 | Article processing charge | BMC | ⚠️ | `[VERIFY: current APC and any institutional agreement (e.g., Karolinska Institutet / Bibsam)]` |

## Changes made for BMC Psychiatry (2026-10-23)
- **Section files:**
  - `02_introduction.md` was replaced by `02_background.md`;
  - `03_materials_and_methods.md` was replaced by `03_methods.md`;
  - `07_declarations.md` was replaced by `06_backmatter.md`.
  - The AIIM versions are kept in `archive/aiim/`.
- **Headings:** section numbers were removed.
- **Tables:** renumbered. Tables 1–2 are in Methods and Tables 3–4 in Results.
- **Results:** the depth-configuration result was moved to the Additional file.
- **Framing:** a neutral benchmark plus an attribution check, written for a psychiatric readership. The AIIM-specific contributions are no longer promised: the simulation (E-16), learning curves (E-14) and stream ablation.
