# Contribution Matrix and Framing Options

Assessed 2026-10-14 against the repository evidence in `../discrepancies.md` and the verified literature in `../literature/reference_registry.md`.

## 1. Component-level novelty classification

| Component | Origin | Classification |
|---|---|---|
| Iterative gated fusion (IGF): bilinear tanh fusion + sigmoid update gate, repeated L times | Huang et al., GI 2024 [R-DCMF] | **Established** (taken from the source paper) |
| Treating tabular variable groups (substance use, PRS, ADHD, ACE…) as "modalities" with substance use as the guided/anchor modality | This work | **Adaptation**. It is a design choice, not a new mechanism. Group-wise tabular encoders exist in the multimodal-EHR literature [R-BORSOS, R-ANDREEVA – content unverified]. |
| Per-instance sigmoid feature gating after fusion ("SE-inspired") | SE networks [R-SE]; gated linear units | **Established** (and mis-described in the thesis; D-06) |
| Parallel "independent" stream (all raw inputs, gated) concatenated with fused stream | This work | **Adaptation**. It is functionally a skip connection, which makes the model a superset of a gated linear model. |
| Imbalanced-regression loss: Huber + inverse-frequency bin weights + focal residual weighting | Combination of known ideas (Huber; focal loss; label-distribution reweighting) `[CITATION NEEDED: Yang et al. 2021 Delving into deep imbalanced regression – to verify]` | **Novel combination (minor)** |
| Per-modality IGF depth search used as an "interaction complexity" probe | This work | **Potentially novel idea**, but the evidence is weak and confounded (D-04, D-05) |
| Gate-value "importance" analysis | This work | **Established practice with known validity problems** [R-JAIN] (D-12) |

## 2. Candidate contributions

| # | Proposed contribution | Closest previous work | Technical difference | Experimental evidence | Significance | Limitations | Confidence | Status |
|---|---|---|---|---|---|---|---|---|
| C1 | DCMFNet improves prediction of age-18 symptom scores over statistical and ML baselines | Christodoulou 2019 [R-CHRIST]; Grinsztajn 2022 [R-GRIN]; Shwartz-Ziv 2022 [R-SHWARTZ]; Tate 2020 (CATSS ML) [R-TATE] | — | **Contradicted.** Exceeded by 20/40 (Pos) and 17/40 (Neg) baselines on identical splits; OLS R² 0.239 vs 0.206 (Pos) (D-11). Beats GLMM/GAMM only on non-identical splits with a mis-specified interaction set (D-01, D-02). | — | Leakage favouring DCMFNet (D-04, D-09) | High (that the claim fails) | **UNSUPPORTED** |
| C2 | Adaptation of an image–text fusion architecture (IGF) to grouped tabular risk factors with an exposure-anchored design | Huang 2024; multimodal tabular fusion (DAFT in Borsos 2024 [R-BORSOS]); TabNet [R-TABNET] | Anchoring fusion on one exposure group; per-group IGF blocks | Model trains stably (training curves). No ablation shows the fused stream adds predictive value over the independent stream alone. | Low–moderate as engineering | AIIM states that applying published algorithms is not original research (Guide p.3) | Medium | **PARTIALLY SUPPORTED** (feasibility only) |
| C3 | The exposure-anchored fusion stream captures substance-use × factor interactions that matter for prediction | Interaction detection: H-statistic, SHAP interaction values `[CITATION NEEDED: Friedman & Popescu 2008; Lundberg 2020 TreeSHAP interactions – to verify]` | — | None. The thesis evidence is gate means (D-12) and agreement with a mis-specified GLMM (D-01). OLS without interactions outperforms all interaction models. | Would be high if shown | Requires fused-stream ablation + interaction attribution | Low | **REQUIRES ADDITIONAL EXPERIMENTS** |
| C4 | Per-modality IGF depth reveals modality-specific interaction complexity | Neural architecture search; none directly comparable | Using depth sensitivity as an interpretive probe | Depth effects are within seed noise: the per-modality optimal Pos validation RMSE ranges only 0.0877–0.0882 across modalities, with across-seed SDs 0.0014–0.0024 <!-- src: Models/DCMFNet/Results/Complexity Analysis/Avg complexity results/Pos_optimal_depths.csv -->. 5-seed test comparison: per-modality RMSE 0.0898 ± 0.0015 vs uniform 0.0905 ± 0.0010 (Pos); Neg 0.1569 vs 0.1569. The chosen optimum differs between single-seed and 5-seed sweeps (D-05). | Conceptually interesting | Depth ≠ interaction complexity; differences are within noise | Low | **UNSUPPORTED** in current form |
| C5 | Imbalanced-regression loss improves tail (high-symptom) prediction | Imbalanced regression literature `[CITATION NEEDED]` | Huber+focal+bin reweighting | Thesis claims better upper-tail calibration from seed-42 scatter plots only. There is no quantitative tail metric and no ablation vs plain MSE/Huber under the final setup. (`Results/Old Model/*_loss_training.log` exist but predate the final architecture; `[VERIFY]`) | Clinically relevant if true (high-risk identification) | Needs a pre-specified tail metric | Low–medium | **REQUIRES ADDITIONAL EXPERIMENTS** |
| C6 | Rigorous, leakage-aware benchmark showing that, for adolescent questionnaire + cell-type PRS data, flexible models (deep fusion, GBDT) do not outperform regularized linear models for age-18 psychopathology scores | Christodoulou 2019; Grinsztajn 2022; Tate 2020 (CATSS, binary outcomes) | Continuous symptom scores; cell-type-specific PRS; twin-grouped evaluation; deep multimodal fusion included | Largely present in the saved results (identical-split ML benchmark incl. DCMFNet). Needs fixes for D-01/D-02/D-04/D-09, paired uncertainty, and calibration metrics. | Moderate for the field (a negative result, honestly reported) | AIIM wants methodological novelty, so a negative benchmark alone may be judged out of scope | Medium | **PARTIALLY SUPPORTED** |
| C7 | Substance use acts as a "context-dependent amplifier" (thesis substantive claim) | Gene–environment psychosis literature [R-ROBINSON, R-DIFORTI, R-VARESE] | — | Rests on D-01 (arbitrary interaction subset), uncorrected multiplicity (~200 coefficients), predictive models without causal design, and concurrent age-18 covariates (D-14) | Would be clinically meaningful | Observational; not identifiable as causal moderation | Low | **UNSUPPORTED** |

## 3. Research gap (evidence-based draft)

What is established:
- Adolescent cannabis/substance use, childhood adversity and polygenic liability are each associated with psychosis outcomes [R-DIFORTI, R-VARESE, R-PGC3, R-ROBINSON].
- ML applied to CATSS questionnaire data predicts adolescent mental-health problems with performance comparable to logistic regression [R-TATE; abstract-level verification pending].
- Across clinical prediction, ML has not shown systematic gains over regression in low-bias comparisons [R-CHRIST]. On medium-sized tabular data, deep models generally do not beat tree ensembles [R-GRIN, R-SHWARTZ].
- Cell-type-specific PRS are a recent resource [R-YAO2025] that has not been evaluated as model inputs alongside environmental exposures for continuous adolescent symptom outcomes `[CITATION NEEDED: confirm absence via targeted search]`.

What remains insufficient, and why it matters:
- It is unclear whether architectures designed to model *structured cross-group interactions* extract predictive signal beyond additive models in population cohorts of this size (n ≈ 4,000). This matters because interaction-heavy models are increasingly proposed for psychiatric risk, and their added complexity costs interpretability and stability.

How this study can address it (only after fixes):
- A twin-aware, leakage-controlled comparison of an exposure-anchored deep fusion model against additive, regularized, interaction-specified and tree-based models. The comparison would include an ablation of the fusion stream and validated attribution.

## 4. Framing options

> **Author decision 2026-10-19: Option B, target journal AIIM.** Options A and C are kept below only for the record.

### Option A — "Methodological contribution: exposure-anchored deep fusion for heterogeneous tabular risk factors"
- **Central question:** Does an exposure-anchored iterative gated fusion network improve prediction of adolescent psychopathology over standard models?
- **Main contribution:** C2 (+ C3, C5 if new experiments succeed)
- **Supporting evidence now:** Feasibility only. Current results contradict the performance claim (C1).
- **Likely reviewer objections:** "Mere application" (Guide p.3); a linear baseline beats it; leakage; misdescribed components; no ablations.
- **Additional evidence needed:** E-01–E-07 at minimum. Novelty would still be moderate.
- **Risk:** High probability of desk rejection unless new experiments change the picture. **Not recommended** with current evidence.

### Option B — "Rigorous negative/benchmark result: do interaction-focused deep models help for psychiatric risk scores from cohort questionnaires and cell-type PRS?" (Recommended)
- **Central question:** Under twin-aware, leakage-controlled evaluation, do interaction-focused models (pre-specified mixed models, deep cross-modal fusion, tree ensembles) outperform penalized additive models for age-18 psychotic-like/hypomanic and depressive symptom scores?
- **Main contribution:** C6, with DCMFNet as the strongest representative of explicit interaction modelling, plus a diagnostic analysis of *why* (fused-stream ablation, signal ceiling, effective sample size, Riley-type sample-size reasoning [R-RILEY]).
- **Supporting evidence now:** Strong directional evidence (D-11). It needs the fixes to be clean.
- **Likely reviewer objections:** AIIM's emphasis on methodological novelty. A negative result might be redirected to a sister journal (e.g., via Article Transfer Service).
- **Additional evidence needed:** E-01, E-02, E-03, E-04, E-05, E-06, A-01 (paired uncertainty), calibration.
- **Mitigation for scope:** Add a methodological element, e.g., a leakage-aware nested evaluation protocol for twin cohorts, or an interaction-attribution comparison across model classes. It must be real work, not wording.

### Option C — "Substantive epidemiology: substance use × genetic/environmental risk in CATSS"
- **Central question:** Do associations between adolescent substance use and age-18 symptoms vary by cell-type PRS and adversity?
- **Main contribution:** C7.
- **Supporting evidence now:** Unsupported (D-01, D-14, multiplicity).
- **Fit:** Poor fit for AIIM. It belongs in a psychiatric-epidemiology journal and would require a causal/moderation design (pre-specified hypotheses, multiplicity control, twin-difference models).
- **Not recommended for AIIM.**

**Recommendation:** Option B, conditional on the Critical TODOs. If the author prefers Option A, the experiments must be run *before* drafting claims, with pre-specified success criteria (see `experiment_recommendations.md`), and the author must be prepared for the result to remain negative.
