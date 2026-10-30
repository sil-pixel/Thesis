# DCMFNet: deep cross-modal fusion for adolescent psychopathology in CATSS

This repository holds three related pieces of work:

1. **[Manuscript](#1-manuscript-in-preparation)**: the analyses behind a paper, in preparation, that compares DCMFNet with mixed-effects and machine-learning models on the same twin-pair-grouped splits.
2. **[Master's thesis](#2-masters-thesis)**: the original thesis project (2026).
3. **[Bodhica synthetic-data workflow](#3-bodhica-synthetic-data-workflow)**: training and export of DCMFNet models on **synthetic data** for the [Bodhica](https://github.com/sil-pixel/Clinical-Risk-Agent) research demonstration.

**No CATSS data are included.** The analyses use data from the Child and Adolescent Twin Study in Sweden (CATSS), which contain sensitive personal information and are held on a secure server at Karolinska Institutet. The repository contains code and aggregated results only.

---

## 1. Manuscript (in preparation)

*Does deep cross-modal fusion improve prediction of late-adolescent psychopathology? A comparison with mixed-effects models in a Swedish twin cohort.*

The results below are from a draft that has not been peer reviewed.

**Summary.** In 4,033 twins (2,690 pairs) with complete data on 105 predictors, DCMFNet was compared with generalised linear and additive mixed models (GLMM, GAMM) that had the same inputs, 24 substance-use interactions and a twin-pair random intercept, and with linear regression, an elastic net and gradient boosting tuned within the training data. All models used the same five twin-pair-grouped splits (seeds 42–46).

- Mean test R² for psychotic-like and hypomanic experiences: DCMFNet 0.206, GLMM 0.228, GAMM 0.227. For depressive symptoms: 0.198, 0.202 and 0.200. No paired difference was statistically significant.
- The elastic net did best (0.260 and 0.237), but no difference survived correction for multiple comparisons.
- No substance-use interaction term survived correction, and in no model did the interactions improve test R² significantly (removing them slightly raised the mixed models' R²). DCMFNet's gate values did not agree with SHAP values or permutation importance.

**Where things are**

| Folder | Contents |
|---|---|
| `Manuscript/draft/` | Manuscript sections (`01_title_abstract.md` … `06_backmatter.md`), Additional files 1–2, figures, TODO list and discrepancy log |
| `Manuscript/draft/latex/` | Builds the PDF: `python3 build.py` (manuscript) and `bash build_additional.sh` (additional files) |
| `Manuscript/draft/versions_python.txt`, `versions_R.txt` | Software versions on the analysis server |
| `Models/*/Results/`, `Models/Model Comparison/` | Aggregated results used in the paper |

**Run order on the analysis server.** All scripts run from one working folder that holds the data. Files named `SERVER_ONLY_*` and the `split_seed_*.csv` files contain participant-level information and must not leave the server.

| Step | Script | Purpose |
|---|---|---|
| 1 | `Data Preprocessing/split_data.py` | Five twin-pair-grouped 75/25 splits shared by all models |
| 2 | `Data Preprocessing/Data Visualisation/analysis_sample_descriptives.R` | Participant flow and Table 1 |
| 3 | `Models/GLMM/Method/GLMM_updated_split.R` | GLMM fits and test metrics |
| 4 | `Models/GAMM/Method/GAMM_updated_split.R` | GAMM fits and test metrics |
| 5 | `Models/Model Comparison/glmm_gamm_test_predictions.R` | GLMM and GAMM test predictions (full and age-15 predictor sets) |
| 6 | `Models/DCMFNet/Analysis/dcmfnet_test_predictions.py` | Retrains DCMFNet for each seed, outcome and predictor set |
| 7 | `Models/ML/Method/ml_comparators_test_predictions.py` | Linear regression, elastic net and gradient boosting with grouped inner cross-validation |
| 8 | `Models/Model Comparison/prediction_evaluation.py` | Metrics, paired corrected t-tests, calibration, results by sex, age-15 analysis |
| 9 | `Models/DCMFNet/Analysis/shap_analysis.py` | SHAP values and gate values for DCMFNet |
| 10 | `Models/GLMM/Analysis/GLMM_shap_updated.R`, `Models/GAMM/Analysis/GAMM_shap_updated.R` | Exact SHAP values for the mixed models |
| 11 | `Models/Model Comparison/attribution_agreement.py`, `attribution_importance_plot.py` | Agreement between gates, SHAP and permutation importance |
| 12 | `Models/Model Comparison/interaction_ablation_mixed.R`, then `Models/DCMFNet/Analysis/dcmfnet_fusion_ablation.py` | Every model refitted without its SUD15 interactions (mixed models) or fusion blocks (DCMFNet) |

Supporting code:
- **Variable screening:** `Data Preprocessing/Feature Selection/feature_selection.py` and `Data Preprocessing/Data Analysis/Feature Importance/feature_importance.py`.
- **Default-setting screening of 40 regressors:** `Models/ML/Method/lazy_predict.py`.
- **DCMFNet:** the model, loss and training code are in `Models/DCMFNet/Method/` (`model.py`, `loss.py`, `legacy/train.py`, imported through `train.py`). The hyperparameters used are in `Models/DCMFNet/hyperparameters.json`; the file with the same name under `Results/Complexity Analysis/` is from an earlier search and gives different results.

`Manuscript/draft/server_handover_README.md` gives the full run order, expected values for checking a rerun and the data-protection rules.

---

## 2. Master's thesis

*Modeling Schizophrenia Symptom Severity: A Deep Learning and Statistical Approach to Substance Use Interactions in Adolescence.* The report is in `Thesis Report/`.

> The manuscript (section 1) re-analyses this work with all models on identical splits, tuned machine-learning comparators and paired statistical tests. Some of the thesis conclusions below, in particular the improvement from DCMFNet and the role of interactions, are not supported by that re-analysis.

Background: Schizophrenia is a multifactorial disorder shaped by genetic, environmental, and behavioral influences, yet how these factors interact during development remains unclear. Adolescent substance use is a key exposure, but its role as an independent risk factor versus an interaction-driven modifier remains debated.

Aim: To investigate how adolescent substance use interacts with genetic liability, adverse childhood experiences, and developmental factors to influence schizophrenia symptom severity, and to assess whether a deep learning framework improves modeling of these interactions.

Methods: Using data from the Swedish Twin Registry (CATSS), we applied Generalized Linear Mixed Models (GLMM), Generalized Additive Mixed Models (GAMM), and a Deep Cross-Modal Fusion Network (DCMFNet). The DCMFNet explicitly models interactions between substance use and multiple modalities. Models were evaluated using error-based and rank-based metrics, with interpretability assessed through feature and interaction importance.

Results: Risk was not driven by single factors but by interacting vulnerabilities. Adverse childhood experiences, prior symptoms, and genetic liability formed a consistent baseline, while substance use modified risk selectively. Positive symptoms showed stronger interaction effects, particularly with ACE and cell-type PRS, whereas negative symptoms were dominated by main effects with weaker interactions. DCMFNet provided modest but consistent improvements, especially in capturing rank-based structure.

Conclusion: Schizophrenia symptom severity is best understood through interacting risk factors. Substance use acts as a context-dependent amplifier rather than an independent driver. DCMFNet offers a flexible framework for modeling such interactions, complementing traditional statistical approaches

---

## 3. Bodhica synthetic-data workflow

> **Synthetic data only.** These models are trained and tested on data generated by `synthetic_data/`, not on CATSS. Their metrics show how well DCMFNet fits the synthetic generator. They say nothing about real-world performance and cannot be compared with the manuscript or thesis results.

The models trained here are served by [Bodhica](https://github.com/sil-pixel/Clinical-Risk-Agent), a research demonstration that is not a diagnostic or medical tool.

```bash
python generate_synthetic_data.py --n-samples 20000 --seed 42
python train.py
python export_model.py
```

Training outputs both `artifacts/dcmfnet_pos.pt` and `artifacts/dcmfnet_neg.pt`. Export writes both models to `exports/`, each with its corresponding `.metadata.json` file. Copy the exported inference files to the Bodhica repository and serve the required `.pt` artifacts with its local FastAPI runtime. See `Models/DCMFNet/Method/DEPLOYMENT.md` and `synthetic_data/SYNTHETIC_DATA.md`.

### Synthetic held-out metrics

Measured on a held-out part of the synthetic data.

| Model | Artifact | MAE | R² | Spearman's ρ | RMSE |
|---|---|---|---|---|---|
| Positive-symptom model | `artifacts/dcmfnet_pos.pt` | 0.0271 | 0.793 | 0.870 | 0.0339 |
| Negative-symptom model | `artifacts/dcmfnet_neg.pt` | 0.0532 | 0.374 | 0.609 | 0.0668 |
