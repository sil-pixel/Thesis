# DCMFNet paper: handover note for the analysis server

**Written by:** [your name], [date]. My data access ends on 31 October 2026.
**For:** [supervisor names], for any analyses requested during peer review (journal: BMC Psychiatry).
**Code:** GitHub repository `[repo URL]`, commit `[hash of the final run]`. Each script's header explains its inputs and outputs in more detail.

---

## 1. Data protection rules

- **Never export** `catss_final_data.csv`, the `split_seed_*.csv` files or any `SERVER_ONLY_*` file. The split files hold pseudonymised twin-pair IDs; the `SERVER_ONLY_*` files hold one row per participant.
- **Only export** the aggregated outputs listed in section 4 (summaries over seeds, deciles or groups). The descriptives script already shows counts below 5 as "<5".
- **Logs:** check them before exporting. Training logs contain only summary metrics and bin counts.

## 2. Folder layout

Everything runs from one working folder, which holds both the data and the scripts:

```
[server path]/
  catss_final_data.csv           # analysis data (9,261 rows; 4,033 complete cases)
  split_seed_42.csv ... _46.csv  # twin-pair train/test splits shared by all models (from split_data.py)
  hyperparameters.json           # DCMFNet hyperparameters (see the warning below)
  train.py, model.py, loss.py    # DCMFNet training code (Models/DCMFNet/Method/), imported by the analysis scripts
  GLMM_updated_split.R, GAMM_updated_split.R   # also sourced by the R scripts below
  ... all scripts listed in section 3
```

**Warning: use the right `hyperparameters.json`.** It must be the file in `Models/DCMFNet/hyperparameters.json`:
- Pos `num_layers` = `[5, 2, 1, 5, 4, 1, 1, 1, 5]`;
- Neg `num_layers` = `2`.

The file with the same name in `Results/Complexity Analysis/Avg complexity results/` belongs to an earlier depth search. Using it changes the DCMFNet results. This happened once (D-23, 24 Oct 2026).

**Software:** versions are saved in `versions_R.txt` (`sessionInfo()`) and `versions_python.txt` (`pip freeze`) in this folder. ``

## 3. Run order

| Step | Command | Main outputs | Notes |
|---|---|---|---|
| 1 | `python split_data.py` | `split_seed_<seed>.csv`, `split_summary.csv` | Already run. Do **not** rerun unless the data file changes; it reproduces the same splits. |
| 2 | `Rscript analysis_sample_descriptives.R` | `analysis_sample_flow.csv`, `table1_included_vs_excluded.csv`, `missing_code_check.csv`, `analysis_sample_twin_structure.csv` | Table 1 and Fig. 1 |
| 3 | `Rscript GLMM_updated_split.R` | `GLMM_updated_summary.csv`, `GLMM_updated_{Pos,Neg}_per_seed.csv`, coefficient tables, diagnostic plots | Mixed-model main results (Table 3) |
| 4 | `Rscript GAMM_updated_split.R` | Same set with the `GAMM_` prefix, plus PRS smooth terms | The slowest R step |
| 5 | `Rscript GLMM_Analysis_updated.R`, `Rscript GAMM_Analysis_updated.R` | Forest and interaction plots | Use the outputs of steps 3–4 |
| 6 | `Rscript glmm_gamm_test_predictions.R` | `SERVER_ONLY_predictions_{GLMM,GAMM}.csv` | Full and age-15-only predictor sets. Stays on the server. |
| 7 | `python dcmfnet_test_predictions.py` | `SERVER_ONLY_predictions_DCMFNet.csv` | Retrains DCMFNet for 5 seeds × 2 outcomes × 2 predictor sets. Stays on the server. |
| 7b | `python ml_comparators_test_predictions.py` | `SERVER_ONLY_predictions_ML.csv`, `ML_comparators_tuning.csv` | Linear regression, elastic net and gradient boosting, tuned inside each training split (grouped inner CV). Stays on the server, except the tuning file. |
| 8 | `python prediction_evaluation.py` | `prediction_metrics_*`, `paired_comparisons.csv`, `calibration_bins.csv`, `calibration_curves_*` | Needs steps 6–7 (7b optional; its comparisons are a separate Holm family). Gives calibration, sex-specific, top-decile and age-15 results and the paired tests. |
| 9 | `python shap_analysis.py` | `DCMFNet_attribution_per_seed.csv`, `DCMFNet_retrained_test_metrics.csv`, SHAP plots | Retrains DCMFNet; needs `pip install shap` |
| 10 | `Rscript GLMM_shap_updated.R`, `Rscript GAMM_shap_updated.R` | `{GLMM,GAMM}_attribution_per_seed.csv` | The log should show "SHAP additivity check ... ~0" (about 1e-17) |
| 11 | `python attribution_agreement.py` | `attribution_agreement_*`, `attribution_stability.csv`, `attribution_modality_summary.csv` | Needs the three attribution files from steps 9–10 |
| 12 | `python models_comparison_plot_generation_updated.py` | `model_comparison_updated.{pdf,png}` | Needs the summary CSVs from steps 3–4 and the DCMFNet `test_results_summary.csv` |
| — | `python lazy_predict.py` | `{Pos,Neg}_benchmark_*` | Machine-learning baselines, already run. It uses the same splits, so there is no need to rerun. |

Steps 6–11 depend on steps 1, 3 and 4 only through the shared scripts and splits. You can rerun any of them on its own.

**DCMFNet training order.** Steps 7 and 9 train the Pos and Neg models of each seed together (`train_seed_models()` in `dcmfnet_test_predictions.py`). The seed is set once, then Pos is trained and evaluated, then Neg is trained, exactly as in the final run. Do not reseed between the two models: the Neg model would then start from different weights, and its R² changes (mean 0.190 instead of 0.198; D-23).

## 4. Checks that a rerun reproduces the paper

| Quantity | Expected value | File |
|---|---|---|
| Analysis sample | 4,033 individuals, 2,690 twin pairs; test sets 998–1,028 | `analysis_sample_flow.csv`, `prediction_metrics_per_seed.csv` |
| GLMM test R², full predictors | PLE-HM 0.228 ± 0.016, DEP 0.202 ± 0.022 | `GLMM_updated_summary.csv` |
| GAMM test R², full predictors | PLE-HM 0.227 ± 0.015, DEP 0.200 ± 0.023 | `GAMM_updated_summary.csv` |
| DCMFNet test R² per seed (42–46) | PLE-HM 0.229, 0.198, 0.226, 0.181, 0.195; DEP 0.202, 0.220, 0.219, 0.186, 0.165 (reproduced on 24 Oct 2026; DEP seed 42 gave 0.20234 vs 0.20226) | `DCMFNet_retrained_test_metrics.csv` vs `Results/Final Best Model/*_test_results_per_seed.csv` |

The R models reproduce exactly. DCMFNet retraining was reproducible to 7 digits on this server (identical seeds and splits), but may differ on other hardware.

## 5. Likely reviewer requests and where to change them

- **Another predictor subset** (for example, without genetic data): add a set to `PREDICTOR_SETS` in `glmm_gamm_test_predictions.R` and `dcmfnet_test_predictions.py`, following the `age15` example. Then rerun steps 6–8.
- **More splits:** add seeds to `SEEDS` in `split_data.py` and all scripts. This requires rerunning everything from step 1, and the new splits will not match the published ones.
- **Different tuning of DCMFNet** (for example, nested selection without the test set): not scripted. It would need a new tuning run (`tuning.py`) inside each training split. Please discuss with me first. `[your contact after 31 October]`
- **Other metrics or subgroups:** add them to `prediction_evaluation.py`. It reads the existing `SERVER_ONLY_*` prediction files, so no model needs refitting.

Please export new results to the repository under the same folder names (`Models/.../Results/`, `Models/Model Comparison/`). That keeps the manuscript sources (`Manuscript/draft/`) traceable.
