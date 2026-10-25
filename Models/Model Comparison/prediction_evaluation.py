'''
Prediction-level evaluation of DCMFNet, GLMM and GAMM on the shared test sets, with the same metric code for all models.
    1. Overall metrics per model, predictor set, outcome and seed (R², RMSE, MAE, Spearman, Pearson).
    2. Calibration: calibration-in-the-large (mean observed - mean predicted), calibration slope and intercept
       (observed regressed on predicted), and binned calibration curves.
    3. Sex-stratified metrics.
    4. Top-decile metrics: the 10% of test participants with the highest observed scores.
    5. Paired comparisons across the 5 seeds (corrected resampled t-test, Holm-adjusted):
       DCMFNet vs GLMM vs GAMM, and full vs age-15-only predictors for each model.
       If the ML comparator predictions are present (ml_comparators_test_predictions.py), DCMFNet is also compared
       with each comparator, and each comparator with itself (full vs age 15). These comparisons form a second
       Holm family ('baselines'), so the main comparisons keep the same correction as before.
Date: 23/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from a folder containing the prediction files written by dcmfnet_test_predictions.py and
glmm_gamm_test_predictions.R (SERVER_ONLY_predictions_DCMFNet.csv, _GLMM.csv, _GAMM.csv; missing files are skipped),
and optionally ml_comparators_test_predictions.py (SERVER_ONLY_predictions_ML.csv).
    python prediction_evaluation.py

Output (aggregated only; these files can be exported):
    prediction_metrics_per_seed.csv, prediction_metrics_summary.csv
    prediction_metrics_by_sex_per_seed.csv, prediction_metrics_by_sex_summary.csv
    paired_comparisons.csv
    calibration_bins.csv, calibration_curves_{predictor_set}.png
'''

import glob
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAIN_MODELS = ["GLMM", "GAMM", "DCMFNet"]
BASELINE_MODELS = ["LinearRegression", "ElasticNet", "GradientBoosting"]
MODELS = MAIN_MODELS + BASELINE_MODELS
SEX_LABELS = {1: "Male", 2: "Female"}       # coding as in descriptive_statistics.R
TOP_FRACTION = 0.10
N_CALIBRATION_BINS = 10
TEST_TRAIN_RATIO = 0.25 / 0.75              # twin pairs in test vs training, for the corrected t-test
PAIRED_METRICS = ["r2", "rmse", "mae", "spearman_rho", "calibration_slope"]


def core_metrics(y, p):
    residual = y - p
    return {
        'n': len(y),
        'r2': 1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2),
        'rmse': np.sqrt(np.mean(residual ** 2)),
        'mae': np.mean(np.abs(residual)),
        'spearman_rho': stats.spearmanr(y, p).correlation,
        'pearson_r': stats.pearsonr(y, p)[0],
    }


'''
Calibration slope and intercept from regressing observed on predicted; calibration-in-the-large is the mean
difference. A well-calibrated model has slope 1, intercept 0 and calibration-in-the-large 0.
'''
def calibration_metrics(y, p):
    slope, intercept = np.polyfit(p, y, 1)
    return {'calibration_in_the_large': y.mean() - p.mean(),
            'calibration_slope': slope, 'calibration_intercept': intercept}


'''
Top decile of the observed outcome in the test set. Ties at the threshold are included, so the observed
fraction can exceed 10%; it is reported as top_observed_fraction.
'''
def top_decile_metrics(y, p):
    threshold = np.quantile(y, 1 - TOP_FRACTION)
    is_top = y >= threshold
    k = int(np.ceil(TOP_FRACTION * len(y)))
    predicted_top = np.argsort(-p)[:k]
    return {
        'top_observed_fraction': is_top.mean(),
        'top_precision_at_10pct': is_top[predicted_top].mean(),
        'top_auroc': roc_auc_score(is_top, p) if 0 < is_top.sum() < len(y) else np.nan,
        'top_rmse': np.sqrt(np.mean((y[is_top] - p[is_top]) ** 2)),
        'top_mean_bias': np.mean(p[is_top] - y[is_top]),
    }


def summarise(per_seed, group_cols):
    metric_cols = [c for c in per_seed.columns if c not in group_cols + ['seed']]
    grouped = per_seed.groupby(group_cols)
    summary = grouped[metric_cols].mean().add_suffix('_mean')
    summary = summary.join(grouped[metric_cols].agg(lambda x: np.std(x, ddof=0)).add_suffix('_std'))
    summary['n_seeds'] = grouped['seed'].nunique()
    return summary.reset_index()


'''
Corrected resampled t-test (Nadeau and Bengio, 2003) for the per-seed differences between two settings
evaluated on the same 5 splits. The correction accounts for the overlap of training sets across splits.
'''
def corrected_t_test(differences):
    k = len(differences)
    mean_d = np.mean(differences)
    var_d = np.var(differences, ddof=1)
    if k < 2 or var_d == 0:
        return mean_d, np.nan, np.nan
    t = mean_d / np.sqrt((1 / k + TEST_TRAIN_RATIO) * var_d)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return mean_d, t, p


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    order = np.argsort(p[valid])
    m = valid.sum()
    adj_valid = np.maximum.accumulate((m - np.arange(m)) * p[valid][order])
    result = np.empty(m)
    result[order] = np.minimum(adj_valid, 1)
    adjusted[valid] = result
    return adjusted


def paired_comparisons(per_seed):
    keys = ['model', 'predictor_set', 'outcome']
    table = per_seed.set_index(keys + ['seed'])
    pairs = []
    for predictor_set in per_seed['predictor_set'].unique():
        present = set(per_seed[per_seed['predictor_set'] == predictor_set]['model'])
        main = [m for m in MAIN_MODELS if m in present]
        for a, b in combinations(main, 2):
            pairs.append(((a, predictor_set), (b, predictor_set), 'main'))
        if 'DCMFNet' in present:
            for baseline in [m for m in BASELINE_MODELS if m in present]:
                pairs.append((('DCMFNet', predictor_set), (baseline, predictor_set), 'baselines'))
    for model in [m for m in MODELS if m in set(per_seed['model'])]:
        if {'full', 'age15'} <= set(per_seed[per_seed['model'] == model]['predictor_set']):
            family = 'main' if model in MAIN_MODELS else 'baselines'
            pairs.append(((model, 'full'), (model, 'age15'), family))

    rows = []
    for (model_a, set_a), (model_b, set_b), family in pairs:
        for outcome in sorted(per_seed['outcome'].unique()):
            for metric in PAIRED_METRICS:
                try:
                    a = table.loc[(model_a, set_a, outcome)][metric]
                    b = table.loc[(model_b, set_b, outcome)][metric]
                except KeyError:
                    continue
                common = a.index.intersection(b.index)
                differences = (a.loc[common] - b.loc[common]).values
                mean_d, t, p = corrected_t_test(differences)
                rows.append({'family': family, 'outcome': outcome, 'metric': metric,
                             'setting_a': f"{model_a} ({set_a})", 'setting_b': f"{model_b} ({set_b})",
                             'n_seeds': len(common), 'mean_difference_a_minus_b': mean_d,
                             'sd_difference': np.std(differences, ddof=1) if len(common) > 1 else np.nan,
                             'seeds_a_better': int(np.sum(differences < 0)) if metric in ('rmse', 'mae')
                             else int(np.sum(differences > 0)),
                             't_corrected': t, 'p_value': p})
    result = pd.DataFrame(rows)
    # Holm correction within each family and metric, across all comparisons and outcomes
    result['p_holm'] = np.nan
    for _, idx in result.groupby(['family', 'metric']).groups.items():
        result.loc[idx, 'p_holm'] = holm(result.loc[idx, 'p_value'].values)
    return result


def calibration_bins(predictions):
    rows = []
    for (model, predictor_set, outcome, seed), g in predictions.groupby(['model', 'predictor_set', 'outcome', 'seed']):
        bins = pd.qcut(g['y_pred'].rank(method='first'), N_CALIBRATION_BINS, labels=False)
        for b, gb in g.groupby(bins):
            rows.append({'model': model, 'predictor_set': predictor_set, 'outcome': outcome, 'seed': seed,
                         'bin': int(b) + 1, 'n': len(gb),
                         'mean_predicted': gb['y_pred'].mean(), 'mean_observed': gb['y_true'].mean()})
    return pd.DataFrame(rows)


def plot_calibration(bins_df, predictor_set):
    df = bins_df[bins_df['predictor_set'] == predictor_set]
    averaged = df.groupby(['model', 'outcome', 'bin'])[['mean_predicted', 'mean_observed']].mean().reset_index()
    colors = {"GLMM": "#F0B8C0", "GAMM": "#E07080", "DCMFNet": "#6B0F1A",
              "LinearRegression": "#9EC5E8", "ElasticNet": "#4A86C5", "GradientBoosting": "#1F3F6E"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, outcome in zip(axes, ["Pos", "Neg"]):
        sub = averaged[averaged['outcome'] == outcome]
        for model in MODELS:
            m = sub[sub['model'] == model]
            if len(m):
                ax.plot(m['mean_predicted'], m['mean_observed'], marker='o', color=colors[model], label=model,
                        linestyle='-' if model in MAIN_MODELS else ':')
        lims = [min(sub['mean_predicted'].min(), sub['mean_observed'].min()),
                max(sub['mean_predicted'].max(), sub['mean_observed'].max())]
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=0.8, label='Ideal')
        ax.set_title(f"{outcome} outcome", fontweight='bold')
        ax.set_xlabel("Mean predicted (decile of prediction)")
        ax.set_ylabel("Mean observed")
        ax.legend(fontsize=8)
    fig.suptitle(f"Calibration, {predictor_set} predictors (bins averaged over 5 seeds)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"calibration_curves_{predictor_set}.png", dpi=300)
    plt.savefig(f"calibration_curves_{predictor_set}.pdf")
    plt.close()
    print(f"Saved: calibration_curves_{predictor_set}.png/.pdf")


if __name__ == "__main__":
    files = sorted(glob.glob("SERVER_ONLY_predictions_*.csv"))
    if not files:
        raise FileNotFoundError("No SERVER_ONLY_predictions_*.csv files in this folder")
    predictions = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {', '.join(files)}")
    print(predictions.groupby(['model', 'predictor_set', 'outcome'])['seed'].nunique().rename('seeds').to_string())

    # All models must be evaluated on the same test rows with the same observed values
    for (outcome, seed), g in predictions.groupby(['outcome', 'seed']):
        reference = None
        for (model, predictor_set), gm in g.groupby(['model', 'predictor_set']):
            observed = gm.set_index('row_id')['y_true'].sort_index()
            if reference is None:
                reference = observed
            elif not (observed.index.equals(reference.index) and np.allclose(observed.values, reference.values)):
                raise ValueError(f"{model} ({predictor_set}), {outcome}, seed {seed}: test rows differ from the "
                                 "other models. Check that all models used the same data file and splits.")
    print("Check passed: all models share the same test rows and observed values per outcome and seed.")

    group_cols = ['model', 'predictor_set', 'outcome']
    overall_rows, sex_rows = [], []
    for keys, g in predictions.groupby(group_cols + ['seed']):
        y, p = g['y_true'].values, g['y_pred'].values
        overall_rows.append({**dict(zip(group_cols + ['seed'], keys)),
                             **core_metrics(y, p), **calibration_metrics(y, p), **top_decile_metrics(y, p)})
        for sex_value, gs in g.groupby('sex'):
            ys, ps = gs['y_true'].values, gs['y_pred'].values
            sex_rows.append({**dict(zip(group_cols + ['seed'], keys)),
                             'sex': SEX_LABELS.get(sex_value, str(sex_value)),
                             **core_metrics(ys, ps), **calibration_metrics(ys, ps)})

    per_seed = pd.DataFrame(overall_rows)
    per_seed.to_csv("prediction_metrics_per_seed.csv", index=False)
    summarise(per_seed, group_cols).to_csv("prediction_metrics_summary.csv", index=False)
    print("Saved: prediction_metrics_per_seed.csv, prediction_metrics_summary.csv")

    sex_per_seed = pd.DataFrame(sex_rows)
    sex_per_seed.to_csv("prediction_metrics_by_sex_per_seed.csv", index=False)
    summarise(sex_per_seed, group_cols + ['sex']).to_csv("prediction_metrics_by_sex_summary.csv", index=False)
    print("Saved: prediction_metrics_by_sex_per_seed.csv, prediction_metrics_by_sex_summary.csv")

    paired = paired_comparisons(per_seed)
    paired.to_csv("paired_comparisons.csv", index=False)
    print("Saved: paired_comparisons.csv")
    print(paired[paired['metric'] == 'r2'][['family', 'outcome', 'setting_a', 'setting_b', 'mean_difference_a_minus_b',
                                            'seeds_a_better', 'p_value', 'p_holm']].to_string(index=False))

    bins_df = calibration_bins(predictions)
    bins_df.to_csv("calibration_bins.csv", index=False)
    print("Saved: calibration_bins.csv")
    for predictor_set in bins_df['predictor_set'].unique():
        plot_calibration(bins_df, predictor_set)
