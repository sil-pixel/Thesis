'''
Fusion ablation for DCMFNet: do the SUD15 fusion blocks improve prediction?
Date: 29/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

DCMFNet has two streams: the fused stream (one iterative gated fusion block per modality, each fusing SUD15 with
that modality) and the independent stream (all inputs together). The ablated network keeps only the independent
stream: all inputs -> squeeze-and-excitation attention -> final attention -> linear output, the same layers that
follow the concatenation in the full network, sized to the independent stream only. It is retrained with the same
seeds, splits, hyperparameters, loss and training procedure as the full network (dcmfnet_test_predictions.py).
The squeeze-and-excitation layers multiply inputs by learned weights, so the ablated network can still model some
interactions; the ablation removes the dedicated SUD15 fusion, not every interaction.

The script then compares, over the five splits, test R² with vs without
    - the SUD15 fusion blocks in DCMFNet (this script; full predictions from SERVER_ONLY_predictions_DCMFNet.csv),
    - the 24 SUD15 interaction terms in the GLMM and GAMM (interaction_ablation_mixed_per_seed.csv, if present),
with the corrected resampled t-test (as in prediction_evaluation.py) and a Holm correction across all comparisons.

Usage: run from the folder containing catss_final_data.csv, hyperparameters.json and the
SERVER_ONLY_predictions_DCMFNet.csv written by dcmfnet_test_predictions.py
(and interaction_ablation_mixed_per_seed.csv if interaction_ablation_mixed.R has been run).
    python dcmfnet_fusion_ablation.py                  # retrain the ablated network, then summarise
    python dcmfnet_fusion_ablation.py --summary-only   # only the summary (e.g. after rerunning the R script)

Output:
    SERVER_ONLY_predictions_DCMFNet_no_fusion.csv  participant-level; keep on the secure server
    fusion_ablation_dcmfnet_per_seed.csv           aggregated; can be exported
    interaction_ablation_summary.csv               aggregated; can be exported
'''

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcmfnet_test_predictions as base          # sets up the import path of the training code
import legacy.train as legacy_train
from dcmfnet.model import SEAttention

SEEDS = base.SEEDS
FULL_PREDICTIONS = "SERVER_ONLY_predictions_DCMFNet.csv"
OUT_PREDICTIONS = "SERVER_ONLY_predictions_DCMFNet_no_fusion.csv"
OUT_DCMFNET = "fusion_ablation_dcmfnet_per_seed.csv"
MIXED_FILE = "interaction_ablation_mixed_per_seed.csv"
OUT_SUMMARY = "interaction_ablation_summary.csv"
TEST_TRAIN_RATIO = 0.25 / 0.75              # as in prediction_evaluation.py


'''
DCMFNet without the fused stream. Same constructor signature as DeepCrossModalFusionModel, so train() can build it.
'''
class IndependentStreamModel(nn.Module):
    def __init__(self, M, L, n_features_per_modality, se_reduction=2, dropout=0.3, hidden_dim_min=8):
        super().__init__()
        self.M = M
        independent_dim = int(sum(n_features_per_modality))
        self.attn_independent = SEAttention(independent_dim, se_reduction, dropout, hidden_dim_min)
        self.attn_final = SEAttention(independent_dim, se_reduction, dropout, hidden_dim_min)
        self.fc = nn.Linear(independent_dim, 1)

    def forward(self, inputs):
        if len(inputs) != self.M + 2:
            raise ValueError(f"Expected {self.M + 2} modality tensors")
        raw_modalities = self.attn_independent(torch.cat(list(inputs), dim=-1))
        return self.fc(self.attn_final(raw_modalities))


def test_metrics(y, p):
    return {'r2': 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2), 'rmse': np.sqrt(np.mean((y - p) ** 2))}


def run_ablation():
    legacy_train.DCMFNet = IndependentStreamModel      # train() builds the model under this name

    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna().reset_index(drop=True)
    print(f"Data shape after dropping na: {df.shape}")
    modality_sizes = legacy_train.calculate_modality_sizes(df)
    sex_col = [c for c in df.columns if c.startswith("SEX")][0]
    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    full = pd.read_csv(FULL_PREDICTIONS)
    full = full[(full['model'] == 'DCMFNet') & (full['predictor_set'] == 'full')]

    predictions, rows = [], []
    for seed in SEEDS:
        print(f"\n{'='*60}\n  DCMFNet without fusion, Pos and Neg, seed {seed}\n{'='*60}")
        train_df, test_df = legacy_train.random_split(df, test_size=0.25, random_state=seed)
        models = base.train_seed_models(train_df, test_df, seed, modality_sizes, hyperparameters_json)

        for model_tag, outcome in [("Pos", "SCZ18_Pos_Norm"), ("Neg", "SCZ18_Neg_Norm")]:
            X_test, Y_test = legacy_train.prepare_data(test_df, model_tag)
            y_pred = base.predict(models[model_tag], X_test)
            predictions.append(pd.DataFrame({
                'model': 'DCMFNet_no_fusion', 'predictor_set': 'full', 'outcome': model_tag, 'seed': seed,
                'row_id': test_df.index.values, 'y_true': test_df[outcome].values, 'y_pred': y_pred,
                'sex': test_df[sex_col].values,
            }))
            pd.concat(predictions, ignore_index=True).to_csv(OUT_PREDICTIONS, index=False)

            # The full network's predictions must be for the same test participants
            f = full[(full['outcome'] == model_tag) & (full['seed'] == seed)]
            if sorted(f['row_id']) != sorted(test_df.index.values):
                raise ValueError(f"Test rows differ from {FULL_PREDICTIONS} for seed {seed}, {model_tag}")
            m_without = test_metrics(Y_test.flatten(), y_pred)
            m_with = test_metrics(f['y_true'].values, f['y_pred'].values)
            print(f"  {model_tag} test R²: with fusion {m_with['r2']:.4f}, without {m_without['r2']:.4f}")
            rows.append({'model': 'DCMFNet', 'outcome': model_tag, 'seed': seed, 'n_test': len(test_df),
                         'r2_with': m_with['r2'], 'r2_without': m_without['r2'],
                         'rmse_with': m_with['rmse'], 'rmse_without': m_without['rmse'],
                         'r2_difference': m_with['r2'] - m_without['r2']})
            pd.DataFrame(rows).to_csv(OUT_DCMFNET, index=False)

    print(f"\nSaved: {OUT_PREDICTIONS} (keep on the secure server) and {OUT_DCMFNET}")


'''
Corrected resampled t-test (Nadeau and Bengio, 2003) with a 95% confidence interval from the same corrected variance.
'''
def corrected_t_test(differences):
    k = len(differences)
    mean_d = np.mean(differences)
    var_d = np.var(differences, ddof=1)
    if k < 2 or var_d == 0:
        return mean_d, np.nan, np.nan, np.nan, np.nan
    se = np.sqrt((1 / k + TEST_TRAIN_RATIO) * var_d)
    t = mean_d / se
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    half = stats.t.ppf(0.975, df=k - 1) * se
    return mean_d, t, p, mean_d - half, mean_d + half


def summarise():
    per_seed = [pd.read_csv(OUT_DCMFNET)]
    if Path(MIXED_FILE).exists():
        per_seed.append(pd.read_csv(MIXED_FILE))
    else:
        print(f"{MIXED_FILE} not found: the summary covers DCMFNet only")
    per_seed = pd.concat(per_seed, ignore_index=True)

    rows = []
    for (model, outcome), g in per_seed.groupby(['model', 'outcome'], sort=False):
        g = g.sort_values('seed')
        d = g['r2_difference'].values
        mean_d, t, p, lo, hi = corrected_t_test(d)
        row = {'model': model, 'outcome': outcome, 'n_seeds': len(g),
               'r2_with_mean': g['r2_with'].mean(), 'r2_without_mean': g['r2_without'].mean(),
               'r2_difference_mean': mean_d, 'r2_difference_sd': np.std(d, ddof=1),
               'ci95_low': lo, 'ci95_high': hi, 'seeds_with_better': int(np.sum(d > 0)),
               't_corrected': t, 'p_value': p}
        for test in ['wald', 'lrt']:
            if f'{test}_p' in g and g[f'{test}_p'].notna().any():
                row[f'{test}_p_min'] = g[f'{test}_p'].min()
                row[f'{test}_p_max'] = g[f'{test}_p'].max()
                row[f'{test}_n_seeds_p_below_0.05'] = int((g[f'{test}_p'] < 0.05).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary['p_holm'] = holm(summary['p_value'].values)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUT_SUMMARY}")


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = valid.sum()
    order = np.argsort(p[valid])
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[valid][order]), 1)
    result = np.empty(m)
    result[order] = adj
    adjusted[valid] = result
    return adjusted


if __name__ == "__main__":
    if "--summary-only" not in sys.argv:
        run_ablation()
    summarise()
