'''
Agreement between attribution methods for DCMFNet and the updated GLMM and GAMM, at the modality level.
Questions answered:
    1. Do DCMFNet's attention gates rank the modalities like SHAP and permutation importance on the same model?
    2. Do DCMFNet and the GLMM (and GAMM) rely on the same modalities?
    3. How stable is each attribution across the 5 seeds?
Date: 23/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from a folder containing
    DCMFNet_attribution_per_seed.csv (from Models/DCMFNet/Analysis/shap_analysis.py)
    GLMM_attribution_per_seed.csv    (from Models/GLMM/Analysis/GLMM_shap_updated.R)
    GAMM_attribution_per_seed.csv    (optional, from Models/GAMM/Analysis/GAMM_shap_updated.R)
    python attribution_agreement.py

Output:
    attribution_agreement_summary.csv : Spearman rank correlation and top-3 overlap between each pair of
                                        sources, per seed, then mean and SD across seeds
    attribution_stability.csv         : mean Spearman correlation between seeds, per source
    attribution_modality_summary.csv  : share of total importance per modality and source, mean and SD across seeds
    attribution_agreement_{tag}.png   : heatmap of the mean Spearman correlation between sources

With 9 to 11 modalities, rank correlations are coarse; read them together with the modality shares.
'''

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCES = ["DCMFNet_gate_independent", "DCMFNet_gate_fused", "DCMFNet_SHAP", "DCMFNet_permutation",
           "GLMM_SHAP", "GLMM_permutation", "GAMM_SHAP", "GAMM_permutation"]
TOP_K = 3


def top_k_overlap(a, b, k=TOP_K):
    return len(set(a.nlargest(k).index) & set(b.nlargest(k).index)) / k


'''
Spearman correlation and top-k overlap for every pair of sources, per model and seed.
Each pair is compared on the modalities both sources have (the fused gates cover 9 of the 11 modalities).
'''
def pairwise_agreement(df):
    rows = []
    for (model_tag, seed), group in df.groupby(['model_tag', 'seed']):
        table = group.pivot(index='modality', columns='source', values='importance')
        for source_a, source_b in combinations([s for s in SOURCES if s in table.columns], 2):
            pair = table[[source_a, source_b]].dropna()
            rho = spearmanr(pair[source_a], pair[source_b]).correlation
            rows.append({'model_tag': model_tag, 'seed': seed, 'source_a': source_a, 'source_b': source_b,
                         'n_modalities': len(pair), 'spearman': rho,
                         'top3_overlap': top_k_overlap(pair[source_a], pair[source_b])})
    per_seed = pd.DataFrame(rows)
    summary = (per_seed.groupby(['model_tag', 'source_a', 'source_b', 'n_modalities'])
               .agg(spearman_mean=('spearman', 'mean'),
                    spearman_std=('spearman', lambda x: np.std(x, ddof=0)),
                    top3_overlap_mean=('top3_overlap', 'mean'),
                    n_seeds=('seed', 'nunique'))
               .reset_index())
    return per_seed, summary


'''
Stability of each source: mean Spearman correlation of the modality ranking between every pair of seeds.
'''
def seed_stability(df):
    rows = []
    for (model_tag, source), group in df.groupby(['model_tag', 'source']):
        table = group.pivot(index='modality', columns='seed', values='importance').dropna()
        rhos = [spearmanr(table[s1], table[s2]).correlation for s1, s2 in combinations(table.columns, 2)]
        rows.append({'model_tag': model_tag, 'source': source, 'n_modalities': len(table),
                     'between_seed_spearman_mean': np.mean(rhos),
                     'between_seed_spearman_min': np.min(rhos)})
    return pd.DataFrame(rows)


'''
Share of the total importance per modality, so sources on different scales can be read side by side.
Negative permutation importances (shuffling did not hurt the model) are set to 0 before taking shares.
'''
def modality_shares(df):
    df = df.copy()
    df['clipped'] = df['importance'].clip(lower=0)
    totals = df.groupby(['model_tag', 'seed', 'source'])['clipped'].transform('sum')
    df['share'] = np.where(totals > 0, df['clipped'] / totals, np.nan)
    return (df.groupby(['model_tag', 'source', 'modality'])
            .agg(importance_mean=('importance', 'mean'),
                 importance_std=('importance', lambda x: np.std(x, ddof=0)),
                 share_mean=('share', 'mean'),
                 share_std=('share', lambda x: np.std(x, ddof=0)))
            .reset_index())


def plot_agreement(summary, model_tag):
    sub = summary[summary['model_tag'] == model_tag]
    sources = [s for s in SOURCES if s in set(sub['source_a']) | set(sub['source_b'])]
    matrix = pd.DataFrame(np.eye(len(sources)), index=sources, columns=sources)
    for _, row in sub.iterrows():
        matrix.loc[row['source_a'], row['source_b']] = row['spearman_mean']
        matrix.loc[row['source_b'], row['source_a']] = row['spearman_mean']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(sources)))
    ax.set_yticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(sources, fontsize=9)
    for i in range(len(sources)):
        for j in range(len(sources)):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman ρ (mean over seeds)")
    ax.set_title(f"{model_tag} — agreement of modality rankings", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"attribution_agreement_{model_tag}.png", dpi=300)
    plt.close()
    print(f"Saved: 'attribution_agreement_{model_tag}.png'")


if __name__ == "__main__":
    dcmfnet = pd.read_csv("DCMFNet_attribution_per_seed.csv")
    glmm = pd.read_csv("GLMM_attribution_per_seed.csv")
    frames = [dcmfnet, glmm]
    if os.path.exists("GAMM_attribution_per_seed.csv"):
        frames.append(pd.read_csv("GAMM_attribution_per_seed.csv"))
    else:
        print("GAMM_attribution_per_seed.csv not found: GAMM sources are left out")
    df = pd.concat(frames, ignore_index=True)

    unknown = set(df['source']) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unexpected sources: {unknown}")
    for model_tag in ["Pos", "Neg"]:
        seeds = {s: set(df[(df['model_tag'] == model_tag) & (df['source'] == s)]['seed'])
                 for s in SOURCES if s in set(df['source'])}
        if len({frozenset(v) for v in seeds.values()}) != 1:
            print(f"WARNING: {model_tag} sources do not cover the same seeds: {seeds}")

    per_seed, summary = pairwise_agreement(df)
    summary.to_csv("attribution_agreement_summary.csv", index=False)
    per_seed.to_csv("attribution_agreement_per_seed.csv", index=False)
    print("Saved: 'attribution_agreement_summary.csv' and 'attribution_agreement_per_seed.csv'")

    stability = seed_stability(df)
    stability.to_csv("attribution_stability.csv", index=False)
    print("Saved: 'attribution_stability.csv'")

    shares = modality_shares(df)
    shares.to_csv("attribution_modality_summary.csv", index=False)
    print("Saved: 'attribution_modality_summary.csv'")

    for model_tag in ["Pos", "Neg"]:
        print(f"\n{model_tag}: Spearman ρ between sources (mean ± SD over seeds)")
        sub = summary[summary['model_tag'] == model_tag]
        for _, row in sub.iterrows():
            print(f"  {row['source_a']:<26} vs {row['source_b']:<26} "
                  f"{row['spearman_mean']:+.2f} ± {row['spearman_std']:.2f}  (n = {row['n_modalities']})")
        plot_agreement(summary, model_tag)
