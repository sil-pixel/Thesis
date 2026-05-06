'''
Benchmark visualization - best model per family.
Groups ML models into families, picks the best from each,
and creates a clean comparison plot.

Author: Silpa Soni Nallacheruvu
Date: 06/05/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage:
    python plot_benchmarks.py

Optionally add DCMFNet, GLMM, and GAMM results for comparison.
'''

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# -- Model family classification --
FAMILY_MAP = {
    # Regularized Linear
    'LassoCV': 'Regularized Linear',
    'LassoLarsCV': 'Regularized Linear',
    'ElasticNetCV': 'Regularized Linear',
    'LassoLarsIC': 'Regularized Linear',
    'RidgeCV': 'Regularized Linear',
    'Ridge': 'Regularized Linear',
    'ElasticNet': 'Regularized Linear',
    'Lasso': 'Regularized Linear',
    'LassoLars': 'Regularized Linear',
    'LarsCV': 'Regularized Linear',
    'Lars': 'Regularized Linear',

    # Linear (unregularized)
    'LinearRegression': 'Linear',
    'QuantileRegressor': 'Linear',

    # Bayesian
    'BayesianRidge': 'Bayesian',
    'HuberRegressor': 'Robust Linear',
    'SGDRegressor': 'Robust Linear',
    'PassiveAggressiveRegressor': 'Robust Linear',
    'RANSACRegressor': 'Robust Linear',

    # Kernel
    'KernelRidge': 'Kernel',
    'SVR': 'Kernel (SVM)',
    'NuSVR': 'Kernel (SVM)',
    'LinearSVR': 'Kernel (SVM)',
    'GaussianProcessRegressor': 'Kernel (GP)',

    # Tree-based
    'DecisionTreeRegressor': 'Decision Tree',
    'ExtraTreeRegressor': 'Decision Tree',

    # Ensemble
    'RandomForestRegressor': 'Ensemble (Bagging)',
    'ExtraTreesRegressor': 'Ensemble (Bagging)',
    'BaggingRegressor': 'Ensemble (Bagging)',
    'GradientBoostingRegressor': 'Ensemble (Boosting)',
    'HistGradientBoostingRegressor': 'Ensemble (Boosting)',
    'LGBMRegressor': 'Ensemble (Boosting)',
    'AdaBoostRegressor': 'Ensemble (Boosting)',

    # Neural Network
    'MLPRegressor': 'Neural Network',

    # Sparse
    'OrthogonalMatchingPursuitCV': 'Sparse',
    'OrthogonalMatchingPursuit': 'Sparse',

    # GLM
    'TweedieRegressor': 'GLM',
    'PoissonRegressor': 'GLM',

    # Neighbors
    'KNeighborsRegressor': 'Nearest Neighbors',

    # Meta
    'TransformedTargetRegressor': 'Meta-Estimator',

    # Baseline
    'DummyRegressor': 'Baseline',
}


# -- Optional: add your model results here --
# Set to None if not available
EXTRA_MODELS = {
    # "DCMFNet": {"family": "Deep Learning (Ours)", "rmse_mean": 0.082, "rmse_std": 0.003, "r2_mean": 0.30, "r2_std": 0.02, "spearman_rho_mean": 0.52, "spearman_rho_std": 0.01},
    # "GLMM": {"family": "Mixed Effects", "rmse_mean": 0.085, "rmse_std": 0.002, "r2_mean": 0.25, "r2_std": 0.015, "spearman_rho_mean": 0.48, "spearman_rho_std": 0.01},
    # "GAMM": {"family": "Mixed Effects", "rmse_mean": 0.084, "rmse_std": 0.002, "r2_mean": 0.27, "r2_std": 0.015, "spearman_rho_mean": 0.50, "spearman_rho_std": 0.01},
}


def load_and_clean(filepath):
    '''Load benchmark CSV, handle semicolon separator.'''
    try:
        df = pd.read_csv(filepath, sep=';')
    except Exception:
        df = pd.read_csv(filepath)

    # Assign families
    df['Family'] = df['Model'].map(FAMILY_MAP).fillna('Other')

    # Drop models with missing R2 or negative R2 (worse than predicting the mean)
    df = df.dropna(subset=['r2_mean'])
    df = df[df['r2_mean'] > 0]

    return df


def get_best_per_family(df):
    '''Pick the best model (lowest RMSE) from each family.'''
    best = df.loc[df.groupby('Family')['rmse_mean'].idxmin()]
    best = best.sort_values('rmse_mean', ascending=True).reset_index(drop=True)
    return best


def add_extra_models(best_df):
    '''Add DCMFNet, GLMM, GAMM etc. if provided.'''
    extra_rows = []
    for name, vals in EXTRA_MODELS.items():
        if vals is not None:
            row = {'Model': name, 'Family': vals['family']}
            for key in ['rmse_mean', 'rmse_std', 'r2_mean', 'r2_std',
                        'spearman_rho_mean', 'spearman_rho_std']:
                row[key] = vals.get(key, np.nan)
            extra_rows.append(row)

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        best_df = pd.concat([best_df, extra_df], ignore_index=True)
        best_df = best_df.sort_values('rmse_mean', ascending=True).reset_index(drop=True)

    return best_df


def plot_family_comparison(best_df, model_tag):
    '''
    Three-panel dot plot: RMSE, R2, Spearman rho.
    One dot per family (best model), color-coded by family.
    '''
    best_df = best_df.sort_values('rmse_mean', ascending=False)
    n = len(best_df)

    # Create display labels: "Family (Model)"
    labels = [f"{row['Family']}\n({row['Model']})" for _, row in best_df.iterrows()]

    # Color by family
    families = best_df['Family'].unique()
    cmap = plt.cm.Set2(np.linspace(0, 1, len(families)))
    family_colors = dict(zip(families, cmap))
    colors = [family_colors[f] for f in best_df['Family']]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, n * 0.45)))
    y_pos = np.arange(n)

    metrics = [
        ('rmse_mean', 'rmse_std', 'RMSE (lower is better)'),
        ('r2_mean', 'r2_std', 'R2 (higher is better)'),
        ('spearman_rho_mean', 'spearman_rho_std', 'Spearman rho (higher is better)'),
    ]

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        means = best_df[mean_col].values
        stds = best_df[std_col].fillna(0).values

        ax.errorbar(y_pos, means, xerr=stds,
                     fmt='none', ecolor='gray', elinewidth=1, capsize=3, zorder=1)
        ax.scatter(means, y_pos, c=colors, s=100, zorder=2,
                   edgecolors='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(title, fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        # Value labels
        for j, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                ax.annotate(f'{m:.4f}±{s:.4f}', (m, j),
                            textcoords="offset points", xytext=(10, 0),
                            fontsize=7, color='gray')

    plt.suptitle(f'{model_tag} SCZ - Best Model per Family (mean ± std, 5 seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_tag}_benchmark_by_family.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_tag}_benchmark_by_family.png")


def plot_compact_bar(best_df, model_tag):
    '''
    Single compact horizontal bar chart of R2 with RMSE annotated.
    Good for thesis where space is limited.
    '''
    best_df = best_df.sort_values('r2_mean', ascending=True)
    n = len(best_df)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))

    labels = [f"{row['Family']} ({row['Model']})" for _, row in best_df.iterrows()]
    y_pos = np.arange(n)

    # Color: highlight top 3 by R2
    colors = ['#e74c3c' if i >= n - 1 else '#2ecc71' if i >= n - 3 else '#3498db'
              for i in range(n)]

    bars = ax.barh(y_pos, best_df['r2_mean'], xerr=best_df['r2_std'],
                   color=colors, alpha=0.8, capsize=3, edgecolor='white')

    # Annotate with RMSE
    for j, (_, row) in enumerate(best_df.iterrows()):
        rmse_text = f"RMSE={row['rmse_mean']:.4f}"
        ax.annotate(rmse_text, (row['r2_mean'] + row['r2_std'] + 0.005, j),
                    fontsize=7, color='gray', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('R2 (mean ± std, 5 seeds)', fontsize=11)
    ax.set_title(f'{model_tag} SCZ - Model Family Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, best_df['r2_mean'].max() + best_df['r2_std'].max() + 0.05)

    plt.tight_layout()
    plt.savefig(f'{model_tag}_benchmark_compact.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_tag}_benchmark_compact.png")


def save_family_table(best_df, model_tag):
    '''Save the best-per-family table as CSV.'''
    out = best_df[['Family', 'Model', 'rmse_mean', 'rmse_std',
                    'r2_mean', 'r2_std', 'spearman_rho_mean', 'spearman_rho_std']].copy()
    out.columns = ['Family', 'Best Model', 'RMSE (mean)', 'RMSE (std)',
                   'R2 (mean)', 'R2 (std)', 'Spearman rho (mean)', 'Spearman rho (std)']
    out = out.sort_values('RMSE (mean)', ascending=True)
    out.to_csv(f'{model_tag}_benchmark_by_family.csv', index=False)
    print(f"Saved: {model_tag}_benchmark_by_family.csv")


if __name__ == "__main__":
    for model_tag in ["Pos", "Neg"]:
        filepath = f"{model_tag}_benchmark_summary.csv"
        if not os.path.exists(filepath):
            print(f"Skipping {model_tag}: {filepath} not found")
            continue

        print(f"\n{'='*50}")
        print(f"  {model_tag} SCZ - Benchmark by Family")
        print(f"{'='*50}")

        df = load_and_clean(filepath)
        print(f"  Loaded {len(df)} valid models from {len(df['Family'].unique())} families")

        best_df = get_best_per_family(df)
        best_df = add_extra_models(best_df)
        print(f"  Best per family: {len(best_df)} models")

        for _, row in best_df.iterrows():
            print(f"    {row['Family']:<25} {row['Model']:<35} "
                  f"RMSE={row['rmse_mean']:.4f}  R2={row['r2_mean']:.4f}")

        plot_family_comparison(best_df, model_tag)
        plot_compact_bar(best_df, model_tag)
        save_family_table(best_df, model_tag)