'''
Benchmark visualization - best model per family.
Groups ML models into families, picks the best from each,
and creates a clean comparison plot.
DCMFNet, GLMM, and GAMM are highlighted with red markers and bold labels.

Author: Silpa Soni Nallacheruvu
Date: 06/05/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage:
    python plot_benchmarks.py
'''

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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


# Models to highlight
HIGHLIGHT_MODELS = {'DCMFNet', 'GLMM', 'GAMM'}

# -- Optional: add your model results here --
EXTRA_MODELS = {
    "Pos" : {
        "DCMFNet": {"family": "Deep Learning (Ours)", "rmse_mean": 0.09, "rmse_std": 0.001, "r2_mean": 0.21, "r2_std": 0.02, "spearman_rho_mean": 0.45, "spearman_rho_std": 0.019},
        "GLMM": {"family": "Mixed Effects", "rmse_mean": 0.094, "rmse_std": 0.002, "r2_mean": 0.181, "r2_std": 0.020, "spearman_rho_mean": 0.422, "spearman_rho_std": 0.021},
        "GAMM": {"family": "Mixed Effects", "rmse_mean": 0.094, "rmse_std": 0.002, "r2_mean": 0.1805, "r2_std": 0.019, "spearman_rho_mean": 0.4205, "spearman_rho_std": 0.021}
    },
    "Neg" : {
        "DCMFNet": {"family": "Deep Learning (Ours)", "rmse_mean": 0.156, "rmse_std": 0.002, "r2_mean": 0.198, "r2_std": 0.021, "spearman_rho_mean": 0.451, "spearman_rho_std": 0.013},
        "GLMM": {"family": "Mixed Effects", "rmse_mean": 0.161, "rmse_std": 0.005, "r2_mean": 0.179, "r2_std": 0.021, "spearman_rho_mean": 0.435, "spearman_rho_std": 0.012},
        "GAMM": {"family": "Mixed Effects", "rmse_mean": 0.1615, "rmse_std": 0.005, "r2_mean": 0.178, "r2_std": 0.021, "spearman_rho_mean": 0.434, "spearman_rho_std": 0.01}
    }
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


def add_extra_models(best_df, model_tag):
    extra_rows = []
    MODEL_ITEMS = EXTRA_MODELS.get(model_tag, {})
    for name, vals in MODEL_ITEMS.items():
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


def is_highlighted(model_name):
    return model_name in HIGHLIGHT_MODELS


def plot_family_comparison(best_df, model_tag):
    '''
    Three-panel dot plot with highlighted models.
    '''
    best_df = best_df.sort_values('rmse_mean', ascending=False)
    n = len(best_df)
 
    labels = [f"{row['Family']}\n({row['Model']})" for _, row in best_df.iterrows()]
    highlight_mask = [is_highlighted(row['Model']) for _, row in best_df.iterrows()]
 
    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, n * 0.5)))
    y_pos = np.arange(n)
 
    metrics = [
        ('rmse_mean', 'rmse_std', 'RMSE (lower is better)'),
        ('r2_mean', 'r2_std', 'R² (higher is better)'),
        ('spearman_rho_mean', 'spearman_rho_std', 'Spearman ρ (higher is better)'),
    ]
 
    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        means = best_df[mean_col].values
        stds = best_df[std_col].fillna(0).values
 
        # Plot non-highlighted models
        for j in range(n):
            if not highlight_mask[j]:
                ax.errorbar(means[j], y_pos[j], xerr=stds[j],
                            fmt='o', color='#3498db', ecolor='gray',
                            elinewidth=1, capsize=3, markersize=6, zorder=2)
 
        # Plot highlighted models on top
        for j in range(n):
            if highlight_mask[j]:
                ax.errorbar(means[j], y_pos[j], xerr=stds[j],
                            fmt='D', color='#00897B', ecolor='#00695C',
                            elinewidth=1.5, capsize=4, markersize=9,
                            markeredgecolor='#004D40', markeredgewidth=1.5, zorder=3)
                # Red highlight box behind the label
                ax.annotate(
                    '', xy=(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else means[j] - stds[j] * 3, y_pos[j]),
                    xytext=(0, 0), textcoords='offset points'
                )
 
        # Y-axis labels: bold for highlighted
        label_texts = []
        for j, label in enumerate(labels):
            if highlight_mask[j]:
                label_texts.append(label)
            else:
                label_texts.append(label)
 
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
 
        # Bold the highlighted labels
        for j, tick_label in enumerate(ax.get_yticklabels()):
            if highlight_mask[j]:
                tick_label.set_fontweight('bold')
                tick_label.set_color('#00695C')
                tick_label.set_fontsize(10)
 
        # Add red background band for highlighted models
        for j in range(n):
            if highlight_mask[j]:
                ax.axhspan(y_pos[j] - 0.4, y_pos[j] + 0.4,
                           color='#00897B', alpha=0.08, zorder=0)
 
        ax.set_xlabel(title, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
 
        # Value labels
        for j, (m, s) in enumerate(zip(means, stds)):
            if not np.isnan(m):
                weight = 'bold' if highlight_mask[j] else 'normal'
                color = '#00695C' if highlight_mask[j] else 'gray'
                ax.annotate(f'{m:.4f}±{s:.4f}', (m, y_pos[j]),
                            textcoords="offset points", xytext=(12, 0),
                            fontsize=7, color=color, fontweight=weight)
 
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#00897B',
               markeredgecolor='#004D40', markersize=10, label='Our Models (DCMFNet / GLMM / GAMM)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='ML Baselines (best per family)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
 
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })
    plt.suptitle(f'{model_tag} SCZ - Best Model per Family (mean ± std, 5 seeds)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'{model_tag}_benchmark_by_family.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_tag}_benchmark_by_family.pdf")
 
 
def plot_compact_bar(best_df, model_tag):
    '''
    Horizontal bar chart of R² with highlighted models.
    '''
    best_df = best_df.sort_values('r2_mean', ascending=True)
    n = len(best_df)
 
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.4)))
 
    labels = [f"{row['Family']} ({row['Model']})" for _, row in best_df.iterrows()]
    highlight_mask = [is_highlighted(row['Model']) for _, row in best_df.iterrows()]
    y_pos = np.arange(n)
 
    # Colors: red for highlighted, green for top ML, blue for rest
    colors = []
    for i, (_, row) in enumerate(best_df.iterrows()):
        if is_highlighted(row['Model']):
            colors.append('#00897B')
        #elif i >= n - 3 and not any(is_highlighted(best_df.iloc[j]['Model']) for j in range(max(0, n-3), n)):
        #    colors.append('#2ecc71') 
        else:
            colors.append('#3498db')
 
    # Edge colors: thick red border for highlighted
    edgecolors = ['#004D40' if highlight_mask[i] else 'white' for i in range(n)]
    linewidths = [2.0 if highlight_mask[i] else 0.5 for i in range(n)]
 
    bars = ax.barh(y_pos, best_df['r2_mean'].values, xerr=best_df['r2_std'].values,
                   color=colors, alpha=0.85, capsize=3,
                   edgecolor=edgecolors, linewidth=linewidths)
 
    # Annotate with RMSE
    for j, (_, row) in enumerate(best_df.iterrows()):
        weight = 'bold' if is_highlighted(row['Model']) else 'normal'
        color = '#00695C' if is_highlighted(row['Model']) else 'gray'
        rmse_text = f"RMSE={row['rmse_mean']:.4f}"
        ax.annotate(rmse_text, (row['r2_mean'] + row['r2_std'] + 0.005, j),
                    fontsize=7, color=color, va='center', fontweight=weight)
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
 
    # Bold highlighted labels
    for j, tick_label in enumerate(ax.get_yticklabels()):
        if highlight_mask[j]:
            tick_label.set_fontweight('bold')
            tick_label.set_color('#00695C')
            tick_label.set_fontsize(10)
 
    # Red background bands
    for j in range(n):
        if highlight_mask[j]:
            ax.axhspan(y_pos[j] - 0.4, y_pos[j] + 0.4,
                       color='#00897B', alpha=0.08, zorder=0)
 
    ax.set_xlabel('R² (mean ± std, 5 seeds)', fontsize=11)
    ax.set_title(f'{model_tag} SCZ - Model Family Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, best_df['r2_mean'].max() + best_df['r2_std'].max() + 0.06)
 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00897B', edgecolor='#004D40', linewidth=2, label='Our Models'),
        Patch(facecolor='#3498db', edgecolor='white', label='ML Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
 
    plt.tight_layout()
    plt.savefig(f'{model_tag}_benchmark_compact.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_tag}_benchmark_compact.png")
 
def plot_spearman_compact_bar(best_df, model_tag):

    '''
    Horizontal bar chart of spearman rho with highlighted models.
    '''
    best_df = best_df.sort_values('spearman_rho_mean', ascending=True)
    n = len(best_df)
 
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.4)))
 
    labels = [f"{row['Family']} ({row['Model']})" for _, row in best_df.iterrows()]
    highlight_mask = [is_highlighted(row['Model']) for _, row in best_df.iterrows()]
    y_pos = np.arange(n)
 
    # Colors: red for highlighted, green for top ML, blue for rest
    colors = []
    for i, (_, row) in enumerate(best_df.iterrows()):
        if is_highlighted(row['Model']):
            colors.append('#00897B')
        #elif i >= n - 3 and not any(is_highlighted(best_df.iloc[j]['Model']) for j in range(max(0, n-3), n)):
        #    colors.append('#2ecc71') 
        else:
            colors.append('#3498db')
 
    # Edge colors: thick red border for highlighted
    edgecolors = ['#004D40' if highlight_mask[i] else 'white' for i in range(n)]
    linewidths = [2.0 if highlight_mask[i] else 0.5 for i in range(n)]
 
    bars = ax.barh(y_pos, best_df['spearman_rho_mean'].values, xerr=best_df['spearman_rho_std'].values,
                   color=colors, alpha=0.85, capsize=3,
                   edgecolor=edgecolors, linewidth=linewidths)
 
    # Annotate with RMSE
    for j, (_, row) in enumerate(best_df.iterrows()):
        weight = 'bold' if is_highlighted(row['Model']) else 'normal'
        color = '#00695C' if is_highlighted(row['Model']) else 'gray'
        rmse_text = f"RMSE={row['rmse_mean']:.4f}"
        ax.annotate(rmse_text, (row['spearman_rho_mean'] + row['spearman_rho_std'] + 0.005, j),
                    fontsize=7, color=color, va='center', fontweight=weight)
 
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
 
    # Bold highlighted labels
    for j, tick_label in enumerate(ax.get_yticklabels()):
        if highlight_mask[j]:
            tick_label.set_fontweight('bold')
            tick_label.set_color('#00695C')
            tick_label.set_fontsize(10)
 
    # Red background bands
    for j in range(n):
        if highlight_mask[j]:
            ax.axhspan(y_pos[j] - 0.4, y_pos[j] + 0.4,
                       color='#00897B', alpha=0.08, zorder=0)
 
    ax.set_xlabel('Spearman ρ (mean ± std, 5 seeds)', fontsize=11)
    ax.set_title(f'{model_tag} SCZ - Model Family Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, best_df['spearman_rho_mean'].max() + best_df['spearman_rho_std'].max() + 0.06)
 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00897B', edgecolor='#004D40', linewidth=2, label='Our Models'),
        Patch(facecolor='#3498db', edgecolor='white', label='ML Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
 
    plt.tight_layout()
    plt.savefig(f'{model_tag}_spearman_benchmark_compact.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_tag}_spearman_benchmark_compact.png")
 

def save_family_table(best_df, model_tag):
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
        best_df = add_extra_models(best_df, model_tag)
        print(f"  Best per family: {len(best_df)} models")

        for _, row in best_df.iterrows():
            marker = " ◄" if is_highlighted(row['Model']) else ""
            print(f"    {row['Family']:<25} {row['Model']:<35} "
                  f"RMSE={row['rmse_mean']:.4f}  R2={row['r2_mean']:.4f}{marker}")

        plot_family_comparison(best_df, model_tag)
        plot_compact_bar(best_df, model_tag)
        plot_spearman_compact_bar(best_df, model_tag)
        save_family_table(best_df, model_tag)