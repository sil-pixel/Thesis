'''
Fig. 4 of the manuscript: importance of each predictor group by attribution source, for both outcomes.
Each cell is the share of the group in the total importance of that source (mean over the 5 seeds), from
attribution_modality_summary.csv written by attribution_agreement.py. Shares make the sources comparable,
since gate values, |SHAP| and drops in R² are on different scales.
    - Gate sources: mean gate value of the group divided by the sum over groups. The fused stream has no gate
      for SUD15 (the anchor) or for batch*PC (not fused); these cells are left empty.
    - Permutation sources: negative importances (shuffling did not hurt the model) count as 0.
Date: 25/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from the folder containing attribution_modality_summary.csv (aggregated; no participant data needed).
    python attribution_importance_plot.py

Output: attribution_importance_by_group.png / .pdf

The shares describe what each fitted model uses for prediction, not causal effects of the predictors.
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCES = ["DCMFNet_gate_fused", "DCMFNet_gate_independent", "DCMFNet_SHAP", "DCMFNet_permutation",
           "GLMM_SHAP", "GAMM_SHAP", "GLMM_permutation", "GAMM_permutation"]
SOURCE_LABELS = {
    "DCMFNet_gate_fused": "DCMFNet gate (fused)", "DCMFNet_gate_independent": "DCMFNet gate (independent)",
    "DCMFNet_SHAP": "DCMFNet SHAP", "DCMFNet_permutation": "DCMFNet permutation",
    "GLMM_SHAP": "GLMM SHAP", "GAMM_SHAP": "GAMM SHAP",
    "GLMM_permutation": "GLMM permutation", "GAMM_permutation": "GAMM permutation",
}
OUTCOME_LABELS = {"Pos": "PLE-HM", "Neg": "DEP"}


'''
Group order shared by both panels: by the mean GLMM SHAP share over the two outcomes, largest first.
'''
def group_order(shares):
    reference = shares[shares['source'] == "GLMM_SHAP"].groupby('modality')['share_mean'].mean()
    return reference.sort_values(ascending=False).index.tolist()


def share_matrix(shares, model_tag, groups):
    sub = shares[shares['model_tag'] == model_tag]
    matrix = sub.pivot(index='modality', columns='source', values='share_mean')
    return matrix.reindex(index=groups, columns=[s for s in SOURCES if s in matrix.columns])


if __name__ == "__main__":
    shares = pd.read_csv("attribution_modality_summary.csv")
    groups = group_order(shares)
    vmax = shares['share_mean'].max()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, model_tag in zip(axes, ["Pos", "Neg"]):
        matrix = share_matrix(shares, model_tag, groups)
        values = np.ma.masked_invalid(matrix.values.astype(float))
        cmap = matplotlib.colormaps["Reds"].copy()
        cmap.set_bad("#EEEEEE")
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = matrix.values[i, j]
                text = "–" if np.isnan(v) else f"{100 * v:.0f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if not np.isnan(v) and v > 0.6 * vmax else "black")
        # Separate the DCMFNet gate columns from the model-agnostic attributions
        ax.axvline(1.5, color="black", linewidth=1)
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels([SOURCE_LABELS[s] for s in matrix.columns], fontsize=8, rotation=40, ha="right",
                           rotation_mode="anchor")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=9)
        ax.set_title(OUTCOME_LABELS[model_tag], fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Share of total importance (%, mean over 5 splits)",
                 format=matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}"))
    plt.savefig("attribution_importance_by_group.png", dpi=300, bbox_inches="tight")
    plt.savefig("attribution_importance_by_group.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: attribution_importance_by_group.png / .pdf")
