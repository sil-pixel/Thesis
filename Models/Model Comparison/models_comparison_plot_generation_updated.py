'''
Generate the plot for the models comparison, using the updated GLMM and GAMM results.
Updated version of models_comparison_plot_generation.py (2026-10-22). Changes:
    1. The values are read from the summary CSVs instead of being typed into the script.
    2. GLMM and GAMM come from GLMM_updated_split.R and GAMM_updated_split.R: the same twin-pair splits
       as DCMFNet, R² = 1 - SSE/SST (as sklearn r2_score) and population SD across seeds (as np.std).
    3. Outputs are saved as model_comparison_updated.pdf/.png, so the original figure is not overwritten.
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from a folder containing
    GLMM_updated_summary.csv  (from GLMM_updated_split.R)
    GAMM_updated_summary.csv  (from GAMM_updated_split.R)
    test_results_summary.csv  (DCMFNet, from Models/DCMFNet/Results/Final Best Model)
    python models_comparison_plot_generation_updated.py
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────
SUMMARY_FILES = {
    "GLMM":    ("GLMM_updated_summary.csv", ","),
    "GAMM":    ("GAMM_updated_summary.csv", ","),
    "DCMFNet": ("test_results_summary.csv", ";"),
}

# Column prefix in the summary files for each metric
METRIC_COLUMNS = {"R²": "r2", "Spearman ρ": "spearman_rho", "RMSE": "rmse", "MAE": "mae"}


'''
Read one summary file and return {outcome: {metric: (mean, std)}}.
The Model column starts with Pos or Neg in all three files (Pos_GLMM_updated, Pos_GAMM_updated, Pos).
'''
def load_summary(path, sep):
    df = pd.read_csv(path, sep=sep)
    results = {}
    for outcome, prefix in [("Positive", "Pos"), ("Negative", "Neg")]:
        rows = df[df["Model"].str.startswith(prefix)]
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one {prefix} row, found {len(rows)}")
        row = rows.iloc[0]
        results[outcome] = {
            metric: (float(row[f"{col}_mean"]), float(row[f"{col}_std"]))
            for metric, col in METRIC_COLUMNS.items()
        }
    return results


summaries = {model: load_summary(path, sep) for model, (path, sep) in SUMMARY_FILES.items()}
data = {
    outcome: {
        metric: {model: summaries[model][outcome][metric] for model in SUMMARY_FILES}
        for metric in METRIC_COLUMNS
    }
    for outcome in ["Positive", "Negative"]
}

# Print the values used, so they can be checked against the summary files
for outcome in data:
    print(f"\n{outcome}")
    for metric in data[outcome]:
        values = ", ".join(f"{m}: {mean:.3f} ± {std:.3f}" for m, (mean, std) in data[outcome][metric].items())
        print(f"  {metric:<11} {values}")

# Metrics where lower is better (↓)
lower_is_better = {"RMSE", "MAE"}

# ── Style ─────────────────────────────────────────────────────────────────────
CRIMSON      = "#6B0F1A"   # DCMFNet bar
PINK_LIGHT   = "#F0B8C0"   # GLMM bar
PINK_MID     = "#E07080"   # GAMM bar
HEADER_BG    = "#6B0F1A"
HEADER_TEXT  = "white"
BAR_WIDTH    = 0.62
MODELS       = ["GLMM", "GAMM", "DCMFNet"]
COLORS       = [PINK_LIGHT, PINK_MID, CRIMSON]
METRICS      = ["R²", "Spearman ρ", "RMSE"]

def arrow(metric):
    return "↓" if metric in lower_is_better else "↑"

def metric_label(m):
    sym = arrow(m)
    if m == "R²":       return f"$R^2$ ({sym})"
    if m == "Spearman ρ": return f"Spearman $\\rho$ ({sym})"
    return f"{m} ({sym})"

# ── Figure ────────────────────────────────────────────────────────────────────
n_metrics = len(METRICS)
fig_width  = 3.0 * n_metrics          # more compact width
fig_height = 3.4                      # more compact height
fig, axes = plt.subplots(2, n_metrics, figsize=(fig_width, fig_height),
                         gridspec_kw={"hspace": 0.62, "wspace": 0.28})

for row, outcome in enumerate(["Positive", "Negative"]):
    for col, metric in enumerate(METRICS):
        ax = axes[row, col]
        vals  = [data[outcome][metric][m][0] for m in MODELS]
        errs  = [data[outcome][metric][m][1] for m in MODELS]
        xs    = np.arange(len(MODELS))

        bars = ax.bar(xs, vals, BAR_WIDTH, color=COLORS,
                      yerr=errs, capsize=2.5,
                      error_kw=dict(elinewidth=0.8, ecolor="#444444", capthick=0.8),
                      zorder=3)

        # Value labels above each bar
        for bar, v, e in zip(bars, vals, errs):
            x_pos  = bar.get_x() + bar.get_width() / 2
            y_top  = max(v, 0) + e
            ax.text(x_pos, y_top + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#222222")

        # Axis cosmetics
        ax.set_xticks(xs)
        ax.set_xticklabels(MODELS, fontsize=7)
        # Make DCMFNet x-tick label bold & crimson
        ax.get_xticklabels()[2].set_color(CRIMSON)
        ax.get_xticklabels()[2].set_fontweight("bold")

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.6)
        ax.tick_params(axis="x", which="both", length=0)
        ax.tick_params(axis="y", labelsize=6.5, length=2.5, color="#888888")

        # Set y limits with padding. Start at 0 so bar heights are honest
        # (1 - SSE/SST can be negative, so the lower limit follows the data in that case).
        ymax = max(v + e for v, e in zip(vals, errs))
        ymin = min(0, min(v - e for v, e in zip(vals, errs)))
        ax.set_ylim(ymin * 1.18, ymax * 1.18)

        ax.set_facecolor("#FAFAFA")
        ax.grid(axis="y", linewidth=0.4, color="#DDDDDD", zorder=0)
        ax.set_axisbelow(True)

        # Metric title (top row only)
        if row == 0:
            ax.set_title(metric_label(metric), fontsize=9, fontweight="bold", pad=6)

# ── Section headers ───────────────────────────────────────────────────────────
def add_header(fig, axes_row, label):
    # Use the bounding box of the first and last axis in the row
    bbox0 = axes_row[0].get_position()
    bbox1 = axes_row[-1].get_position()
    x0 = bbox0.x0
    x1 = bbox1.x1
    y1 = bbox0.y1 + 0.08   # slightly above the axes
    height = 0.040
    rect = plt.Rectangle((x0, y1), x1 - x0, height,
                          transform=fig.transFigure,
                          color=HEADER_BG, clip_on=False, zorder=5)
    fig.add_artist(rect)
    fig.text((x0 + x1) / 2, y1 + height / 2, label,
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color=HEADER_TEXT, zorder=6)

fig.canvas.draw()           # needed so get_position() is populated
add_header(fig, axes[0], "POSITIVE OUTCOMES")
add_header(fig, axes[1], "NEGATIVE OUTCOMES")

plt.savefig("model_comparison_updated.pdf",
            bbox_inches="tight", dpi=200)
plt.savefig("model_comparison_updated.png",
            bbox_inches="tight", dpi=200, facecolor="white")
print("\nSaved: model_comparison_updated.pdf and model_comparison_updated.png")
