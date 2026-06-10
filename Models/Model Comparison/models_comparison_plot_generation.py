'''
Generate the plot for the models comparison.
Author: Silpa Soni Nallacheruvu
Date: 31/05/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
'''


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
data = {
    "Positive": {
        "R²":           {"GLMM":  (0.181, 0.020), "GAMM":  (0.180, 0.020), "DCMFNet": (0.206, 0.019)},
        "Spearman ρ":   {"GLMM":  (0.422, 0.021), "GAMM":  (0.420, 0.021), "DCMFNet": (0.450, 0.019)},
        "RMSE":         {"GLMM":  (0.094, 0.002), "GAMM":  (0.094, 0.002), "DCMFNet": (0.090, 0.001)},
        "MAE":          {"GLMM":  (0.073, 0.002), "GAMM":  (0.073, 0.002), "DCMFNet": (0.071, 0.000)},
    },
    "Negative": {
        "R²":           {"GLMM":  (0.179, 0.021), "GAMM":  (0.178, 0.021), "DCMFNet": (0.198, 0.021)},
        "Spearman ρ":   {"GLMM":  (0.435, 0.013), "GAMM":  (0.434, 0.010), "DCMFNet": (0.451, 0.013)},
        "RMSE":         {"GLMM":  (0.161, 0.005), "GAMM":  (0.162, 0.005), "DCMFNet": (0.156, 0.002)},
        "MAE":          {"GLMM":  (0.124, 0.003), "GAMM":  (0.124, 0.003), "DCMFNet": (0.124, 0.002)},
    },
}

# Metrics where lower is better (↓)
lower_is_better = {"RMSE"}

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
            y_top  = v + e
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

        # Set y limits with padding. Start at 0 so bar heights are honest.
        ymax = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(0, ymax * 1.18)

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

plt.savefig("model_comparison.pdf",
            bbox_inches="tight", dpi=200)
plt.savefig("model_comparison.png",
            bbox_inches="tight", dpi=200, facecolor="white")
print("Saved.")