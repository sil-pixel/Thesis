'''
Figures 1 and 2 of the manuscript, and copies of Figures 3 and 4 from the analysis folders.
    Fig. 1: participant flow, from Data Preprocessing/Data Analysis/Sample Analysis/analysis_sample_flow.csv
            (aggregated counts; counts below 5 are merged with the neighbouring step so they are not shown).
    Fig. 2: DCMFNet architecture, drawn from Models/DCMFNet/Method/dcmfnet/model.py and the group sizes used
            in training ([6,16,23,19,17,7,4,4,4,1,4]); depths from Models/DCMFNet/hyperparameters.json.
    Fig. 3: calibration curves redrawn from Models/Model Comparison/calibration_bins.csv (decile means per model,
            outcome and split; written by prediction_evaluation.py) with the manuscript's labels.
    Fig. 4: Models/Model Comparison/attribution_importance_by_group.pdf
No participant-level data are read.

Usage: python3 make_figures.py   (any folder; needs matplotlib and pandas)
Output: fig1_flow.pdf/.png, fig2_architecture.pdf/.png, fig3_calibration.pdf/.png, fig4_attribution.pdf in this folder.
'''

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
REPO = FIG_DIR.parents[2]
FLOW_CSV = REPO / "Data Preprocessing" / "Data Analysis" / "Sample Analysis" / "analysis_sample_flow.csv"
HYPERPARAMS = REPO / "Models" / "DCMFNet" / "hyperparameters.json"
CALIBRATION_BINS = REPO / "Models" / "Model Comparison" / "calibration_bins.csv"
METRICS_PER_SEED = REPO / "Models" / "Model Comparison" / "prediction_metrics_per_seed.csv"
COPIES = {
    "fig5_attribution.pdf": REPO / "Models" / "Model Comparison" / "attribution_importance_by_group.pdf",
}

# Colour-blind safe palette (Okabe-Ito)
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#999999"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, text, face="white", edge="black", style="-", fontsize=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                facecolor=face, edgecolor=edge, linestyle=style, linewidth=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, color="black"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1,
                                                                  shrinkA=0, shrinkB=0))


'''
Fig. 1. Exclusions are counted at the first missing predictor group, in the order of the CSV.
Steps that exclude fewer than 5 individuals are merged with the step before them.
'''
def flow_steps():
    flow = pd.read_csv(FLOW_CSV)
    rows = flow.set_index("Step")["Rows"].astype(int)
    labels = {"PRS available (genotyped)": "No polygenic scores (not genotyped)",
              "SEX available": "Sex missing",
              "SCZ15 complete": "Symptoms at age 15 missing",
              "ADHD9 complete": "ADHD symptoms at age 9 missing",
              "ASD9 complete": "Autism-related traits at age 9 missing",
              "ACE15 complete": "Adverse experiences at age 15 missing",
              "ACE18 complete": "Adverse experiences at age 18 missing",
              "SUD18 complete": "Substance use at age 18 missing",
              "SES complete": "Parental socioeconomic data missing"}
    order = list(rows.index)
    steps, pending_label, pending_n = [], None, 0
    prev = rows.iloc[0]
    for step in order[1:]:
        n = int(prev - rows[step])
        prev = rows[step]
        if step not in labels or n == 0:
            continue
        if n < 5 and steps:
            last_label, last_n, _ = steps.pop()
            steps.append((last_label, last_n + n, True))
            continue
        steps.append((labels[step], n, False))
    # Merged steps: name the groups together
    named = []
    for label, n, merged in steps:
        if merged and label == "Symptoms at age 15 missing":
            label = "Symptoms at age 15 or neurodevelopmental\n   measures at age 9 missing"
        elif merged:
            raise ValueError(f"Add a label for the merged step '{label}'")
        named.append((label, n))
    flow_info = {"eligible": int(rows.iloc[0]), "eligible_pairs": int(flow["Twin_pairs"].iloc[0]),
                 "final": int(rows.iloc[-1]), "final_pairs": int(flow["Twin_pairs"].iloc[-1])}
    return named, flow_info


def fig1():
    steps, info = flow_steps()
    excluded = info["eligible"] - info["final"]
    assert excluded == sum(n for _, n in steps)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    cx, w = 0.23, 0.43
    # Starting count and the step to eligibility: thesis Figs 3.1 and 3.3 (author confirmed 2026-10-28, W-02);
    # 13,146 - 3,885 = 9,261, the rows of catss_final_data.csv.
    start, missing_main = 13146, 3885
    assert start - missing_main == info["eligible"]
    box(ax, cx, 0.95, w, 0.07, f"CATSS twins in the study data\nn = {start:,}", fontsize=8.5)
    box(ax, 0.745, 0.855, 0.49, 0.06, f"Excluded, n = {missing_main:,}\nSubstance use at age 15 or an outcome\nat age 18 missing",
        fontsize=7.4)
    arrow(ax, cx, 0.855, 0.745 - 0.49 / 2, 0.855)
    box(ax, cx, 0.75, w, 0.10, f"Eligible: substance use at age 15 and\nboth outcomes at age 18\n"
                               f"n = {info['eligible']:,} ({info['eligible_pairs']:,} twin pairs)", fontsize=8.5)
    box(ax, cx, 0.36, w, 0.10, f"Analysis sample (complete cases)\n"
                               f"n = {info['final']:,} ({info['final_pairs']:,} twin pairs)", weight="bold", fontsize=8.5)
    box(ax, cx, 0.12, w, 0.13, "Five random splits by twin pair\n(seeds 42–46), each 75% training\n"
                               "and 25% test (998–1,028 per test set);\nsame splits for all models", fontsize=8)
    arrow(ax, cx, 0.915, cx, 0.80)
    arrow(ax, cx, 0.70, cx, 0.41)
    arrow(ax, cx, 0.31, cx, 0.185)
    # Exclusion box
    ex_text = f"Excluded, n = {excluded:,}\n(counted at the first missing group)\n" + \
              "\n".join(f"• {label}: {n:,}" for label, n in steps)
    ex_x, ex_w, ex_y, ex_h = 0.745, 0.49, 0.555, 0.30
    ax.add_patch(FancyBboxPatch((ex_x - ex_w / 2, ex_y - ex_h / 2), ex_w, ex_h,
                                boxstyle="round,pad=0.01,rounding_size=0.015", facecolor="white", edgecolor="black"))
    ax.text(ex_x - ex_w / 2 + 0.015, ex_y + ex_h / 2 - 0.015, ex_text, ha="left", va="top", fontsize=7.4,
            linespacing=1.35)
    arrow(ax, cx, 0.555, ex_x - ex_w / 2, 0.555)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig1_flow.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


'''
Fig. 2. Left: the two streams. Right: one layer of an iterative gated fusion (IGF) block, Eqs 1-3 in Methods.
'''
def fig2():
    hp = json.load(open(HYPERPARAMS))
    groups = ["PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "Sex"]
    sizes = [16, 23, 19, 17, 7, 4, 4, 4, 1]
    pos_depths, neg_depth = hp["Pos"]["num_layers"], hp["Neg"]["num_layers"]

    fig = plt.figure(figsize=(10, 6.4))
    ax = fig.add_axes([0, 0, 0.66, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ys = [0.86 - i * 0.075 for i in range(9)]
    # Inputs
    ax.text(0.09, 0.97, "Inputs (105)", ha="center", weight="bold")
    box(ax, 0.09, 0.93 + 0.0, 0.15, 0.045, "SUD15 (6)\nanchor", face="#FBE3B5", edge=ORANGE, fontsize=7.5)
    for g, s, y in zip(groups, sizes, ys):
        box(ax, 0.09, y - 0.03, 0.15, 0.045, f"{g} ({s})", fontsize=7.5)
    box(ax, 0.09, ys[-1] - 0.105, 0.15, 0.045, "Batch × PC (4)", fontsize=7.5)
    # IGF blocks
    ax.text(0.37, 0.97, "Fused stream", ha="center", weight="bold", color=BLUE)
    for i, (g, y) in enumerate(zip(groups, ys)):
        yb = y - 0.03
        box(ax, 0.37, yb, 0.22, 0.05, f"IGF block: SUD15 × {g}\n$L$ = {pos_depths[i]} (PLE-HM), {neg_depth} (DEP)",
            face="#D6E9F5", edge=BLUE, fontsize=7)
        arrow(ax, 0.165, yb + 0.008, 0.26, yb + 0.008)
        box(ax, 0.535, yb, 0.05, 0.04, r"$\phi_{%d}$" % (i + 1), face="#EEEEEE", edge=GREY, fontsize=8)
        arrow(ax, 0.48, yb, 0.51, yb)
        arrow(ax, 0.56, yb, 0.60, 0.53)
    # Anchor feeds every block: a vertical line between the input column and the IGF blocks
    xa = 0.215
    ax.plot([0.165, xa], [0.93, 0.93], color=ORANGE, lw=1.4)
    ax.plot([xa, xa], [0.93, ys[-1] - 0.018], color=ORANGE, lw=1.4)
    for y in ys:
        arrow(ax, xa, y - 0.018, 0.26, y - 0.018, color=ORANGE)
    ax.text(xa + 0.008, 0.905, "anchor to\nevery block", fontsize=6.5, color=ORANGE, ha="left", va="top")
    # Concatenation and streams
    box(ax, 0.66, 0.53, 0.12, 0.07, "concat\n" + r"$\phi_F$", face="#D6E9F5", edge=BLUE, fontsize=8)
    ax.text(0.66, 0.085, "Independent stream", ha="center", weight="bold", color=GREEN)
    box(ax, 0.66, 0.15, 0.20, 0.07, "all 105 inputs\n" + r"$\phi_I$", face="#D5EFE6", edge=GREEN, fontsize=8)
    ax.plot([0.165, 0.20, 0.20, 0.56], [ys[-1] - 0.105, ys[-1] - 0.105, 0.15, 0.15], color=GREEN, lw=1)
    arrow(ax, 0.20, 0.15, 0.56, 0.15, color=GREEN)
    box(ax, 0.86, 0.36, 0.13, 0.08, "concat\n" + r"$\phi_{out}$", face="#EEEEEE", edge=GREY, fontsize=8)
    arrow(ax, 0.72, 0.53, 0.80, 0.40)
    arrow(ax, 0.76, 0.15, 0.80, 0.32)
    box(ax, 0.86, 0.20, 0.11, 0.06, "linear\n" + r"$\hat{y}$", face="white", fontsize=8)
    arrow(ax, 0.86, 0.32, 0.86, 0.23)
    ax.text(0.5, 0.01, r"$\phi$ = feature gate: $\mathbf{u} \odot \sigma(\mathbf{V}_2\,\mathrm{ReLU}(\mathbf{V}_1\mathbf{u}))$,"
                       " computed per individual (Eq. 4)", ha="center", fontsize=7.5)

    # Right panel: one IGF layer
    bx = fig.add_axes([0.67, 0.08, 0.32, 0.86]); bx.set_xlim(0, 1); bx.set_ylim(0, 1); bx.axis("off")
    bx.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.94, boxstyle="round,pad=0.01", facecolor="#F7FBFE",
                                edgecolor=BLUE, linestyle="--"))
    bx.text(0.5, 0.93, "One layer $l$ of IGF block $m$", ha="center", weight="bold", color=BLUE)
    box(bx, 0.25, 0.82, 0.32, 0.06, r"anchor $\mathbf{x}_a$", face="#FBE3B5", edge=ORANGE, fontsize=8)
    box(bx, 0.73, 0.82, 0.38, 0.06, r"$\mathbf{g}_m^{(l-1)}$ ($\mathbf{g}^{(0)}=\mathbf{x}_m$)", fontsize=8)
    box(bx, 0.5, 0.64, 0.80, 0.08, r"$\mathbf{f}=\tanh(\tanh(\mathbf{W}_1\mathbf{x}_a)\odot\tanh(\mathbf{W}_2\mathbf{g}))$"
                                   "\nmultiplicative fusion (Eq. 1)", fontsize=7.5)
    arrow(bx, 0.25, 0.79, 0.35, 0.685); arrow(bx, 0.73, 0.79, 0.65, 0.685)
    box(bx, 0.5, 0.46, 0.80, 0.08, r"$\mathbf{z}=\sigma(\mathbf{W}_z(\mathbf{f}+\mathbf{g}))$" "\ngate (Eq. 2)", fontsize=7.5)
    arrow(bx, 0.5, 0.60, 0.5, 0.505)
    box(bx, 0.5, 0.28, 0.80, 0.08, r"$\mathbf{g}_m^{(l)}=\tanh(\mathbf{W}_g[\mathbf{f};\,\mathbf{f}\odot\mathbf{z}])$"
                                   "\nupdate (Eq. 3)", fontsize=7.5)
    arrow(bx, 0.5, 0.42, 0.5, 0.325)
    bx.text(0.5, 0.12, "Block output: $[\\mathbf{f}^{(1)};\\dots;\\mathbf{f}^{(L_m)}]$\n"
                       "$\\mathbf{g}_m^{(l)}$ feeds layer $l+1$", ha="center", fontsize=7.5)
    arrow(bx, 0.5, 0.235, 0.5, 0.18)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig2_architecture.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


'''
Fig. 3. Decile means averaged over the five splits, as in calibration_curves_full.pdf. Model names follow the
manuscript tables; change MODEL_LABELS if the mixed model is relabelled (Q-01).
'''
MODEL_LABELS = {"GLMM": "GLMM", "GAMM": "GAMM", "DCMFNet": "DCMFNet", "LinearRegression": "Linear regression",
                "ElasticNet": "Elastic net", "GradientBoosting": "Gradient boosting"}
MODEL_STYLE = {"GLMM": (ORANGE, "-", "o"), "GAMM": ("#D55E00", "-", "s"), "DCMFNet": ("#000000", "-", "D"),
               "LinearRegression": ("#56B4E9", ":", "^"), "ElasticNet": (BLUE, ":", "v"),
               "GradientBoosting": (GREEN, ":", "P")}


ORDER = ["DCMFNet", "GLMM", "GAMM", "LinearRegression", "ElasticNet", "GradientBoosting"]
OUTCOMES = [("Pos", "PLE-HM"), ("Neg", "DEP")]


'''
Fig. 3. Test R² of every model on each of the five splits (all predictors). Lines join the results of the same
split, so the within-split (paired) differences are visible; the bar marks the mean over splits.
'''
def fig3():
    d = pd.read_csv(METRICS_PER_SEED)
    d = d[d["predictor_set"] == "full"]
    seeds = sorted(d["seed"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = range(len(ORDER))
    for ax, (tag, title) in zip(axes, OUTCOMES):
        sub = d[d["outcome"] == tag].pivot(index="seed", columns="model", values="r2")[ORDER]
        for seed in seeds:
            ax.plot(x, sub.loc[seed].values, color="#CCCCCC", lw=0.8, zorder=1)
        for k, model in enumerate(ORDER):
            color, _, marker = MODEL_STYLE[model]
            ax.scatter([k] * len(seeds), sub[model], color=color, marker=marker, s=22, zorder=2,
                       edgecolor="white", linewidth=0.4)
            mean = sub[model].mean()
            ax.plot([k - 0.28, k + 0.28], [mean, mean], color="black", lw=2, zorder=3)
            ax.text(k, 1.01, f"{mean:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, weight="bold")
        ax.axvspan(-0.5, 0.5, color="#F2F2F2", zorder=0)
        ax.axvline(2.5, color=GREY, lw=0.8, ls="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels([MODEL_LABELS[m].replace(" ", "\n", 1) if m in ("LinearRegression", "ElasticNet",
                            "GradientBoosting") else MODEL_LABELS[m] for m in ORDER], fontsize=8)
        ax.set_xlim(-0.5, len(ORDER) - 0.5)
        ax.set_title(title, weight="bold", pad=22)
        ax.text(-0.5, 1.01, "mean:", transform=ax.get_xaxis_transform(), ha="right", va="bottom", fontsize=7.5)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Test R²")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig3_performance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


'''
Fig. 4. Calibration, one panel per model so the curves do not overlap; the other models are shown in light grey
for reference. Deciles of predicted score, averaged over the five splits (all predictors).
'''
def fig4():
    bins = pd.read_csv(CALIBRATION_BINS)
    bins = bins[bins["predictor_set"] == "full"]
    curves = bins.groupby(["model", "outcome", "bin"])[["mean_predicted", "mean_observed"]].mean().reset_index()
    fig, axes = plt.subplots(2, len(ORDER), figsize=(12, 4.6), sharex="row", sharey="row")
    for row, (tag, title) in enumerate(OUTCOMES):
        sub = curves[curves["outcome"] == tag]
        low = min(sub["mean_predicted"].min(), sub["mean_observed"].min()) * 0.95
        high = max(sub["mean_predicted"].max(), sub["mean_observed"].max()) * 1.03
        for col, model in enumerate(ORDER):
            ax = axes[row, col]
            ax.plot([low, high], [low, high], color=GREY, ls="--", lw=0.9)
            for other in ORDER:
                if other != model:
                    o = sub[sub["model"] == other].sort_values("bin")
                    ax.plot(o["mean_predicted"], o["mean_observed"], color="#DDDDDD", lw=0.8)
            m = sub[sub["model"] == model].sort_values("bin")
            color, _, marker = MODEL_STYLE[model]
            ax.plot(m["mean_predicted"], m["mean_observed"], color=color, marker=marker, ms=3.5, lw=1.5)
            ax.set_xlim(low, high); ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=9, weight="bold")
            if col == 0:
                ax.set_ylabel(f"{title}\nmean observed", fontsize=8)
            if row == 1:
                ax.set_xlabel("mean predicted", fontsize=8)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig4_calibration.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    for name, source in COPIES.items():
        shutil.copy(source, FIG_DIR / name)
    print("Saved: fig1_flow, fig2_architecture, fig3_performance, fig4_calibration, fig5_attribution in", FIG_DIR)
