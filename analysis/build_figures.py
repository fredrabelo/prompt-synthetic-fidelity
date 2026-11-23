"""Build the four paper figures from the result tables in results/."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
MODELS = {"haiku": "Claude Haiku 4.5", "ministral": "Ministral 3B", "qwen": "Qwen3 32B", "gptoss": "GPT-OSS-120B"}
fidelity = pd.read_csv(RESULTS / "fidelity.csv").set_index("predictor")
order = pd.read_csv(RESULTS / "order_effects.csv")
mirror = pd.read_csv(RESULTS / "adjusted_mirror.csv").set_index("model")
format_components = pd.read_csv(RESULTS / "format_decomposition.csv")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160})

keys = [model + "/" + condition for model in MODELS for condition in "BC"]
labels = [MODELS[model] + " " + condition for model in MODELS for condition in "BC"]
selected = fidelity.loc[keys]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
for ax, metric, title in zip(
    axes,
    ["mae", "r", "sd_pct"],
    ["Individual prediction error (MAE)", "Mean per-item correlation", "Mean per-item SD (% of human)"],
):
    values = selected[metric].to_numpy()
    lower = selected[metric + "_lo"].to_numpy()
    upper = selected[metric + "_hi"].to_numpy()
    ax.errorbar(values, np.arange(len(values)), xerr=[values - lower, upper - values], fmt="o", color="#247ba0", capsize=2)
    ax.axvline(fidelity.loc["OLS (OOF)", metric], color="#b34a29", linestyle="--", label="OOF OLS")
    ax.set_yticks(np.arange(len(values)), labels if ax is axes[0] else [""] * len(values))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
axes[0].legend()
fig.savefig(FIGURES / "fig1_fidelity.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
colors = ["#214e65", "#4b8da0", "#93c5c6"]
bottom = np.zeros(len(MODELS))
for component, color in zip(["respondent", "item", "respondent_item"], colors):
    values = format_components[format_components.component == component].set_index("model").loc[list(MODELS), "value"].to_numpy()
    ax.bar(np.arange(len(MODELS)), values, bottom=bottom, label=component.replace("_", " × "), color=color)
    bottom += values
ax.set_xticks(np.arange(len(MODELS)), MODELS.values())
ax.set_ylabel("Noise-adjusted contrast variance (points²)")
ax.legend()
fig.savefig(FIGURES / "fig2_format_components.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
for j, condition in enumerate("BC"):
    frame = order[order.condition == condition].set_index("model").loc[list(MODELS)]
    values = frame.signed_mean.to_numpy()
    ax.errorbar(np.arange(len(MODELS)) + (j - 0.5) * 0.16, values,
                yerr=[values - frame.lo.to_numpy(), frame.hi.to_numpy() - values], fmt="o", capsize=4, label=condition)
ax.axhline(0, color="gray", linewidth=1)
ax.set_xticks(np.arange(len(MODELS)), MODELS.values())
ax.set_ylabel("Reversed − original mean (points)")
ax.legend(title="Representation")
fig.savefig(FIGURES / "fig3_order_warming.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
frame = mirror.loc[list(MODELS)]
values = frame.ratio_all.to_numpy()
ax.errorbar(np.arange(len(MODELS)), values,
            yerr=[values - frame.ratio_all_lo.to_numpy(), frame.ratio_all_hi.to_numpy() - values],
            fmt="o", capsize=4, color="#247ba0")
ax.axhline(1, color="gray", linestyle="--")
ax.set_xticks(np.arange(len(MODELS)), MODELS.values())
ax.set_ylabel("Model / adjusted human contrast magnitude")
fig.savefig(FIGURES / "fig4_adjusted_mirror.png", bbox_inches="tight")
plt.close(fig)
print(f"Wrote four figures to {FIGURES}")
