"""Figure 1: the four-panel headline result (RQ1-RQ4, one panel each).

Usage: python3 make_figure1_headline.py
Requires: bootstrap_summary.json (run bootstrap_analysis.py first).
Writes: ../figures/figure1_headline.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SUMMARY_PATH = HERE / "bootstrap_summary.json"
OUT_PATH = HERE.parent / "figures" / "figure1_headline.png"

CONDITIONS = ["A_generic_llm", "B_persona_demographic", "C_population_grounded", "D_evidence_grounded"]
LABELS = ["A", "B", "C", "D"]
COLORS = ["#8b93a7", "#e66f83", "#e66f83", "#e66f83"]

# RQ4 (conditional fidelity) has no bootstrap CI - see the paper's Metrics
# section for why - so these come directly from conditional_fidelity_analysis.R's
# printed summary rather than bootstrap_summary.json.
CONDITIONAL_FAILURES = [43.0, 33.2, 34.3, 33.6]


def interval_errors(points, intervals):
    points = np.asarray(points)
    lows = np.asarray([bounds[0] for bounds in intervals])
    highs = np.asarray([bounds[1] for bounds in intervals])
    return np.vstack((points - lows, highs - points))


def main():
    if not SUMMARY_PATH.exists():
        raise SystemExit(f"{SUMMARY_PATH} not found — run bootstrap_analysis.py first.")
    with SUMMARY_PATH.open(encoding="utf-8") as stream:
        data = json.load(stream)

    summary = data["summary"]
    human_sd = data["human_sd"]
    x = np.arange(len(CONDITIONS))

    mae = [summary[key]["point"]["mae"] for key in CONDITIONS]
    mae_ci = [summary[key]["mae_ci"] for key in CONDITIONS]
    corr = [summary[key]["point"]["corr"] for key in CONDITIONS]
    corr_ci = [summary[key]["corr_ci"] for key in CONDITIONS]
    sd_pct = [summary[key]["point"]["sd_pct"] for key in CONDITIONS]
    sd_pct_ci = [[100 * value / human_sd for value in summary[key]["sd_ci"]] for key in CONDITIONS]

    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.1), constrained_layout=True)
    panels = [
        (mae, mae_ci, "A. Aggregate error", "Marginal MAE ↓", None),
        (corr, corr_ci, "B. Individual fidelity", "Human–synthetic correlation ↑", 0),
        (sd_pct, sd_pct_ci, "C. Distributional fidelity", "Synthetic SD / human SD (%) ↑", 100),
        (CONDITIONAL_FAILURES, None, "D. Conditional fidelity", "Coefficients different (%) ↓", 48),
    ]

    for axis, (values, intervals, title, ylabel, reference) in zip(axes, panels):
        axis.bar(x, values, color=COLORS, edgecolor="#333946", linewidth=0.8, width=0.68)
        if intervals is not None:
            axis.errorbar(
                x, values, yerr=interval_errors(values, intervals),
                fmt="none", ecolor="#202531", elinewidth=1.15, capsize=3, zorder=3,
            )
        if reference is not None:
            axis.axhline(reference, color="#626978", linestyle="--", linewidth=1.1)
        axis.set_title(title, fontsize=12.5, weight="bold", pad=9)
        axis.set_ylabel(ylabel, fontsize=10.5)
        axis.set_xticks(x, LABELS, fontsize=9.5)
        axis.grid(axis="y", color="#d9dce3", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0, 26)
    axes[1].set_ylim(-0.06, 0.54)
    axes[2].set_ylim(0, 108)
    axes[3].set_ylim(0, 54)

    for index, value in enumerate(mae):
        axes[0].text(index, value + 0.65, f"{value:.2f}", ha="center", fontsize=9)
    for index, value in enumerate(corr):
        axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    for index, value in enumerate(sd_pct):
        axes[2].text(index, value + 2.4, f"{value:.1f}%", ha="center", fontsize=9)
    for index, value in enumerate(CONDITIONAL_FAILURES):
        axes[3].text(index, value + 1.25, f"{value:.1f}%", ha="center", fontsize=9)

    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
