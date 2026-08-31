"""Figure 3: human vs. synthetic attribute-response coefficients, reproducing
Bisbee et al. (2024)'s Figure 3 design.

Usage: python3 make_figure3_coefficients.py
Requires: conditional_fidelity_results.csv, produced by
          conditional_fidelity_analysis.R (run that first).
Writes: ../figures/figure3_coefficients.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "conditional_fidelity_results.csv"
OUT_PATH = HERE.parent / "figures" / "figure3_coefficients.png"

CONDITIONS = ["A_generic_llm", "B_persona_demographic", "C_population_grounded", "D_evidence_grounded"]
TITLES = ["A — No information", "B — Demographic persona", "C — Structured facts", "D — Structured + provenance"]


def main():
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found — run conditional_fidelity_analysis.R first.")
    df = pd.read_csv(RESULTS_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    lim = max(df.human_coef.abs().max(), df.llm_coef.abs().max()) * 1.05

    for ax, condition, title in zip(axes, CONDITIONS, TITLES):
        subset = df[df.condition == condition]
        significant = subset[subset.sig]
        not_significant = subset[~subset.sig]
        ax.scatter(not_significant.human_coef, not_significant.llm_coef, s=12, color="#c9c9d0", label="p ≥ .05", alpha=0.7)
        ax.scatter(significant.human_coef, significant.llm_coef, s=12, color="#E84A5F", label="p < .05", alpha=0.8)
        ax.plot([-lim, lim], [-lim, lim], "--", color="#1a1a2e", linewidth=0.8)
        ax.axhline(0, color="#ccc", linewidth=0.5)
        ax.axvline(0, color="#ccc", linewidth=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(title, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    axes[0].legend(fontsize=9, loc="upper left")
    axes[2].set_xlabel("Human coefficient", fontsize=10)
    axes[3].set_xlabel("Human coefficient", fontsize=10)
    axes[0].set_ylabel("Synthetic coefficient", fontsize=10)
    axes[2].set_ylabel("Synthetic coefficient", fontsize=10)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
