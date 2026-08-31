"""Figure 2: human vs. synthetic response distributions (violin plots) for
the three thermometers with the highest human variance in the sample.

Usage: python3 make_figure2_distributions.py
Reads: ../data/human_benchmark.csv, ../data/synthetic_responses.csv
Writes: ../figures/figure2_distributions.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent.parent / "figures" / "figure2_distributions.png"

THERMOMETERS = ["thermometer_republican_party", "thermometer_democratic_party", "thermometer_liberals"]
TITLES = ["Republican Party", "Democratic Party", "Liberals"]
CONDITIONS = ["A_generic_llm", "B_persona_demographic", "C_population_grounded", "D_evidence_grounded"]
GROUP_LABELS = ["Human\n(ANES)", "A\nNo info", "B\nPersona", "C\nStructured", "D\nStruct.+\nprov."]
COLORS = ["#2E7D32", "#9a9aa5", "#E84A5F", "#E84A5F", "#E84A5F"]


def main():
    human = pd.read_csv(DATA_DIR / "human_benchmark.csv")
    synth = pd.read_csv(DATA_DIR / "synthetic_responses.csv")
    synth["synthetic_value"] = pd.to_numeric(synth["synthetic_value"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)

    for ax, thermometer, title in zip(axes, THERMOMETERS, TITLES):
        datasets = [human[human.thermometer == thermometer].human_value.dropna().values]
        for condition in CONDITIONS:
            values = synth[(synth.condition == condition) & (synth.thermometer == thermometer)].synthetic_value.dropna().values
            datasets.append(values)

        parts = ax.violinplot(datasets, showmeans=True, showextrema=False, widths=0.8)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS[i])
            body.set_edgecolor("#1a1a2e")
            body.set_alpha(0.75)
        parts["cmeans"].set_color("#1a1a2e")
        parts["cmeans"].set_linewidth(1.5)

        ax.set_xticks(range(1, 6))
        ax.set_xticklabels(GROUP_LABELS, fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(-5, 105)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel("Feeling thermometer (0-100)", fontsize=10)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
