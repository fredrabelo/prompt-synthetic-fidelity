"""Figures, generated from the pipeline CSVs (no hand-typed values).
Usage: python3 make_figures.py [results_dir=../results] [out_dir=../figures]
Conventions: one condition = one fixed colour across figures (B blue, C orange, D aqua; A neutral grey as the baseline); one panel
per model (small multiples) with shared axes; points with thin 95% intervals; direct labels; PNG at 300 dpi plus vector PDF.
Each figure has a matching table in tables.md (build_tables.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL = {"A": "#8a8a85", "B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2dc"
NAMES = {"haiku": "Claude Haiku 4.5", "ministral": "Ministral 3B", "qwen": "Qwen3 32B", "gptoss": "GPT-OSS-120B"}
ORDER = ["haiku", "ministral", "qwen", "gptoss"]
CLAB = {"A": "A  none", "B": "B  narrative", "C": "C  fields", "D": "D  mirror"}
GLAB = {"republican_party": "Republican Party", "democratic_party": "Democratic Party", "liberals": "Liberals", "conservatives": "Conservatives", "muslims": "Muslims",
        "gays_lesbians": "Gays & Lesbians", "christians": "Christians", "asian_americans": "Asian Americans", "black_americans": "Black Americans",
        "white_americans": "White Americans", "jews": "Jews"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.facecolor": "white", "figure.facecolor": "white"})


def style(ax, gridaxis="y"):
    ax.grid(axis=gridaxis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.5)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def dot(ax, x, v, lo, hi, color, marker="o", size=22, z=3):
    ax.plot([x, x], [lo, hi], color=color, lw=1.2, solid_capstyle="round", zorder=z - 1)
    ax.scatter([x], [v], s=size, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=z)


def fig_fidelity(rd, out, models):
    ft = pd.read_csv(rd / "a2_fidelity_table.csv"); ft = ft[ft.subset == "all"]
    rows = [("mae", "MAE (points)", None), ("corr", "Correlation", None), ("sd_pct", "SD, % of human", 100)]
    fig, axes = plt.subplots(3, len(models), figsize=(1.9 * len(models) + 1.3, 6.2), sharey="row", squeeze=False)
    for j, m in enumerate(models):
        for i, (k, lab, ref) in enumerate(rows):
            ax = axes[i, j]; style(ax)
            for x, c in enumerate("ABC"):
                r = ft[(ft.model == m) & (ft.cond == c)].iloc[0]
                dot(ax, x, r[k], r[f"{k}_lo"], r[f"{k}_hi"], COL[c])
            if ref:
                ax.axhline(ref, color=INK2, lw=0.7, ls=(0, (3, 3)))
                if j == 0:
                    ax.text(-0.45, ref, "human", ha="left", va="bottom", fontsize=7, color=INK2)
            ax.set_xlim(-0.5, 2.5); ax.set_xticks(range(3)); ax.set_xticklabels(["A", "B", "C"])
            if j == 0: ax.set_ylabel(lab, fontsize=8)
            if i == 0: ax.set_title(NAMES[m], fontsize=9, color=INK, loc="left")
    fig.text(0.5, -0.01, "A = no information · B = narrative persona · C = structured fields · 95% respondent-cluster bootstrap intervals (often narrower than the markers)", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(); save(fig, out, "fig1_fidelity")


def fig_noise(rd, out, models):
    rl = pd.read_csv(rd / "a4_reliability_table.csv")
    gv = lambda m, u, s: rl[(rl.model == m) & (rl.unit == u) & (rl.stat == s)].iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    style(a1, "x"); style(a2, "x")
    y = np.arange(len(models))[::-1]
    for yi, m in zip(y, models):
        for dy, (u, c, lab) in zip((0.22, 0, -0.22), (("A-B", COL["A"], "A–B"), ("B-C", COL["B"], "B–C"))):
            pass
        nz = gv(m, "B-C", "noise_mad_avg").value
        xb = gv(m, "B-C", "cross_mad"); xa = gv(m, "A-B", "cross_mad")
        a1.scatter([nz], [yi + 0.2], s=26, marker="D", color=INK2, zorder=3)
        dot(a1, 0, 0, 0, 0, "white", size=0)
        a1.scatter([xb.value], [yi], s=26, color=COL["B"], zorder=3); a1.plot([xb.lo, xb.hi], [yi, yi], color=COL["B"], lw=1.2)
        a1.scatter([xa.value], [yi - 0.2], s=26, color=COL["A"], zorder=3); a1.plot([xa.lo, xa.hi], [yi - 0.2, yi - 0.2], color=COL["A"], lw=1.2)
        for u, c in (("B-C", COL["B"]), ("A-B", COL["A"])):
            e = gv(m, u, "excess_mad")
            yy = yi if u == "B-C" else yi - 0.2
            a2.plot([e.lo, e.hi], [yy, yy], color=c, lw=1.2); a2.scatter([e.value], [yy], s=26, color=c, zorder=3)
    a1.set_yticks(y); a1.set_yticklabels([NAMES[m] for m in models]); a1.set_xlabel("Mean absolute difference between responses (points)")
    a2.set_yticks(y); a2.set_yticklabels([]); a2.set_xlabel("Excess over same-prompt noise (points)"); a2.axvline(0, color=INK2, lw=0.7)
    a1.scatter([], [], s=26, marker="D", color=INK2, label="same prompt (noise)"); a1.scatter([], [], s=26, color=COL["B"], label="B vs C"); a1.scatter([], [], s=26, color=COL["A"], label="A vs B")
    h, l = a1.get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04)); a1.set_title("Distance between conditions and generation noise", fontsize=8.5, loc="left")
    a2.set_title("Distance beyond ordinary noise", fontsize=8.5, loc="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, out, "fig2_noise")


def fig_order(rd, out, models):
    od = pd.read_csv(rd / "a5_order_table.csv")
    fig, axes = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 1.0, 3.0), sharey=True, squeeze=False)
    for j, m in enumerate(models):
        ax = axes[0, j]; style(ax)
        for x, c in enumerate("ABCD"):
            r = od[(od.model == m) & (od.cond == c)].iloc[0]
            dot(ax, x, r.mad, r.mad_lo, r.mad_hi, COL[c])
            ax.scatter([x], [r.noise_mad_same_prompt], s=22, marker="_", color=INK, linewidths=1.6, zorder=4)
        ax.set_xlim(-0.5, 3.5); ax.set_xticks(range(4)); ax.set_xticklabels(list("ABCD")); ax.set_title(NAMES[m], fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel("Mean absolute reversed–original difference (points)", fontsize=8)
            ax.scatter([], [], s=22, marker="_", color=INK, linewidths=1.6, label="same-prompt noise"); ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    fig.tight_layout(); save(fig, out, "fig3_order")


def fig_mirror(rd, out, models):
    pt = pd.read_csv(rd / "a6_mirror_per_thermometer.csv")
    pt["g"] = pt.thermometer.str.replace("thermometer_", "")
    fig, axes = plt.subplots(1, len(models), figsize=(2.15 * len(models) + 1.2, 3.9), sharey=False, squeeze=False)
    h0 = pt[pt.model == models[0]].sort_values("human_t")
    order = list(h0.g)
    for j, m in enumerate(models):
        ax = axes[0, j]; style(ax, "x"); s = pt[pt.model == m].set_index("g").loc[order]
        yy = np.arange(len(order))
        for y_, (_, r) in zip(yy, s.iterrows()):
            ax.plot([r.human_t, r.model_t], [y_, y_], color=GRID, lw=2.2, zorder=1)
            ax.plot([r.model_t_lo, r.model_t_hi], [y_, y_], color=COL["D"], lw=1.0, zorder=2)
        ax.scatter(s.human_t, yy, s=22, color=INK, zorder=3, label="ANES (observed)")
        ax.scatter(s.model_t, yy, s=24, color=COL["D"], zorder=4, label="mirror (model)")
        ax.axvline(0, color=INK2, lw=0.6); ax.set_yticks(yy); ax.set_yticklabels([GLAB[g] for g in order] if j == 0 else [], fontsize=7.5); ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(NAMES[m], fontsize=9, loc="left"); ax.set_xlabel("Dem − Rep contrast (pts)", fontsize=7.8)
        if j == 0: ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); save(fig, out, "fig4_mirror_contrast")


def fig_mirror_strata(rd, out, models):
    st = pd.read_csv(rd / "a6_mirror_strata.csv")
    labs = [("non_moderate", "Non-moderates"), ("moderate", "Moderates"), ("dem_to_rep", "Dem → Rep"), ("rep_to_dem", "Rep → Dem"), ("excl_cross_pressured", "Excl. cross-pressured"), ("all", "All eligible")]
    fig, ax = plt.subplots(figsize=(6.0, 3.2)); style(ax, "x")
    n = len(models); w = 0.16
    for k, m in enumerate(models):
        for i, (s_, lab) in enumerate(labs):
            r = st[(st.model == m) & (st.stratum == s_)].iloc[0]
            y = len(labs) - 1 - i + (k - (n - 1) / 2) * w
            ax.plot([r.mean_aligned_gap_shift_lo, r.mean_aligned_gap_shift_hi], [y, y], color=COL["D"], lw=1.1, alpha=0.9)
            ax.scatter([r.mean_aligned_gap_shift], [y], s=20, color=COL["D"], marker=["o", "s", "^", "D"][k], edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([l for _, l in labs][::-1]); ax.set_xlabel("Shift in the synthetic Democratic − Republican gap, toward the mirror (points; max 200)")
    for k, m in enumerate(models):
        ax.scatter([], [], s=20, color=COL["D"], marker=["o", "s", "^", "D"][k], label=NAMES[m])
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(1.01, 1.0)); fig.tight_layout(); save(fig, out, "fig5_mirror_strata")


def fig_rq4(rd, out, models):
    f = rd / "conditional" / "conditional_fidelity_summary.csv"
    if not f.exists():
        print("   (fig6 skipped: conditional results missing)"); return
    k = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (col, lab) in zip(axes, (("pct_significant", "% of interaction coefficients differing from ANES"), ("pct_flip", "% of those with opposite sign"))):
        style(ax)
        for j, m in enumerate(models):
            for x, c in enumerate("ABC"):
                s = k[(k.model == m) & (k.condition == c)]
                if s.empty: continue
                r = s.iloc[0]; xx = j + (x - 1) * 0.22
                dot(ax, xx, r[col], r[col + "_lo"], r[col + "_hi"], COL[c], size=18)
        ax.set_xticks(range(len(models))); ax.set_xticklabels([NAMES[m].replace(" ", "\n", 1) for m in models], fontsize=7.5); ax.set_ylabel(lab, fontsize=8)
    for c in "ABC":
        axes[0].scatter([], [], s=18, color=COL[c], label=CLAB[c])
    axes[0].legend(frameon=False, fontsize=7.2, loc="lower left"); fig.tight_layout(); save(fig, out, "fig6_conditional_fidelity")


def main(rd: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in ORDER if m in set(pd.read_csv(rd / "a2_fidelity_table.csv").model)]
    print("figures:")
    fig_fidelity(rd, out, models); fig_noise(rd, out, models); fig_order(rd, out, models)
    fig_mirror(rd, out, models); fig_mirror_strata(rd, out, models); fig_rq4(rd, out, models)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "results",
         Path(sys.argv[2]) if len(sys.argv) > 2 else here.parent / "figures")
