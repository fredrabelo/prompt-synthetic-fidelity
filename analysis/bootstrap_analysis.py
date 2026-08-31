"""RQ1-RQ3 point estimates and respondent-level bootstrap confidence intervals.

Reproduces every number in Table 1, Table 2, and the Appendix's "Detailed
bootstrap estimates" table of the paper. Reads only the public CSVs in
../data/ - no dependency on any internal system.

Usage: python3 bootstrap_analysis.py
Output: bootstrap_summary.json (point estimates, 95% CIs, and pairwise
        B-C / B-D / C-D difference CIs for MAE, correlation, and SD)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = Path(__file__).resolve().parent / "bootstrap_summary.json"

CONDITIONS = [
    "A_generic_llm",
    "B_persona_demographic",
    "C_population_grounded",
    "D_evidence_grounded",
]
N_BOOTSTRAP = 500
SEED = 42


def load_paired_data():
    """One row per (respondent, thermometer, condition) with both a human and
    a synthetic value - the unit every metric below operates on."""
    human = pd.read_csv(DATA_DIR / "human_benchmark.csv")
    synth = pd.read_csv(DATA_DIR / "synthetic_responses.csv")
    synth["synthetic_value"] = pd.to_numeric(synth["synthetic_value"], errors="coerce")

    paired = {}
    for condition in CONDITIONS:
        rows = synth[synth.condition == condition][["external_key", "thermometer", "synthetic_value"]]
        merged = human.merge(rows, on=["external_key", "thermometer"], how="inner")
        paired[condition] = merged.dropna(subset=["synthetic_value"])
    return human, paired


def compute_metrics(df, thermometers):
    """MAE: per-thermometer mean absolute error, then averaged across
    thermometers (equal-weighted by question). SD: pooled across all matched
    responses. Correlation: per-thermometer Pearson, averaged across
    thermometers. This exactly matches the paper's Metrics section."""
    per_question_mae = df.assign(err=(df.human_value - df.synthetic_value).abs()).groupby("thermometer")["err"].mean()
    mae = per_question_mae.mean()
    sd = df.synthetic_value.std()
    correlations = []
    for thermometer in thermometers:
        subset = df[df.thermometer == thermometer]
        if len(subset) >= 3 and subset.human_value.std() > 0 and subset.synthetic_value.std() > 0:
            correlations.append(subset.human_value.corr(subset.synthetic_value))
    corr = float(np.mean(correlations)) if correlations else float("nan")
    return mae, corr, sd


def main():
    human, paired = load_paired_data()
    thermometers = human.thermometer.unique()
    human_sd = human.human_value.std()

    point = {}
    for condition in CONDITIONS:
        mae, corr, sd = compute_metrics(paired[condition], thermometers)
        point[condition] = {"mae": mae, "corr": corr, "sd": sd, "sd_pct": 100 * sd / human_sd}

    rng = np.random.default_rng(SEED)
    respondent_keys = human.external_key.unique()
    n_respondents = len(respondent_keys)

    boot = {c: {"mae": [], "corr": [], "sd": []} for c in CONDITIONS}
    for _ in range(N_BOOTSTRAP):
        sample_keys = rng.choice(respondent_keys, size=n_respondents, replace=True)
        sample_frame = pd.DataFrame({"external_key": sample_keys})
        for condition in CONDITIONS:
            merged = sample_frame.merge(paired[condition], on="external_key", how="left").dropna(
                subset=["human_value", "synthetic_value"]
            )
            mae, corr, sd = compute_metrics(merged, thermometers)
            boot[condition]["mae"].append(mae)
            boot[condition]["corr"].append(corr)
            boot[condition]["sd"].append(sd)

    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {"point": point[condition]}
        for metric in ("mae", "corr", "sd"):
            values = np.array(boot[condition][metric])
            summary[condition][f"{metric}_ci"] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]

    pairs = [
        ("B_persona_demographic", "C_population_grounded"),
        ("B_persona_demographic", "D_evidence_grounded"),
        ("C_population_grounded", "D_evidence_grounded"),
    ]
    diffs = {}
    for a, b in pairs:
        label = f"{a.split('_')[0]}-{b.split('_')[0]}"
        diffs[label] = {}
        for metric in ("mae", "corr", "sd"):
            delta = np.array(boot[a][metric]) - np.array(boot[b][metric])
            lo, hi = float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))
            diffs[label][metric] = {
                "point_diff": point[a][metric] - point[b][metric],
                "ci": [lo, hi],
                "significant": not (lo < 0 < hi),
            }

    with OUT_PATH.open("w") as f:
        json.dump({"human_sd": human_sd, "point": point, "summary": summary, "diffs": diffs}, f, indent=2)

    print(f"Wrote {OUT_PATH}")
    for condition in CONDITIONS:
        p = point[condition]
        print(f"  {condition}: MAE={p['mae']:.2f} corr={p['corr']:.3f} SD={p['sd']:.2f} ({p['sd_pct']:.1f}% of human)")


if __name__ == "__main__":
    main()
