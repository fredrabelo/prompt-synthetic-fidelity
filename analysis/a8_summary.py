"""Builds, from the analysis tables, (1) `summary.json` with the headline numbers of every analysis and (2) `aapor_table.csv`
(criterion x metric x model). No number is typed by hand: everything comes from the tables the pipeline generates."""
from __future__ import annotations

import json

import pandas as pd


def _rec(df: pd.DataFrame | None):
    return [] if df is None else json.loads(df.to_json(orient="records"))


def build(res: dict) -> dict:
    s = {}
    fid = res.get("fidelity")
    if fid:
        d = fid["diffs"]
        s["fidelity_B_minus_C"] = _rec(d[(d.contrast == "B-C")])
        s["information_value_A_minus_B"] = _rec(d[(d.contrast == "A-B")])
        s["information_value_A_minus_C"] = _rec(d[(d.contrast == "A-C")])
        s["fidelity_by_model_condition"] = _rec(fid["table"])
        s["sensitivity_complete_case_diffs"] = _rec(fid["cc_diffs"])
        s["sensitivity_complete_case_table"] = _rec(fid["cc_table"])
        s["n_complete_case_respondents"] = fid["n_complete_case"]
        s["n"] = {"all": fid["n_all"], "eligible": fid["n_elig"], "human_sd": fid["human_sd"]}
    ag = res.get("agreement")
    if ag:
        s["within_respondent_agreement"] = _rec(ag["table"])
    rel = res.get("reliability")
    if rel:
        t = rel["table"]
        s["reliability_noise"] = _rec(t[t.family == "noise"])
        s["reliability_cross_vs_noise"] = _rec(t[t.family == "cross"])
        s["reliability_divergence"] = _rec(t[t.family == "div"])
    od = res.get("order")
    if od:
        s["order"] = _rec(od["table"])
        s["order_per_item"] = _rec(od["per_item"])
    mr = res.get("mirror")
    if mr:
        s["mirror_strata"] = _rec(mr["strata"])
        s["mirror_contrast"] = _rec(mr["contrast"])
        s["mirror_per_thermometer"] = _rec(mr["per_thermometer"])
    cf = res.get("conditional")
    if cf:
        s["rq4_conditional_fidelity_summary"] = _rec(cf["summary"])
    va = res.get("validity")
    if va:
        s["validity_by_model_condition"] = _rec(va["by_model_cond"])
        s["validity_invalid_reasons"] = _rec(va["invalid_reasons"])
    return s


def aapor_table(res: dict) -> pd.DataFrame:
    rows = []
    fid = res.get("fidelity")
    if fid:
        for r in fid["table"][fid["table"].cond != "D"].itertuples():   # D (mirror) is not a "fidelity" condition
            rows.append(("Validity", f"MAE vs ANES ({r.cond}, {r.subset})", r.model, r.mae, r.mae_lo, r.mae_hi))
            rows.append(("Validity", f"Correlation with ANES ({r.cond}, {r.subset})", r.model, r.corr, r.corr_lo, r.corr_hi))
            rows.append(("Performance", f"SD as % of human ({r.cond}, {r.subset})", r.model, r.sd_pct, r.sd_pct_lo, r.sd_pct_hi))
    va = res.get("validity")
    if va:
        for r in va["by_model_cond"].itertuples():
            rows.append(("Performance", f"Valid-output rate ({r.cond})", r.model, r.valid_rate, None, None))
    ag = res.get("agreement")
    if ag:
        for r in ag["table"].itertuples():
            rows.append(("Sensitivity", f"Within-respondent identical share ({r.pair})", r.model, r.identical, r.identical_lo, r.identical_hi))
            rows.append(("Sensitivity", f"Within-respondent mean abs diff ({r.pair})", r.model, r.mad, r.mad_lo, r.mad_hi))
    od = res.get("order")
    if od:
        for r in od["table"].itertuples():
            rows.append(("Sensitivity", f"Reversed-vs-original MAD ({r.cond})", r.model, r.mad, r.mad_lo, r.mad_hi))
    cf = res.get("conditional")
    if cf:
        for r in cf["summary"].itertuples():
            rows.append(("Validity", f"% interaction coefficients differing from ANES ({r.condition})", r.model, r.pct_significant, r.pct_significant_lo, r.pct_significant_hi))
            rows.append(("Validity", f"% of those with sign flip ({r.condition})", r.model, r.pct_flip, r.pct_flip_lo, r.pct_flip_hi))
    mr = res.get("mirror")
    if mr:
        for r in mr["strata"].itertuples():
            rows.append(("Sensitivity", f"Mirror aligned gap shift ({r.stratum})", r.model, r.mean_aligned_gap_shift, r.mean_aligned_gap_shift_lo, r.mean_aligned_gap_shift_hi))
        for r in mr["contrast"].itertuples():
            rows.append(("Sensitivity", "Mirror exaggeration vs human contrast (11 groups)", r.model, r.exaggeration_all, r.exaggeration_all_lo, r.exaggeration_all_hi))
    rel = res.get("reliability")
    if rel:
        t = rel["table"]
        for r in t[(t.family == "noise") & t.stat.isin(["noise_identical", "noise_mad", "entropy_bits", "icc1"])].itertuples():
            rows.append(("Reliability", f"{r.stat} ({r.unit})", r.model, r.value, r.lo, r.hi))
    return pd.DataFrame(rows, columns=["criterion", "metric", "model", "value", "lo", "hi"])
