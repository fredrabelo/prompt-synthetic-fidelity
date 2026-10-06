"""Generates, from a pipeline results directory, (1) `tables.md` with the paper's tables and (2) `facts.json` with scalar summaries
(ranges across models) of the same tables. No number is typed by hand: everything comes from the pipeline CSVs.

Usage: python3 build_tables.py [results_dir=../results]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = {"haiku": "Claude Haiku 4.5", "ministral": "Ministral 3B", "qwen": "Qwen3 32B", "gptoss": "GPT-OSS-120B", "MEAN_ACROSS_MODELS": "Mean across models"}
ORDER = ["haiku", "ministral", "qwen", "gptoss", "MEAN_ACROSS_MODELS"]
CONDN = {"A": "A  No information", "B": "B  Narrative persona", "C": "C  Structured fields", "D": "D  Partisan mirror"}


def f(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def fci(v, lo, hi, nd=2):
    return f"{f(v, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def pct(v, nd=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{nd}f}%"


def pci(v, lo, hi, nd=1):
    return f"{pct(v, nd)} [{pct(lo, nd)}, {pct(hi, nd)}]"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(out)


def mord(df):
    df = df.copy()
    df["_o"] = df.model.map({m: i for i, m in enumerate(ORDER)})
    return df.sort_values(["_o"] + [c for c in df.columns if c in ("cond", "contrast", "pair", "stratum")]).drop(columns="_o")


def main(rd: Path):
    R = lambda n: pd.read_csv(rd / n)
    facts, tables = {}, {}
    models = [m for m in ORDER[:-1] if m in set(R("a1_validity_by_model_cond.csv").model)]
    facts["models_list"] = ", ".join(NAMES[m] for m in models)
    facts["n_models"] = len(models)
    model_meta = {
        "haiku": ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "batch", "temperature 0.3; max tokens 1,500"),
        "ministral": ("mistral.ministral-3-3b-instruct", "synchronous", "temperature 0.3; max tokens 8,000"),
        "qwen": ("qwen.qwen3-32b-v1:0", "batch", "temperature 0.3; max tokens 8,000; thinking inactive"),
        "gptoss": ("openai.gpt-oss-120b-1:0", "batch", "reasoning effort low; temperature 0.3; max tokens 8,000"),
    }
    tables["models_execution"] = md(pd.DataFrame([
        {"Model": NAMES[m], "Identifier": model_meta[m][0], "Mode": model_meta[m][1], "Parameters": model_meta[m][2]}
        for m in models
    ]))

    # -- Validity -------------------------------------------------------
    v = R("a1_validity_by_model_cond.csv")
    w = v.pivot(index="model", columns="cond", values="valid_rate").reindex(models)
    t = pd.DataFrame({"Model": [NAMES[m] for m in models], **{f"{c}": [pct(w.loc[m, c]) for m in models] for c in "ABCD"}})
    tables["validity"] = md(t)
    facts["valid_min"] = pct(float(v.valid_rate.min()))
    facts["valid_min_model_cond"] = f"{NAMES[v.loc[v.valid_rate.idxmin(), 'model']]}, condition {v.loc[v.valid_rate.idxmin(), 'cond']}"
    facts["n_calls_total"] = f"{int(v.n.sum()):,}"

    # -- Fidelity (A, B, C; all respondents) ---------------------------
    ft = R("a2_fidelity_table.csv")
    ft = ft[ft.subset == "all"]
    rows = []
    for m in models:
        for c in "ABC":
            r = ft[(ft.model == m) & (ft.cond == c)].iloc[0]
            rows.append({"Model": NAMES[m], "Condition": CONDN[c], "MAE": fci(r.mae, r.mae_lo, r.mae_hi), "Correlation": fci(r["corr"], r.corr_lo, r.corr_hi, 3), "SD (% of human)": fci(r.sd_pct, r.sd_pct_lo, r.sd_pct_hi, 1)})
    tables["fidelity"] = md(pd.DataFrame(rows))
    dd = R("a2_fidelity_diffs.csv")
    rows = []
    for ctr, lab in (("A-B", "A − B"), ("A-C", "A − C"), ("B-C", "B − C")):
        for m in ORDER:
            s = dd[(dd.model == m) & (dd.contrast == ctr)]
            if s.empty or (m != "MEAN_ACROSS_MODELS" and m not in models):
                continue
            r = s.iloc[0]
            rows.append({"Contrast": lab, "Model": NAMES[m], "ΔMAE": fci(r.mae, r.mae_lo, r.mae_hi), "ΔCorrelation": fci(r["corr"], r.corr_lo, r.corr_hi, 3), "ΔSD (pp)": fci(r.sd_pct, r.sd_pct_lo, r.sd_pct_hi, 1)})
    tables["contrasts"] = md(pd.DataFrame(rows))
    g = lambda ctr, col: float(dd[(dd.model == "MEAN_ACROSS_MODELS") & (dd.contrast == ctr)][col].iloc[0])
    for ctr, key in (("A-B", "AB"), ("A-C", "AC"), ("B-C", "BC")):
        facts[f"{key}_mae_mean"] = f(g(ctr, "mae"))
        facts[f"{key}_mae_ci"] = f"[{f(g(ctr, 'mae_lo'))}, {f(g(ctr, 'mae_hi'))}]"
        facts[f"{key}_corr_mean"] = f(g(ctr, "corr"), 3)
        facts[f"{key}_corr_ci"] = f"[{f(g(ctr, 'corr_lo'), 3)}, {f(g(ctr, 'corr_hi'), 3)}]"
    facts["BC_mae_between_model_sd"] = f(g("B-C", "mae_between_model_sd"))
    perm = dd[(dd.contrast == "B-C") & (dd.model != "MEAN_ACROSS_MODELS")]
    facts["BC_mae_min"], facts["BC_mae_max"] = f(perm.mae.min()), f(perm.mae.max())
    A = ft[ft.cond == "A"]; B = ft[ft.cond == "B"]
    facts["A_corr_range"] = f"{f(A['corr'].min(), 3)} to {f(A['corr'].max(), 3)}"
    facts["B_corr_range"] = f"{f(B['corr'].min(), 3)}–{f(B['corr'].max(), 3)}"
    facts["A_mae_range"] = f"{f(A.mae.min())}–{f(A.mae.max())}"
    facts["B_mae_range"] = f"{f(B.mae.min())}–{f(B.mae.max())}"
    facts["A_sd_range"] = f"{f(A.sd_pct.min(), 1)}%–{f(A.sd_pct.max(), 1)}%"
    facts["B_sd_range"] = f"{f(B.sd_pct.min(), 1)}%–{f(B.sd_pct.max(), 1)}%"
    ccd = R("a2_fidelity_cc_diffs.csv")
    facts["BC_mae_cc_mean"] = f(float(ccd[(ccd.model == "MEAN_ACROSS_MODELS") & (ccd.contrast == "B-C")].mae.iloc[0]))

    # -- Within-respondent agreement -----------------------------------
    ag = R("a3_agreement_table.csv")
    rows = []
    for m in models:
        for pr in ("B-C", "A-B"):
            r = ag[(ag.model == m) & (ag.pair == pr)].iloc[0]
            rows.append({"Model": NAMES[m], "Pair": pr.replace("-", "–"), "Identical": pci(r.identical, r.identical_lo, r.identical_hi), "Within 5 pts": pct(r.within5), "Within 10 pts": pct(r.within10),
                         "Mean absolute difference": fci(r.mad, r.mad_lo, r.mad_hi), "Per-thermometer r": f(r.per_thermometer_pearson_mean, 3), "Within-respondent r (mean)": f(r.within_respondent_vector_pearson_mean, 3)})
    tables["agreement"] = md(pd.DataFrame(rows))
    bc = ag[ag.pair == "B-C"]
    facts["BC_identical_range"] = f"{pct(bc.identical.min())}–{pct(bc.identical.max())}"
    facts["BC_within5_range"] = f"{pct(bc.within5.min())}–{pct(bc.within5.max())}"
    facts["BC_mad_range"] = f"{f(bc.mad.min(), 1)}–{f(bc.mad.max(), 1)}"
    facts["BC_pertherm_r_range"] = f"{f(bc.per_thermometer_pearson_mean.min(), 2)}–{f(bc.per_thermometer_pearson_mean.max(), 2)}"
    ab = ag[ag.pair == "A-B"]
    facts["AB_identical_range"] = f"{pct(ab.identical.min())}–{pct(ab.identical.max())}"
    facts["AB_mad_range"] = f"{f(ab.mad.min(), 1)}–{f(ab.mad.max(), 1)}"

    # -- Reliability ---------------------------------------------------
    rl = R("a4_reliability_table.csv")
    se250 = R("a4_reliability_paired_draw_se.csv")
    se250_cell = se250.groupby(["model", "cond"], as_index=False).se_paired_mean.mean()
    facts["se250_range"] = f"{f(se250_cell.se_paired_mean.min())}–{f(se250_cell.se_paired_mean.max())}"
    gv = lambda m, unit, stat: rl[(rl.model == m) & (rl.unit == unit) & (rl.stat == stat)].iloc[0]
    rows = []
    for m in models:
        for c in "BC":
            rows.append({"Model": NAMES[m], "Condition": CONDN[c], "Identical across draws": pci(gv(m, c, "noise_identical").value, gv(m, c, "noise_identical").lo, gv(m, c, "noise_identical").hi),
                         "Mean absolute difference across draws": fci(gv(m, c, "noise_mad").value, gv(m, c, "noise_mad").lo, gv(m, c, "noise_mad").hi), "Entropy (bits)": f(gv(m, c, "entropy_bits").value),
                         "ICC(1)": f(gv(m, c, "icc1").value, 2)})
    tables["reliability"] = md(pd.DataFrame(rows))
    rows = []
    for m in models:
        for pr in ("B-C", "A-B"):
            x = gv(m, pr, "cross_mad"); n = gv(m, pr, "noise_mad_avg"); e = gv(m, pr, "excess_mad"); q = gv(m, pr, "ratio_cross_to_noise"); j = gv(m, pr, "jsd_2v2_excess")
            rows.append({"Model": NAMES[m], "Pair": pr.replace("-", "–"), "Between-condition mean absolute difference": f(x.value), "Same-prompt mean absolute difference (noise)": f(n.value),
                         "Excess over noise": fci(e.value, e.lo, e.hi), "Ratio": f(q.value), "Excess JSD (bits)": fci(j.value, j.lo, j.hi, 3)})
    tables["cross_vs_noise"] = md(pd.DataFrame(rows))
    bce = rl[(rl.unit == "B-C") & (rl.stat == "excess_mad")]
    facts["BC_excess_range"] = f"{f(bce.value.min())}–{f(bce.value.max())}"
    bcr = rl[(rl.unit == "B-C") & (rl.stat == "ratio_cross_to_noise")]
    facts["BC_ratio_range"] = f"{f(bcr.value.min())}–{f(bcr.value.max())}"
    abe = rl[(rl.unit == "A-B") & (rl.stat == "excess_mad")]
    facts["AB_excess_range"] = f"{f(abe.value.min(), 1)}–{f(abe.value.max(), 1)}"
    jsbc = rl[(rl.unit == "B-C") & (rl.stat == "jsd_2v2_excess")]
    jsab = rl[(rl.unit == "A-B") & (rl.stat == "jsd_2v2_excess")]
    facts["BC_jsd_range"] = f"{f(jsbc.value.min(), 3)}–{f(jsbc.value.max(), 3)}"
    facts["AB_jsd_range"] = f"{f(jsab.value.min(), 3)}–{f(jsab.value.max(), 3)}"
    nz = rl[(rl.family == "noise") & (rl.unit.isin(["B", "C"])) & (rl.stat == "noise_identical")]
    facts["noise_identical_range"] = f"{pct(nz.value.min())}–{pct(nz.value.max())}"
    nm = rl[(rl.family == "noise") & (rl.unit.isin(["B", "C"])) & (rl.stat == "noise_mad")]
    facts["noise_mad_range"] = f"{f(nm.value.min(), 1)}–{f(nm.value.max(), 1)}"
    ent = rl[(rl.family == "noise") & (rl.unit.isin(["B", "C"])) & (rl.stat == "entropy_bits")]
    facts["entropy_range"] = f"{f(ent.value.min(), 2)}–{f(ent.value.max(), 2)}"

    # -- Item order ----------------------------------------------------
    od = R("a5_order_table.csv")
    rows = []
    for m in models:
        for c in "ABCD":
            r = od[(od.model == m) & (od.cond == c)].iloc[0]
            rows.append({"Model": NAMES[m], "Condition": CONDN[c], "Mean absolute reversed–original difference": fci(r.mad, r.mad_lo, r.mad_hi, 2), "Same-prompt mean absolute difference": f(r.noise_mad_same_prompt), "Order/noise ratio": fci(r.order_to_noise_ratio, r.order_to_noise_ratio_lo, r.order_to_noise_ratio_hi),
                         "Signed mean (rev − orig)": f(r.signed_mean_rev_minus_fwd), "Position slope (diagnostic)": f(r.position_slope_diagnostic, 2)})
    tables["order"] = md(pd.DataFrame(rows))
    obc = od[od.cond.isin(["B", "C"])]
    facts["order_ratio_BC_range"] = f"{f(obc.order_to_noise_ratio.min())}–{f(obc.order_to_noise_ratio.max())}"
    facts["order_mad_BC_range"] = f"{f(obc.mad.min(), 1)}–{f(obc.mad.max(), 1)}"
    oa = od[od.cond == "A"]
    facts["order_mad_A_range"] = f"{f(oa.mad.min(), 1)}–{f(oa.mad.max(), 1)}"
    facts["order_ratio_A_range"] = f"{f(oa.order_to_noise_ratio.min())}–{f(oa.order_to_noise_ratio.max())}"

    # -- Partisan mirror -----------------------------------------------
    st = R("a6_mirror_strata.csv")
    lab = {"all": "All eligible (5,115)", "non_moderate": "Non-moderates (party and ideology change)", "moderate": "Moderates (party only)", "dem_to_rep": "Democrat → Republican", "rep_to_dem": "Republican → Democrat", "excl_cross_pressured": "Excluding cross-pressured profiles"}
    rows = []
    for m in models:
        for s_ in lab:
            r = st[(st.model == m) & (st.stratum == s_)].iloc[0]
            rows.append({"Model": NAMES[m], "Stratum": lab[s_], "n": f"{int(round(r.n_resp_weighted)):,}", "Aligned shift in Dem−Rep gap": fci(r.mean_aligned_gap_shift, r.mean_aligned_gap_shift_lo, r.mean_aligned_gap_shift_hi, 1),
                         "Moves in expected direction": pct(r.expected_dir_share), "Mean absolute spillover (other 9 groups)": f(r.spill_mean_abs, 1)})
    tables["mirror"] = md(pd.DataFrame(rows))
    ct = R("a6_mirror_contrast.csv")
    rows = [{"Model": NAMES[r.model], "Exaggeration, 11 groups": fci(r.exaggeration_all, r.exaggeration_all_lo, r.exaggeration_all_hi), "Exaggeration, 2 party groups": fci(r.exaggeration_party, r.exaggeration_party_lo, r.exaggeration_party_hi),
             "Pattern correlation with human contrast": f(r.pattern_corr, 3)} for r in ct.itertuples() if r.model in models]
    tables["exaggeration"] = md(pd.DataFrame(rows))
    a = st[st.stratum == "all"]
    facts["mirror_all_range"] = f"{f(a.mean_aligned_gap_shift.min(), 0)}–{f(a.mean_aligned_gap_shift.max(), 0)}"
    facts["mirror_expected_range"] = f"{pct(a.expected_dir_share.min())}–{pct(a.expected_dir_share.max())}"
    nm_ = st[st.stratum == "non_moderate"]; mo = st[st.stratum == "moderate"]
    facts["mirror_nonmod_range"] = f"{f(nm_.mean_aligned_gap_shift.min(), 0)}–{f(nm_.mean_aligned_gap_shift.max(), 0)}"
    facts["mirror_mod_range"] = f"{f(mo.mean_aligned_gap_shift.min(), 0)}–{f(mo.mean_aligned_gap_shift.max(), 0)}"
    facts["exag_all_range"] = f"{f(ct.exaggeration_all.min())}–{f(ct.exaggeration_all.max())}"
    facts["exag_party_range"] = f"{f(ct.exaggeration_party.min())}–{f(ct.exaggeration_party.max())}"
    facts["pattern_corr_min"] = f(ct.pattern_corr.min(), 2)
    best, worst = ct.loc[ct.exaggeration_all.idxmin()], ct.loc[ct.exaggeration_all.idxmax()]
    facts["exag_lowest_model"] = f"{NAMES[best.model]} ({f(best.exaggeration_all)})"
    facts["exag_highest_model"] = f"{NAMES[worst.model]} ({f(worst.exaggeration_all)})"
    cr = st[st.stratum == "excl_cross_pressured"]
    facts["mirror_cross_excl_range"] = f"{f(cr.mean_aligned_gap_shift.min(), 0)}–{f(cr.mean_aligned_gap_shift.max(), 0)}"

    # -- Conditional fidelity ------------------------------------------
    cf = rd / "conditional" / "conditional_fidelity_summary.csv"
    if cf.exists():
        k = pd.read_csv(cf)
        rows = []
        for m in models:
            for c in "ABC":
                s_ = k[(k.model == m) & (k.condition == c)]
                if s_.empty:
                    continue
                r = s_.iloc[0]
                rows.append({"Model": NAMES[m], "Condition": CONDN[c], "% coefficients differing from ANES": f"{f(r.pct_significant, 1)} [{f(r.pct_significant_lo, 1)}, {f(r.pct_significant_hi, 1)}]",
                             "% of those with opposite sign": f"{f(r.pct_flip, 1)} [{f(r.pct_flip_lo, 1)}, {f(r.pct_flip_hi, 1)}]"})
        tables["rq4"] = md(pd.DataFrame(rows))
        ka = k[k.condition == "A"]; kb = k[k.condition.isin(["B", "C"])]
        facts["rq4_A_range"] = f"{f(ka.pct_significant.min(), 0)}%–{f(ka.pct_significant.max(), 0)}%"
        facts["rq4_BC_range"] = f"{f(kb.pct_significant.min(), 0)}%–{f(kb.pct_significant.max(), 0)}%"
        facts["rq4_flip_A_range"] = f"{f(ka.pct_flip.min(), 0)}%–{f(ka.pct_flip.max(), 0)}%"
        facts["rq4_flip_BC_range"] = f"{f(kb.pct_flip.min(), 0)}%–{f(kb.pct_flip.max(), 0)}%"
    (rd / "tables.md").write_text("\n\n".join(f"<!-- TABLE:{k} -->\n{v}" for k, v in tables.items()) + "\n")
    (rd / "facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False))
    print(f"{len(tables)} tables, {len(facts)} facts -> {rd}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results")
