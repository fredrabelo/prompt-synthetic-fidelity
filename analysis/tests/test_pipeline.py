"""Pipeline tests on SYNTHETIC data with known ground truth (no API calls, no real model data).

Run: python3 analysis/tests/test_pipeline.py   (about a minute, plus the R step)

The test builds a temporary data directory in the release schema (respondents.csv, human_benchmark.csv,
subsample_250.json, calls.csv) and checks every analysis against an independent computation or a theoretical value.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REAL_DATA = HERE.parents[1] / "data"
tmp = Path(tempfile.mkdtemp(prefix="p1test_"))
DATA = tmp / "data"
DATA.mkdir()
for f in ("respondents.csv", "human_benchmark.csv", "subsample_250.json"):
    shutil.copy(REAL_DATA / f, DATA / f)
os.environ["P1_DATA_DIR"] = str(DATA)
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
from common import THERMS, DataError  # noqa: E402
import a1_validity, a2_fidelity, a3_agreement, a4_reliability, a5_order, a6_mirror  # noqa: E402

rng = np.random.default_rng(7)
hw, attrs = common.load_human_and_attrs()
keys = list(hw.index)
elig = set(common.eligible_keys(attrs))
SUB = common.sub250_keys()
POSF = {t: i + 1 for i, t in enumerate(THERMS)}
SLOPE = 0.5          # built-in position effect: value += SLOPE * (pos_rev - pos_fwd) under reversed order
MIRROR = 40          # size of the mirror shift on the party thermometers
INVALID_RATE = 0.02
CALL_COLS = ["model", "condition", "external_key", "block", "draw", "order", "serving_mode", "valid", "invalid_reason",
             "stop_reason", "input_tokens", "output_tokens", "reasoning_chars", "prompt_sha256"]


def base_value(k, t, cond):
    """Noise-free value implied by the prompt for respondent k and thermometer t."""
    h = hw.at[k, t]
    h = 50.0 if np.isnan(h) else h
    pid = attrs.at[k, "pid"]
    if cond == "A":
        return 50.0
    v = 0.5 * h + 25
    if t in ("thermometer_republican_party", "thermometer_democratic_party") and pid in ("Democrat", "Republican"):
        dem = pid == "Democrat"
        isdem = t == "thermometer_democratic_party"
        v = 70.0 if (dem == isdem) else 30.0
        if cond == "D":
            v = 100.0 - v          # mirror swaps 70 <-> 30
    return v


def make_call(k, cond, draw, order, rng_, model_noise):
    vals = {}
    for t in THERMS:
        v = base_value(k, t, cond) + rng_.normal(0, model_noise)
        if order == "R":
            v += SLOPE * ((len(THERMS) + 1 - POSF[t]) - POSF[t])
        vals[t] = int(np.clip(np.rint(v), 0, 100))
    return vals


def build_model(model, noise):
    """All calls of one model under the final design, as a DataFrame in the release schema."""
    blocks = {"full": [(k, c, 0, "F") for k in keys for c in "ABC"] + [(k, "D", 0, "F") for k in keys if k in elig],
              "cons": [(k, c, dr, "F") for k in SUB for c in "ABCD" for dr in (1, 2, 3, 4)],
              "rev": [(k, c, 0, "R") for k in SUB for c in "ABCD"]}
    rows = []
    for blk, units in blocks.items():
        for (k, c, dr, o) in units:
            valid = rng.random() > INVALID_RATE
            row = {"model": model, "condition": c, "external_key": k, "block": blk, "draw": dr, "order": o,
                   "serving_mode": "batch", "valid": bool(valid), "invalid_reason": None if valid else "synthetic_invalid",
                   "stop_reason": "stop", "input_tokens": 1000, "output_tokens": 500, "reasoning_chars": 0,
                   "prompt_sha256": "0" * 64}
            vals = make_call(k, c, dr, o, rng, noise) if valid else {}
            for t in THERMS:
                row[t] = vals.get(t)
            rows.append(row)
    return pd.DataFrame(rows)


def write_calls(df):
    out = df.copy()
    for t in THERMS + ["input_tokens", "output_tokens", "reasoning_chars"]:
        col = pd.to_numeric(out[t])
        out[t] = col.astype("Int64") if (col.dropna() % 1 == 0).all() else col
    out.to_csv(DATA / "calls.csv", index=False)


def refuses(df, why, models=("haiku",)):
    write_calls(df)
    try:
        common.load_responses(models=list(models))
    except DataError as e:
        ok(f"loader: {why} rejected ({str(e)[:70]})")
        return
    raise SystemExit(f"loader should have failed: {why}")


def ok(name):
    print("OK  ", name)


# -- 1. loader accepts the full design and rejects incomplete or inconsistent data ------
haiku = build_model("haiku", noise=8.0)
qwen = build_model("qwen", noise=8.0)
full = pd.concat([haiku, qwen], ignore_index=True)
write_calls(full)
L = common.load_responses(models=["haiku", "qwen"])
assert L.calls.groupby("model").size().to_dict() == {"haiku": 27693 + 4000 + 1000, "qwen": 27693 + 4000 + 1000}
ok("loader: unit counts per block are correct")

refuses(haiku, "model without calls", models=("ministral",))
refuses(haiku[haiku.block != "rev"], "missing block")
refuses(pd.concat([haiku, haiku.iloc[[0]]], ignore_index=True), "duplicate units")

bad = haiku.copy()
i = bad.index[bad.block == "rev"][0]
bad.loc[i, "draw"] = 5      # same number of units, wrong composition
refuses(bad, "wrong composition with the right count")

bad = haiku.copy()
i = bad.index[bad.valid][0]
bad.loc[i, THERMS[0]] = 150
refuses(bad, "out-of-range value")

bad = haiku.copy()
i = bad.index[bad.valid][0]
bad.loc[i, THERMS[0]] = 50.5
refuses(bad, "non-integer value")

bad = haiku.copy()
i = bad.index[bad.valid][0]
bad.loc[i, THERMS[3]] = np.nan
refuses(bad, "valid call with fewer than eleven responses")

bad = haiku.copy()
i = bad.index[bad.block == "cons"][0]
bad.loc[i, "serving_mode"] = "sync"
refuses(bad, "mixed serving modes within a block")

write_calls(full)
resp = L.resp
M = ["haiku", "qwen"]

# -- 2. validity --───────────────────────────────────────────────────────
v = a1_validity.run(L.calls, resp)["by_model_cond"]
assert abs(v.valid_rate.mean() - (1 - INVALID_RATE)) < 0.01
ok(f"validity ≈ {1 - INVALID_RATE:.2f} (obs {v.valid_rate.mean():.3f})")

# -- 3. fidelity: unit weights reproduce an independent direct computation --
F = a2_fidelity.run(resp, b=40, models=M)
t = F["table"]
for m in M:
    for c in "BC":
        w = common.wide(resp, m, c, 0, "F", hw.index, block="full")
        mae_direct = np.mean([np.nanmean(np.abs((hw[tt] - w[tt]).values)) for tt in THERMS])
        got = float(t[(t.model == m) & (t.cond == c) & (t.subset == "all")].mae.iloc[0])
        assert abs(mae_direct - got) < 1e-4, (m, c, mae_direct, got)
ok("fidelity: MAE == direct computation (per thermometer, mean of 11)")
a_corr = t[(t.cond == "A") & (t.subset == "all")]["corr"].iloc[0]
assert np.isnan(a_corr) or abs(a_corr) < 0.1, a_corr
b_corr = t[(t.cond == "B") & (t.subset == "all")]["corr"].iloc[0]
assert b_corr > 0.5, b_corr
ok(f"fidelity: A uncorrelated ({a_corr:.3f}), B correlated ({b_corr:.3f})")
d = F["diffs"]
bc = d[(d.contrast == "B-C") & (d.model == "haiku")].iloc[0]
assert bc.mae_lo <= 0 <= bc.mae_hi, "B and C have the same distribution, so the CI of the difference must cover 0"
ok("fidelity: B-C (same distribution) has a CI covering 0")
assert not (t.cond == "D").any() and not (d.contrast.str.contains("D")).any()
ok("fidelity: D (mirror) absent from a2 (table and contrasts)")

# paired B-C contrast == MAE_B(intersection) - MAE_C(intersection), computed directly
wb = common.wide(resp, "haiku", "B", 0, "F", hw.index, block="full"); wc = common.wide(resp, "haiku", "C", 0, "F", hw.index, block="full")
joint = wb.notna() & wc.notna()
def mae_on(w, mask):
    return np.mean([np.nanmean(np.abs((hw[tt] - w[tt].where(mask[tt])).values)) for tt in THERMS])
direct_bc = mae_on(wb, joint) - mae_on(wc, joint)
got_bc = float(d[(d.model == "haiku") & (d.contrast == "B-C") & (d.subset == "all")].mae.iloc[0])
assert abs(direct_bc - got_bc) < 1e-4, (direct_bc, got_bc)
ok(f"fidelity: PAIRED B-C contrast == direct computation on the intersection of valid cells ({got_bc:.4f})")
assert len(F["cc_diffs"]) > 0 and F["n_complete_case"]["haiku"] < 7526
ok(f"fidelity: complete-case sensitivity present ({F['n_complete_case']['haiku']} respondents with complete A/B/C)")

# -- 4. within-respondent agreement --
G = a3_agreement.run(resp, b=40, models=M)["table"]
bcg = G[(G.model == "haiku") & (G.pair == "B-C")].iloc[0]
w1 = common.wide(resp, "haiku", "B", 0, "F", hw.index, block="full"); w2 = common.wide(resp, "haiku", "C", 0, "F", hw.index, block="full")
mm = w1.notna() & w2.notna()
ident_direct = float(((w1 == w2) & mm).values.sum() / mm.values.sum())
assert abs(ident_direct - bcg.identical) < 1e-6
assert {"pooled_cell_pearson", "per_thermometer_pearson_mean", "within_respondent_vector_pearson_mean"} <= set(G.columns)
assert bcg.within_respondent_vector_pearson_mean < bcg.per_thermometer_pearson_mean + 0.5 and -1 <= bcg.within_respondent_vector_pearson_mean <= 1
ok("agreement: per-thermometer and within-respondent (11-vector) correlations present and distinct from the pooled one")
ok(f"agreement B-C: identical == direct computation ({ident_direct:.4f})")
ab = G[(G.model == "haiku") & (G.pair == "A-B")].iloc[0]
assert ab.identical < bcg.identical and ab.mad > bcg.mad
ok("agreement: A-B worse than B-C (A has no information)")

# -- 5. reliability: known noise --
R = a4_reliability.run(resp, b=40, models=["haiku"])["table"]
nz = R[(R.model == "haiku") & (R.unit == "B") & (R.stat == "noise_mad")].value.iloc[0]
# two independent N(0, sigma=8) draws: E|X-Y| = 2*sigma/sqrt(pi) = 9.03 (less rounding/clipping)
assert abs(nz - 2 * 8 / np.sqrt(np.pi)) < 0.7, nz
ok(f"reliability: noise MAD ≈ theoretical ({nz:.2f} vs {2 * 8 / np.sqrt(np.pi):.2f})")
ex = R[(R.model == "haiku") & (R.unit == "B-C") & (R.stat == "excess_mad")].iloc[0]
assert ex.lo <= 0.5 and ex.hi >= -0.5, (ex.lo, ex.hi)
ok(f"reliability: B-C excess over noise ≈ 0 ({ex.value:.2f})")
exa = R[(R.model == "haiku") & (R.unit == "A-B") & (R.stat == "excess_mad")].iloc[0]
assert exa.value > 3, exa.value
ok(f"reliability: A-B excess is large ({exa.value:.1f}); information changes responses beyond noise")
jx = R[(R.model == "haiku") & (R.unit == "B-C") & (R.stat == "jsd_2v2_excess")].iloc[0]
assert jx.lo <= 0.003 and abs(jx.value) < 0.005, (jx.value, jx.lo, jx.hi)
jb = R[(R.model == "haiku") & (R.unit == "B-C") & (R.stat == "jsd_2v2_samePrompt_baseline")].value.iloc[0]
jc = R[(R.model == "haiku") & (R.unit == "B-C") & (R.stat == "jsd_2v2_cross")].value.iloc[0]
assert abs(jc - jb) < 0.004, (jc, jb)
jab = R[(R.model == "haiku") & (R.unit == "A-B") & (R.stat == "jsd_2v2_excess")].value.iloc[0]
assert jab > 0.05, jab
ok(f"JSD 2x2: B-C (same distribution) excess ≈ 0 ({jx.value:.4f}); A-B excess large ({jab:.3f}); baseline and cross use equally sized histograms")
n25 = R[(R.model == "haiku") & (R.unit == "B-C") & (R.stat == "cross_mad")].value.iloc[0]
assert abs(n25 - 2 * 8 / np.sqrt(np.pi)) < 0.8
ok(f"reliability: cross_mad over 25 pairs ≈ theoretical ({n25:.2f})")
icc = R[(R.model == "haiku") & (R.unit == "B") & (R.stat == "icc1")].value.iloc[0]
assert 0.2 < icc < 0.95, icc
ok(f"reliability: plausible ICC(1) ({icc:.2f})")

# -- 6. order: built-in slope --
O = a5_order.run(resp, b=40, models=["haiku"])["table"]
sl = O[O.cond == "B"].position_slope_diagnostic.iloc[0]
F0 = common.wide(resp, "haiku", "B", 0, "F", SUB, block="full").values; R0 = common.wide(resp, "haiku", "B", 0, "R", SUB, block="rev").values
dd = (R0 - F0); mk = ~np.isnan(dd); dxm = np.tile((12 - 2 * np.arange(1, 12)).astype(float), (len(SUB), 1))
sl_direct = np.polyfit(dxm[mk], dd[mk], 1)[0]
assert abs(sl_direct - sl) < 1e-6, (sl_direct, sl)       # implementation == direct regression
se = 11.3 / np.sqrt(mk.sum() * 40)                       # theoretical SE of the slope (sd_diff ≈ 8*sqrt(2), var(dx) = 40)
assert abs(sl - SLOPE) < 4 * se, (sl, SLOPE, se)         # recovers the built-in effect within 4 SE
ok(f"order: position slope recovered ({sl:.3f} vs {SLOPE})")
rt = O[O.cond == "B"].order_to_noise_ratio.iloc[0]
assert rt > 1, rt
ok(f"order: order/noise ratio > 1 when there is a position effect ({rt:.2f})")

# -- 7. mirror: +-40 on the party thermometers --
Mi = a6_mirror.run(resp, b=40, models=["haiku"])
S = Mi["strata"]
dr = S[S.stratum == "dem_to_rep"].iloc[0]; rd = S[S.stratum == "rep_to_dem"].iloc[0]
# Dem->Rep: T_rep rises by ~40, T_dem falls by ~40, g falls by ~80 (70->30 and 30->70)
assert abs(dr.mean_delta_rep_party - 40) < 2.0 and abs(dr.mean_delta_dem_party + 40) < 2.0, (dr.mean_delta_rep_party, dr.mean_delta_dem_party)
assert abs(dr.mean_gap_shift + 80) < 3.0 and dr.expected_dir_share > 0.99
assert abs(rd.mean_gap_shift - 80) < 3.0 and rd.expected_dir_share > 0.99
al = S[S.stratum == "all"].iloc[0]
assert abs(al.mean_aligned_gap_shift - 80) < 3.0 and abs(al.mean_gap_shift) < 5, (al.mean_aligned_gap_shift, al.mean_gap_shift)
ok(f"mirror: mixed stratum; signed gap shift cancels ({al.mean_gap_shift:.1f}) but the aligned shift is +80 ({al.mean_aligned_gap_shift:.1f})")
ok(f"mirror: party shifts ±40 and gap ±80 in the expected direction (dem->rep g={dr.mean_gap_shift:.1f}, rep->dem g={rd.mean_gap_shift:.1f})")
assert abs(S[S.stratum == "all"].spill_mean.iloc[0]) < 0.5
ok("mirror: no spillover on the other 9 groups (synthetic data have none)")
n_all = S[S.stratum == "all"].n_resp_weighted.iloc[0]
assert abs(n_all - 5115) < 1
ok("mirror: 5,115 eligible respondents")
# exaggeration with structurally missing human values: compare with a direct pandas computation (NaN-skipping means)
ek_ = common.eligible_keys(attrs); ea_ = attrs.loc[ek_]
isdem_ = (ea_.pid == "Democrat").values
Bw = common.wide(resp, "haiku", "B", 0, "F", ek_, block="full"); Dw = common.wide(resp, "haiku", "D", 0, "F", ek_, block="full")
Dl_ = (Dw - Bw)
H_ = hw.loc[ek_]
model_t = (Dl_[~isdem_].mean() - Dl_[isdem_].mean()) / 2          # pandas skips NaN
human_t = H_[isdem_].mean() - H_[~isdem_].mean()
exag_direct = float(model_t.abs().mean() / human_t.abs().mean())
Cc = Mi["contrast"].iloc[0]
assert H_.isna().sum().sum() > 5000, "the test needs the missing values of the human benchmark"
assert abs(Cc.exaggeration_all - exag_direct) < 1e-3, (Cc.exaggeration_all, exag_direct)
bug = float(((Dl_[~isdem_].mean() - Dl_[isdem_].mean()) / 2).abs().mean() / (H_.fillna(0)[isdem_].mean() - H_.fillna(0)[~isdem_].mean()).abs().mean())
assert abs(bug - exag_direct) > 1e-3, "a NaN->0 computation should differ (ensures the test would detect that bug)"
ok(f"mirror: exaggeration with missing human values == direct computation ({exag_direct:.4f}); NaN->0 would give {bug:.4f} (detectable)")

# integration: run_all end to end, including R (a7) and the conditional-fidelity summary / AAPOR table
outd = tmp / "out_run_all"
r = subprocess.run([sys.executable, str(HERE.parent / "run_all.py"), "--models", "haiku", "--b", "20", "--b-cond", "2", "--out", str(outd)], capture_output=True, text=True, env=dict(os.environ))
assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
sm = json.loads((outd / "summary.json").read_text())
assert sm.get("rq4_conditional_fidelity_summary"), "conditional-fidelity summary missing from summary.json"
at = pd.read_csv(outd / "aapor_table.csv")
assert at.metric.str.contains("interaction coefficients").any(), "conditional-fidelity row missing from the AAPOR table"
man = json.loads((outd / "results_manifest.json").read_text())
assert any("respondents.csv" in k for k in man["input_file_sha256"]) and any("subsample_250" in k for k in man["input_file_sha256"]) and any("calls.csv" in k for k in man["input_file_sha256"])
ok("run_all: end to end with R; conditional fidelity in summary.json and aapor_table; manifest lists respondents, benchmark, subsample and calls")

print("\nALL TESTS PASSED")
shutil.rmtree(tmp)
