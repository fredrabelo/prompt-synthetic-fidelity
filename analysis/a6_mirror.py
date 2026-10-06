"""Partisan mirror (condition D vs B): sensitivity and stereotyping (AAPOR: sensitivity).

Unit: the 5,115 eligible partisans; for each, delta = D - B (same respondent, same model, draw 0, original order).
D mirrors party identification and ideology (Democrat <-> Republican, intensity preserved, moderate stays moderate);
everything else is identical to B.

NOTE ON LANGUAGE: the mirror is a counterfactual paired WITHIN THE MODEL. The ANES Democrat-minus-Republican contrast is
OBSERVATIONAL (not the causal effect of switching party for the same person). Magnitude and direction are compared, not a
"human causal effect".

Pre-specified strata: all eligible; non-moderates (party and ideology change); moderates (only party changes); each direction
(Democrat->Republican, Republican->Democrat); sensitivity excluding cross-pressured profiles (Democrat with a conservative
ideology / Republican with a liberal one).

Statistics per stratum
  mean_delta_rep_party / mean_delta_dem_party : mean delta on the party thermometers
  mean_gap_shift  : mean delta of g = (T_Democrat - T_Republican). SIGNED: only interpretable within one direction; in strata that
                    mix both directions (all, moderate, non_moderate) the directions cancel, so use the ALIGNED versions below
  mean_aligned_gap_shift / mean_aligned_delta_{rep,dem}_party : same quantities oriented so that positive = direction of the mirror
  expected_dir_share : share of respondents whose g moves in the expected direction (Dem->Rep: g falls; Rep->Dem: g rises); ties reported separately
  mean_abs_gap_shift : mean |delta g| (magnitude)
  spill_mean / spill_mean_abs : mean delta (signed and absolute) on the other 9 groups (spillover)
Relative exaggeration (all eligible, N=5,115):
  human_t   = human mean(t | Democrat) - human mean(t | Republican)                 [observational]
  model_t   = ( E_Rep[delta_t] - E_Dem[delta_t] ) / 2 = average paired effect of switching Rep->Dem in the model
  b_t       = B mean(t | Democrat) - B mean(t | Republican)                           [model contrast without the mirror]
  exaggeration_all / exaggeration_party = mean|model_t| / mean|human_t| (11 thermometers / 2 party thermometers)
  pattern_corr = correlation between model_t and human_t over the 11 thermometers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import MODELS, THERMS, boot_weights, eligible_keys, load_human_and_attrs, pct_ci, wide

REP, DEM = 0, 1
OTHER = [i for i in range(len(THERMS)) if i not in (REP, DEM)]


def strata(attrs: pd.DataFrame) -> dict:
    pid, ideo = attrs.pid.values, attrs.ideo.values
    cross = ((pid == "Democrat") & pd.Series(ideo).str.contains("conservative").values) | \
            ((pid == "Republican") & pd.Series(ideo).str.contains("liberal").values)
    mod = ideo == "moderate"
    return {"all": np.ones(len(attrs), bool), "non_moderate": ~mod, "moderate": mod,
            "dem_to_rep": pid == "Democrat", "rep_to_dem": pid == "Republican", "excl_cross_pressured": ~cross}


def _mean_t(W, x, m):
    """Weighted mean per thermometer of x (R x T; may contain NaN) over m (R x T bool). NaN -> 0 only AFTER masking:
    missing cells outside `m` never enter, and cells inside `m` are finite by construction."""
    m = m & ~np.isnan(x)
    return (W @ np.where(m, x, 0.0)) / (W @ m.astype(np.float32))


def stratum_stats(Dl, V, is_dem, mask, W) -> dict:
    """Dl: (R x T) delta = D - B; V: (R x T) bool valid cells; is_dem: (R,) bool; mask: (R,) bool stratum; W: (B x R)."""
    m = V & mask[:, None]
    out = {"n_resp_weighted": (W @ mask.astype(np.float32))}
    mt = _mean_t(W, Dl, m)                                               # (B×T)
    out["mean_delta_rep_party"], out["mean_delta_dem_party"] = mt[:, REP], mt[:, DEM]
    g = Dl[:, DEM] - Dl[:, REP]
    vg = V[:, REP] & V[:, DEM] & mask
    sgn = np.where(is_dem, -1.0, 1.0)
    out["mean_gap_shift"] = (W @ np.where(vg, g, 0)) / (W @ vg.astype(np.float32))      # signed: only interpretable within ONE direction
    # aligned with the expected direction (positive = the model moves towards the mirror): g * sgn with sgn = -1 (Dem->Rep) and +1 (Rep->Dem)
    out["mean_aligned_gap_shift"] = (W @ np.where(vg, g * sgn, 0)) / (W @ vg.astype(np.float32))
    vr, vd = V[:, REP] & mask, V[:, DEM] & mask
    out["mean_aligned_delta_rep_party"] = (W @ np.where(vr, -sgn * Dl[:, REP], 0)) / (W @ vr.astype(np.float32))   # expected > 0
    out["mean_aligned_delta_dem_party"] = (W @ np.where(vd, sgn * Dl[:, DEM], 0)) / (W @ vd.astype(np.float32))    # expected > 0
    aligned = (g * sgn > 0) & vg
    tie = (g == 0) & vg
    out["expected_dir_share"] = (W @ aligned.astype(np.float32)) / (W @ vg.astype(np.float32))
    out["tie_share"] = (W @ tie.astype(np.float32)) / (W @ vg.astype(np.float32))
    out["mean_abs_gap_shift"] = (W @ np.where(vg, np.abs(g), 0)) / (W @ vg.astype(np.float32))
    out["spill_mean"] = mt[:, OTHER].mean(1)
    mta = _mean_t(W, np.abs(Dl), m)
    out["spill_mean_abs"] = mta[:, OTHER].mean(1)
    return out


def contrast_stats(Dl, Bv, Hv, V, is_dem, W) -> dict:
    """Human vs model contrast (full eligible N). Bv: B (R x T); Hv: human (R x T); V: valid cells of delta."""
    dem, rep = is_dem[:, None] & V, (~is_dem)[:, None] & V
    e_dem = _mean_t(W, Dl, dem); e_rep = _mean_t(W, Dl, rep)
    model_t = (e_rep - e_dem) / 2
    hm = ~np.isnan(Hv)
    hdem = _mean_t(W, Hv, is_dem[:, None] & hm); hrep = _mean_t(W, Hv, (~is_dem)[:, None] & hm)
    human_t = hdem - hrep
    bm = ~np.isnan(Bv)
    b_t = _mean_t(W, Bv, is_dem[:, None] & bm) - _mean_t(W, Bv, (~is_dem)[:, None] & bm)
    out = {"exaggeration_all": np.abs(model_t).mean(1) / np.abs(human_t).mean(1),
           "exaggeration_party": np.abs(model_t[:, [REP, DEM]]).mean(1) / np.abs(human_t[:, [REP, DEM]]).mean(1),
           "b_contrast_to_human_all": np.abs(b_t).mean(1) / np.abs(human_t).mean(1)}
    mc = model_t - model_t.mean(1, keepdims=True); hc = human_t - human_t.mean(1, keepdims=True)
    out["pattern_corr"] = (mc * hc).sum(1) / np.sqrt((mc ** 2).sum(1) * (hc ** 2).sum(1))
    for i, t in enumerate(THERMS):
        out[f"model_t:{t}"], out[f"human_t:{t}"], out[f"b_t:{t}"] = model_t[:, i], human_t[:, i], b_t[:, i]
    return out


def run(resp: pd.DataFrame, b: int = 1000, seed: int = 42, models=None) -> dict:
    rng = np.random.default_rng(seed)
    hw, attrs = load_human_and_attrs()
    ek = eligible_keys(attrs)
    ea = attrs.loc[ek]
    Hv = hw.loc[ek].values
    is_dem = (ea.pid == "Democrat").values
    st = strata(ea)
    models = [m for m in (models or MODELS) if (resp.model == m).any()]
    rows, crow, trow = [], [], []
    for m in models:
        Bv = wide(resp, m, "B", 0, "F", ek, block="full").values
        Dv = wide(resp, m, "D", 0, "F", ek, block="full").values
        V = ~np.isnan(Bv) & ~np.isnan(Dv)
        Dl = np.where(V, Dv - Bv, np.nan)
        Dl0 = np.nan_to_num(Dl)
        W1 = np.ones((1, len(ek)), dtype=np.float32)
        acc, cacc = {}, {}
        # point estimates
        pt = {s: {k: float(v[0]) for k, v in stratum_stats(Dl0, V, is_dem, msk, W1).items()} for s, msk in st.items()}
        cpt = {k: float(v[0]) for k, v in contrast_stats(Dl0, Bv, Hv, V, is_dem, W1).items()}      # NaN preserved (the human benchmark has structurally missing cells)
        # median of |g| (point estimate)
        gg = Dl[:, DEM] - Dl[:, REP]
        for s, msk in st.items():
            pt[s]["median_abs_gap_shift"] = float(np.nanmedian(np.abs(gg[msk])))
        for W in boot_weights(len(ek), b, rng):
            for s, msk in st.items():
                for k, v in stratum_stats(Dl0, V, is_dem, msk, W).items():
                    acc.setdefault((s, k), []).append(v)
            for k, v in contrast_stats(Dl0, Bv, Hv, V, is_dem, W).items():
                cacc.setdefault(k, []).append(v)
        for s in st:
            row = {"model": m, "stratum": s}
            for k, v in pt[s].items():
                if (s, k) in acc:
                    lo, hi = pct_ci(np.concatenate(acc[(s, k)]))
                    row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
                else:
                    row[k] = v
            rows.append(row)
        crow_m = {"model": m}
        for k, v in cpt.items():
            if ":" in k:
                continue
            lo, hi = pct_ci(np.concatenate(cacc[k]))
            crow_m.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
        crow.append(crow_m)
        for t in THERMS:
            trow.append({"model": m, "thermometer": t, **{n: cpt[f"{n}:{t}"] for n in ("model_t", "human_t", "b_t")},
                         **{f"{n}_lo": pct_ci(np.concatenate(cacc[f"{n}:{t}"]))[0] for n in ("model_t",)},
                         **{f"{n}_hi": pct_ci(np.concatenate(cacc[f"{n}:{t}"]))[1] for n in ("model_t",)}})
    return {"strata": pd.DataFrame(rows), "contrast": pd.DataFrame(crow), "per_thermometer": pd.DataFrame(trow),
            "n_elig": len(ek)}
