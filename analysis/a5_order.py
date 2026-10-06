"""Sensitivity to item order (AAPOR: sensitivity).

Same 250 people and conditions: original order (draw 0 of block `full`) vs reversed order (block `rev`, draw 0).
Statistics
  signed_mean_rev_minus_fwd : mean bias (reversed minus original), per thermometer and overall
  mad / identical / within5 : distance and agreement between the two orders
  position_slope_diagnostic : (DIAGNOSTIC) slope of the regression (rev - fwd) ~ (position_rev - position_fwd); if > 0, items moved
                              towards the end gain points (position effect). Original position p (1..11) -> p_rev = 12 - p.
  order_to_noise_ratio      : MAD(order) / MAD(same-prompt noise); noise is |draw 0 - draws 1-4| in the original order.
                              A ratio near 1 means the order difference is as large as ordinary generation variability.
Intervals: respondent-level bootstrap (250).
CAUTION: each thermometer has ONE fixed position shift (item identity and position are confounded), so the slope is only a
diagnostic and does not identify a causal recency/primacy effect. The design supports "changing the order changes the
responses", not "position causes X".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import MODELS, THERMS, boot_weights, pct_ci, sub250_keys, wide

POS_FWD = np.arange(1, len(THERMS) + 1)
POS_REV = len(THERMS) + 1 - POS_FWD


def stats(F: np.ndarray, Rv: np.ndarray, N0: np.ndarray, W: np.ndarray) -> dict:
    """F, Rv: (R x T) original/reversed order. N0: (R x T x 4) draws 1-4 in the original order (same-prompt noise). W: (B x R)."""
    M = ~np.isnan(F) & ~np.isnan(Rv)
    d = np.where(M, Rv - F, 0.0)
    ad = np.abs(d)
    n_r = M.sum(1)
    N = W @ n_r
    out = {"signed_mean_rev_minus_fwd": (W @ d.sum(1)) / N, "mad": (W @ ad.sum(1)) / N,
           "identical": (W @ ((ad == 0) & M).sum(1)) / N, "within5": (W @ ((ad <= 5) & M).sum(1)) / N}
    # slope: simple regression of d on dx = pos_rev - pos_fwd, pooled (respondent weights)
    dx = (POS_REV - POS_FWD).astype(float)[None, :] * np.ones_like(d)
    sx = W @ (dx * M).sum(1); sy = W @ d.sum(1)
    sxx = W @ ((dx ** 2) * M).sum(1); sxy = W @ (dx * d).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["position_slope_diagnostic"] = (sxy / N - (sx / N) * (sy / N)) / (sxx / N - (sx / N) ** 2)
    # same-prompt noise: |draw 0 - draw k| in the original order
    nm = np.zeros(F.shape[0]); nn = np.zeros(F.shape[0])
    for k in range(N0.shape[2]):
        m = ~np.isnan(F) & ~np.isnan(N0[:, :, k])
        nm += np.where(m, np.abs(F - N0[:, :, k]), 0).sum(1); nn += m.sum(1)
    noise = (W @ nm) / (W @ nn)
    out["noise_mad_same_prompt"] = noise
    out["order_to_noise_ratio"] = out["mad"] / noise
    return out


def per_item(F: np.ndarray, Rv: np.ndarray):
    M = ~np.isnan(F) & ~np.isnan(Rv)
    d = np.where(M, Rv - F, np.nan)
    return np.nanmean(d, axis=0)


def run(resp: pd.DataFrame, b: int = 1000, seed: int = 42, models=None) -> dict:
    rng = np.random.default_rng(seed)
    keys = sub250_keys()
    models = [m for m in (models or MODELS) if (resp.model == m).any()]
    rows, item_rows = [], []
    data = {}
    for m in models:
        for c in "ABCD":
            F = wide(resp, m, c, 0, "F", keys, block="full").values
            Rv = wide(resp, m, c, 0, "R", keys, block="rev").values
            N0 = np.stack([wide(resp, m, c, d, "F", keys, block="cons").values for d in (1, 2, 3, 4)], axis=2)
            data[(m, c)] = (F, Rv, N0)
            for t, v in zip(THERMS, per_item(F, Rv)):
                item_rows.append({"model": m, "cond": c, "thermometer": t, "position_fwd": int(POS_FWD[THERMS.index(t)]),
                                  "position_rev": int(POS_REV[THERMS.index(t)]), "mean_rev_minus_fwd": float(v)})
    W1 = np.ones((1, len(keys)), dtype=np.float32)
    point = {k: {s: float(v[0]) for s, v in stats(*d, W1).items()} for k, d in data.items()}
    acc = {}
    for W in boot_weights(len(keys), b, rng):
        for k, d in data.items():
            for s, v in stats(*d, W).items():
                acc.setdefault(k, {}).setdefault(s, []).append(v)
    for (m, c), p in point.items():
        row = {"model": m, "cond": c}
        for s, v in p.items():
            lo, hi = pct_ci(np.concatenate(acc[(m, c)][s]))
            row.update({s: v, f"{s}_lo": lo, f"{s}_hi": hi})
        rows.append(row)
    return {"table": pd.DataFrame(rows), "per_item": pd.DataFrame(item_rows)}
