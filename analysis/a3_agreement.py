"""Within-respondent agreement between conditions.

For each model and pair of conditions (B-C is the main representation contrast; A-B and A-C measure the value of
information; B-D measures the response to the partisan mirror), the SAME people and thermometers are compared
(draw 0, original order, valid pairs):
  identical  : share of pairs with the same value
  within5/10 : share with |difference| <= 5 / <= 10 points
  mad / median_ad : mean / median absolute difference
  signed_mean/sd  : signed difference (first minus second condition)
  pooled_cell_pearson : correlation over all pooled cells (mixes agreement, differences between the 11 groups and between people)
  per_thermometer_pearson_mean : correlation across respondents per thermometer, mean of the 11
  within_respondent_vector_pearson_mean/median : correlation of each respondent's 11-response vectors (>= 3 pairs, non-zero variance)
  pooled_cell_spearman_point_only : pooled Spearman, point estimate only (no interval)
  resp_mad_median : median over respondents of the respondent's mean absolute difference
Intervals: respondent-level bootstrap (2.5-97.5 percentiles).
Without repeated draws of the same prompt this agreement mixes representation effects with sampling noise; the noise
benchmark (same prompt, different draws) is in a4_reliability.py (250 partisans) and should accompany these numbers.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

from common import MODELS, THERMS, boot_weights, eligible_keys, load_human_and_attrs, pct_ci, wide

PAIRS = [("B", "C", "all"), ("A", "B", "all"), ("A", "C", "all"), ("B", "D", "elig")]


def pair_stats_weighted(X: np.ndarray, Y: np.ndarray, W: np.ndarray | None):
    """X, Y: (R x T) with NaN. W: (B x R) or None. Returns a dict of arrays of shape (B,)."""
    M = (~np.isnan(X) & ~np.isnan(Y)).astype(np.float32)
    x = np.nan_to_num(X).astype(np.float32) * M
    y = np.nan_to_num(Y).astype(np.float32) * M
    R = X.shape[0]
    if W is None:
        W = np.ones((1, R), dtype=np.float32)
    d = (x - y)
    ad = np.abs(d)
    n_r = M.sum(1)
    N = W @ n_r
    out = {
        "identical": (W @ (((ad == 0) & (M > 0)).sum(1).astype(np.float32))) / N,
        "within5": (W @ (((ad <= 5) & (M > 0)).sum(1).astype(np.float32))) / N,
        "within10": (W @ (((ad <= 10) & (M > 0)).sum(1).astype(np.float32))) / N,
        "mad": (W @ ad.sum(1)) / N,
        "signed_mean": (W @ d.sum(1)) / N,
    }
    out["signed_sd"] = np.sqrt(np.maximum((W @ (d * d).sum(1)) / N - out["signed_mean"] ** 2, 0))

    def pooled_corr(a, bb):
        sa, sb = W @ a.sum(1), W @ bb.sum(1)
        saa, sbb, sab = W @ (a * a).sum(1), W @ (bb * bb).sum(1), W @ (a * bb).sum(1)
        cov = sab / N - (sa / N) * (sb / N)
        return cov / np.sqrt((saa / N - (sa / N) ** 2) * (sbb / N - (sb / N) ** 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pooled_cell_pearson"] = pooled_corr(x, y)      # all cells pooled: mixes agreement, differences between the 11 groups and between people
        # (1) correlation per THERMOMETER (across respondents), then mean of the 11
        n_t = W @ M
        sx, sy = W @ x, W @ y
        sxx, syy, sxy = W @ (x * x), W @ (y * y), W @ (x * y)
        cov = sxy / n_t - (sx / n_t) * (sy / n_t)
        out["per_thermometer_pearson_mean"] = (cov / np.sqrt((sxx / n_t - (sx / n_t) ** 2) * (syy / n_t - (sy / n_t) ** 2))).mean(axis=1)
        # (2) WITHIN-respondent correlation: each person's vector of 11 responses (>= 3 pairs and variance > 0), weighted mean over people
        r_i, def_i = _within_resp_corr(X, Y)
        out["within_respondent_vector_pearson_mean"] = (W @ np.where(def_i, r_i, 0.0)) / (W @ def_i.astype(np.float32))
    return out


def _within_resp_corr(X: np.ndarray, Y: np.ndarray):
    """Pearson correlation between each respondent's vectors X_i and Y_i (11 thermometers); defined with >= 3 pairs and non-zero variance in both."""
    M = ~np.isnan(X) & ~np.isnan(Y)
    n = M.sum(1)
    x = np.where(M, X, 0.0); y = np.where(M, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = x.sum(1) / n, y.sum(1) / n
        dx, dy = np.where(M, X - mx[:, None], 0.0), np.where(M, Y - my[:, None], 0.0)
        r = (dx * dy).sum(1) / np.sqrt((dx ** 2).sum(1) * (dy ** 2).sum(1))
    ok = (n >= 3) & np.isfinite(r)
    return np.where(ok, r, 0.0), ok


def _ranks(X: np.ndarray, Y: np.ndarray):
    """Average ranks of the valid pairs, pooled over all cells."""
    M = ~np.isnan(X) & ~np.isnan(Y)
    rx = np.full(X.shape, np.nan); ry = np.full(X.shape, np.nan)
    rx[M] = pd.Series(X[M]).rank(method="average").values
    ry[M] = pd.Series(Y[M]).rank(method="average").values
    return rx, ry


def run(resp: pd.DataFrame, b: int = 1000, seed: int = 42, models=None) -> dict:
    rng = np.random.default_rng(seed)
    hw, attrs = load_human_and_attrs()
    keys = hw.index
    elig = keys.isin(eligible_keys(attrs))
    models = [m for m in (models or MODELS) if (resp.model == m).any()]
    S = {(m, c): wide(resp, m, c, 0, "F", keys, block="full").values for m in models for c in "ABCD"}
    point, acc = {}, {}
    for m in models:
        for c1, c2, ss in PAIRS:
            msk = elig if ss == "elig" else np.ones(len(keys), bool)
            X, Y = S[(m, c1)][msk], S[(m, c2)][msk]
            point[(m, c1, c2)] = {k: float(v[0]) for k, v in pair_stats_weighted(X, Y, None).items()}
            rx, ry = _ranks(X, Y)                # pooled Spearman: point estimate only (ranks would change with every resample)
            Mv = ~np.isnan(rx)
            point[(m, c1, c2)]["pooled_cell_spearman_point_only"] = float(np.corrcoef(rx[Mv], ry[Mv])[0, 1])
            r_i, def_i = _within_resp_corr(X, Y)
            point[(m, c1, c2)]["within_respondent_vector_pearson_median"] = float(np.median(r_i[def_i]))        # median: point estimate only
            # pooled median of |difference| and per-respondent median (point estimates only)
            M = ~np.isnan(X) & ~np.isnan(Y)
            point[(m, c1, c2)]["median_ad"] = float(np.median(np.abs(X - Y)[M])) if M.any() else float("nan")
            with np.errstate(invalid="ignore"):
                per = np.nanmean(np.where(M, np.abs(X - Y), np.nan), axis=1)
            point[(m, c1, c2)]["resp_mad_median"] = float(np.nanmedian(per))
            point[(m, c1, c2)]["n_pairs"] = int(M.sum())
            point[(m, c1, c2)]["n_resp"] = int(M.any(1).sum())
    n = len(keys)
    for Wfull in boot_weights(n, b, rng):
        for m in models:
            for c1, c2, ss in PAIRS:
                msk = elig if ss == "elig" else np.ones(n, bool)
                r = pair_stats_weighted(S[(m, c1)][msk], S[(m, c2)][msk], Wfull[:, msk])
                for k, v in r.items():
                    acc.setdefault((m, c1, c2), {}).setdefault(k, []).append(v)
    rows = []
    for (m, c1, c2), p in point.items():
        row = {"model": m, "pair": f"{c1}-{c2}", "subset": "elig" if c2 == "D" else "all", **{k: p[k] for k in ("n_pairs", "n_resp", "median_ad", "resp_mad_median")}}
        for k, v in acc[(m, c1, c2)].items():
            lo, hi = pct_ci(np.concatenate(v))
            row.update({k: p[k], f"{k}_lo": lo, f"{k}_hi": hi})
        for k in ("pooled_cell_spearman_point_only", "within_respondent_vector_pearson_median"):
            row[k] = p[k]
        rows.append(row)
    return {"table": pd.DataFrame(rows)}
