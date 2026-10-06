"""Reliability of generation (AAPOR: reliability) and sampling noise.

Data: 250 partisans (subsample_250.json) x 4 conditions x 5 identical draws per prompt (draw 0 of block `full` + draws 1-4 of `cons`).

Three families of statistics, kept separate:
 1. Consistency within the same prompt (noise): exact / within-5 agreement, mean absolute difference between pairs of draws
    (10 pairs), SD across draws, empirical entropy (bits) of the 5 draws per (respondent, item), ICC(1) per thermometer.
    Entropy and ICC use only cells with all 5 draws valid (share reported in `complete_share`).
 2. Distance between DIFFERENT prompts relative to noise: excess = E|X_d - Y_d| (same draw index, different conditions) minus the
    mean noise of X and Y. If about 0, the difference between representations cannot be told apart from the prompt's own variability.
 3. Distributional divergence: JSD (and KL with 0.5 smoothing) between pooled histograms (10 bins) per thermometer.
    Equal-size comparison: 2 draws vs 2 draws, averaged over all pairs of subsets; between conditions (any subsets) against the
    same-prompt baseline (disjoint subsets). The 5-vs-5 comparison is descriptive only (never subtracted from a baseline of a different size).
    The mean cross-condition distance uses all 25 X_i x Y_j pairs (within-prompt noise uses the 10 pairs i<j).
Intervals: respondent-level bootstrap (resamples the 250). Entropy and ICC describe the generation regime used (temperature 0.3,
low reasoning effort for GPT-OSS), not a universal property of the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import MODELS, THERMS, boot_weights, pct_ci, sub250_keys, wide

DRAWS = [0, 1, 2, 3, 4]
BINS = np.linspace(0, 100.0001, 11)   # 10 bins; 100 falls in the last one
CROSS = [("B", "C"), ("A", "B"), ("B", "D")]
DIV_PAIRS = [("B", "C"), ("A", "B")]   # B-D omitted: the mirror swaps Dem/Rep and the ~50/50 mix leaves the AGGREGATE histogram almost unchanged (JSD insensitive)


def draws_array(resp: pd.DataFrame, model: str, cond: str, keys) -> np.ndarray:
    """(R x 11 x 5) with NaN; draw 0 from block full, draws 1-4 from cons (original order)."""
    out = np.full((len(keys), len(THERMS), len(DRAWS)), np.nan)
    for i, d in enumerate(DRAWS):
        out[:, :, i] = wide(resp, model, cond, d, "F", keys, block="full" if d == 0 else "cons").values
    return out


def paired_draw_se_table(arrays: dict, models: list[str]) -> pd.DataFrame:
    """SE of the paired mean difference between two draws of the same prompt."""
    rows = []
    for model in models:
        for cond in "BC":
            A = arrays[(model, cond)]
            for draw_i in range(A.shape[2]):
                for draw_j in range(draw_i + 1, A.shape[2]):
                    diff = A[:, :, draw_i] - A[:, :, draw_j]
                    valid_items = np.sum(~np.isnan(diff), axis=1)
                    respondent_mean = np.divide(
                        np.nansum(diff, axis=1), valid_items,
                        out=np.full(A.shape[0], np.nan), where=valid_items > 0,
                    )
                    x = respondent_mean[~np.isnan(respondent_mean)]
                    rows.append({
                        "model": model, "cond": cond, "draw_i": draw_i,
                        "draw_j": draw_j, "n_resp": len(x),
                        "se_paired_mean": float(np.std(x, ddof=1) / np.sqrt(len(x))),
                    })
    return pd.DataFrame(rows)


def _w(W, per_resp):
    """Weighted sum over respondents of an array (R, ...) with weights W (B x R)."""
    return np.tensordot(W, per_resp, axes=([1], [0]))


def noise_prep(A: np.ndarray) -> dict:
    """PER-RESPONDENT quantities (independent of the bootstrap weights), computed once."""
    K = A.shape[2]
    ident = np.zeros(A.shape[0]); w5 = np.zeros_like(ident); ad = np.zeros_like(ident); n = np.zeros_like(ident)
    for i in range(K):
        for j in range(i + 1, K):
            m = ~np.isnan(A[:, :, i]) & ~np.isnan(A[:, :, j])
            d = np.abs(np.where(m, A[:, :, i] - A[:, :, j], 0))
            ident += ((d == 0) & m).sum(1); w5 += ((d <= 5) & m).sum(1); ad += d.sum(1); n += m.sum(1)
    valid = (~np.isnan(A)).sum(2)
    with np.errstate(invalid="ignore"):
        sd = np.where(valid >= 2, np.nanstd(np.where(valid[:, :, None] >= 2, A, 0.0), axis=2, ddof=1), 0.0)
    comp = valid == K
    ent = np.zeros(A.shape[:2])
    for r, t in zip(*np.nonzero(comp)):
        _, c = np.unique(np.rint(A[r, t]), return_counts=True)
        p = c / c.sum(); ent[r, t] = -(p * np.log2(p)).sum()
    icc_parts = []
    for t in range(A.shape[1]):
        c = comp[:, t]
        Y = np.where(c[:, None], A[:, t, :], 0.0)
        m_r = Y.mean(1)
        icc_parts.append((c, m_r, ((Y - m_r[:, None]) ** 2).sum(1)))
    return {"K": K, "ident": ident, "w5": w5, "ad": ad, "n": n, "sd": sd, "sd_cnt": (valid >= 2).sum(1), "comp": comp,
            "ent": ent, "icc": icc_parts, "R": A.shape[0], "T": A.shape[1]}


def noise_stats(P: dict, W: np.ndarray) -> dict:
    K = P["K"]; N = W @ P["n"]
    out = {"noise_identical": (W @ P["ident"]) / N, "noise_within5": (W @ P["w5"]) / N, "noise_mad": (W @ P["ad"]) / N,
           "draw_sd": (W @ P["sd"].sum(1)) / (W @ P["sd_cnt"]),
           "complete_share": (W @ P["comp"].sum(1)) / (W @ np.full(P["R"], P["T"], float)),
           "entropy_bits": (W @ P["ent"].sum(1)) / (W @ P["comp"].sum(1))}
    iccs = []
    for c, m_r, ssw_r in P["icc"]:
        if c.sum() < 3:
            continue
        wc = W * c[None, :]
        N_t = wc.sum(1)
        ybar = (wc @ m_r) / N_t
        ssb = K * (wc @ (m_r ** 2) - N_t * ybar ** 2)
        ssw = wc @ ssw_r
        msb = ssb / (N_t - 1); msw = ssw / (N_t * (K - 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            iccs.append((msb - msw) / (msb + (K - 1) * msw))
    out["icc1"] = np.nanmean(np.vstack(iccs), axis=0)
    return out


def cross_prep(X: np.ndarray, Y: np.ndarray) -> dict:
    """Distance between conditions using ALL K x K pairs of draws (X_i vs Y_j; 25 pairs with K=5). Draws are not naturally
    paired across conditions, so using only i=j would waste data; within-prompt noise uses the K(K-1)/2 = 10 pairs."""
    K = X.shape[2]
    ad = np.zeros(X.shape[0]); idn = np.zeros_like(ad); n = np.zeros_like(ad)
    for i in range(K):
        for j in range(K):
            m = ~np.isnan(X[:, :, i]) & ~np.isnan(Y[:, :, j])
            diff = np.abs(np.where(m, X[:, :, i] - Y[:, :, j], 0))
            ad += diff.sum(1); idn += ((diff == 0) & m).sum(1); n += m.sum(1)
    return {"ad": ad, "idn": idn, "n": n}


def cross_stats(C: dict, W: np.ndarray, noise_mad_avg: np.ndarray) -> dict:
    """Distance between conditions and excess over the mean noise of the two conditions."""
    N = W @ C["n"]
    cross_mad = (W @ C["ad"]) / N
    return {"cross_mad": cross_mad, "cross_identical": (W @ C["idn"]) / N, "noise_mad_avg": noise_mad_avg,
            "excess_mad": cross_mad - noise_mad_avg, "ratio_cross_to_noise": cross_mad / noise_mad_avg}


def _hist(A: np.ndarray, draws=None) -> np.ndarray:
    """(R x T x bins): counts per respondent/thermometer, aggregating the requested draws."""
    draws = DRAWS if draws is None else draws
    R, T, _ = A.shape
    h = np.zeros((R, T, len(BINS) - 1))
    for d in draws:
        v = A[:, :, d]
        idx = np.clip(np.digitize(v, BINS) - 1, 0, len(BINS) - 2)
        ok = ~np.isnan(v)
        for r, t in zip(*np.nonzero(ok)):
            h[r, t, idx[r, t]] += 1
    return h


def _divs(hp: np.ndarray, hq: np.ndarray, W: np.ndarray) -> dict:
    """hp, hq: (R x T x bins). Divergences between weighted pooled histograms; mean over thermometers."""
    P = np.tensordot(W, hp, axes=([1], [0])); Q = np.tensordot(W, hq, axes=([1], [0]))        # (B×T×bins)
    Ps = P + 0.5; Qs = Q + 0.5                                                                   # smoothing for KL
    P = P / P.sum(2, keepdims=True); Q = Q / Q.sum(2, keepdims=True)
    Ps = Ps / Ps.sum(2, keepdims=True); Qs = Qs / Qs.sum(2, keepdims=True)
    M = (P + Q) / 2

    def kl(a, b_):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nansum(np.where(a > 0, a * np.log2(a / b_), 0.0), axis=2)
    return {"jsd": ((kl(P, M) + kl(Q, M)) / 2).mean(1), "kl_pq": kl(Ps, Qs).mean(1)}


def hist_by_draw(A: np.ndarray) -> np.ndarray:
    """(K x R x T x bins): one histogram per draw; subsets of draws are summed later."""
    return np.stack([_hist(A, [d]) for d in range(A.shape[2])])


def div_equal_size(HX: np.ndarray, HY: np.ndarray, W: np.ndarray, size: int = 2, disjoint: bool = False) -> dict:
    """Divergence between X and Y with histograms of the SAME size (`size` draws per side), averaged over all pairs of subsets.
    disjoint=True (within-prompt baseline): X and Y are the same prompt with DISJOINT subsets of draws; False (between conditions): any subsets."""
    from itertools import combinations
    K = HX.shape[0]
    subs = list(combinations(range(K), size))
    acc = {"jsd": 0.0, "kl_pq": 0.0}; cnt = 0
    for sx in subs:
        for sy in subs:
            if disjoint and set(sx) & set(sy):
                continue
            r = _divs(HX[list(sx)].sum(0), HY[list(sy)].sum(0), W)
            for k in acc: acc[k] = acc[k] + r[k]
            cnt += 1
    return {k: v / cnt for k, v in acc.items()}


def run(resp: pd.DataFrame, b: int = 1000, seed: int = 42, models=None) -> dict:
    rng = np.random.default_rng(seed)
    keys = sub250_keys()
    models = [m for m in (models or MODELS) if (resp.model == m).any()]
    W1 = np.ones((1, len(keys)), dtype=np.float32)
    arrays = {(m, c): draws_array(resp, m, c, keys) for m in models for c in "ABCD"}
    hist = {(m, c): _hist(arrays[(m, c)]) for (m, c) in arrays}                  # 5 pooled draws (descriptive, 5 vs 5)
    hbd = {(m, c): hist_by_draw(arrays[(m, c)]) for (m, c) in arrays}             # per draw (for equal-size 2 vs 2 comparisons)

    npre = {key: noise_prep(a) for key, a in arrays.items()}
    cpre = {(m, c1, c2): cross_prep(arrays[(m, c1)], arrays[(m, c2)]) for m in models for c1, c2 in CROSS}

    def compute(W):
        out = {}
        for m in models:
            for c in "ABCD":
                for k, v in noise_stats(npre[(m, c)], W).items():
                    out[(m, c, "noise", k)] = v
            for c1, c2 in CROSS:
                nm = (out[(m, c1, "noise", "noise_mad")] + out[(m, c2, "noise", "noise_mad")]) / 2
                for k, v in cross_stats(cpre[(m, c1, c2)], W, nm).items():
                    out[(m, f"{c1}-{c2}", "cross", k)] = v
                if (c1, c2) not in DIV_PAIRS:
                    continue
                dv5 = _divs(hist[(m, c1)], hist[(m, c2)], W)                       # 5 draws vs 5 draws: descriptive
                cross2 = div_equal_size(hbd[(m, c1)], hbd[(m, c2)], W, 2, disjoint=False)   # 2 vs 2, between conditions
                base2 = [div_equal_size(hbd[(m, c)], hbd[(m, c)], W, 2, disjoint=True) for c in (c1, c2)]   # 2 vs 2, same prompt, disjoint subsets
                for k in dv5:
                    out[(m, f"{c1}-{c2}", "div", k + "_5v5_descriptive")] = dv5[k]
                    out[(m, f"{c1}-{c2}", "div", k + "_2v2_cross")] = cross2[k]
                    out[(m, f"{c1}-{c2}", "div", k + "_2v2_samePrompt_baseline")] = (base2[0][k] + base2[1][k]) / 2
                    out[(m, f"{c1}-{c2}", "div", k + "_2v2_excess")] = cross2[k] - (base2[0][k] + base2[1][k]) / 2
        return out
    point = {k: float(v[0]) for k, v in compute(W1).items()}
    boots = {}
    for W in boot_weights(len(keys), b, rng):
        for k, v in compute(W).items():
            boots.setdefault(k, []).append(v)
    rows = []
    for k, p in point.items():
        lo, hi = pct_ci(np.concatenate(boots[k]))
        rows.append({"model": k[0], "unit": k[1], "family": k[2], "stat": k[3], "value": p, "lo": lo, "hi": hi})
    return {"table": pd.DataFrame(rows), "paired_draw_se": paired_draw_se_table(arrays, models), "n_resp": len(keys)}
