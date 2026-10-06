"""Aggregate fidelity to the ANES, by model and condition.

Metrics (same definitions as the original replication):
  mae     : mean absolute error per thermometer, then simple mean over the 11 thermometers (valid human x synthetic pairs).
  sd_pct  : SD of the pooled synthetic responses as a percentage of the human SD (a constant of the full benchmark).
  corr    : Pearson correlation per thermometer, then mean over thermometers.
Uncertainty: respondent-level bootstrap (whole respondents are resampled; conditions and models stay paired).
Paired contrasts (A-B, A-C, B-C on the full sample) evaluate both conditions on the SAME valid cells (intersection) with the
same resample. Complete-case sensitivity (`cc_*`): only respondents with valid A, B and C on all 11 thermometers.
Between-model heterogeneity: mean of the model differences (interval from the same resample) and SD across models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CONDS, MODELS, THERMS, boot_weights, eligible_keys, load_human_and_attrs, pct_ci, wide


def _prep(H: np.ndarray, S: np.ndarray):
    M = (~np.isnan(H) & ~np.isnan(S)).astype(np.float32)
    h = np.nan_to_num(H).astype(np.float32)
    s = np.nan_to_num(S).astype(np.float32)
    return M, h * M, s * M


def metrics_weighted(H: np.ndarray, S: np.ndarray, W: np.ndarray | None, human_sd: float) -> dict:
    """H, S: (R x T) with NaN. W: (B x R) resampling weights (None = observed sample). Returns arrays of length B (or 1)."""
    M, h, s = _prep(H, S)
    R = H.shape[0]
    if W is None:
        W = np.ones((1, R), dtype=np.float32)
    ae = np.abs(h - s) * M
    n_t = W @ M                                   # (B×T)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = (W @ ae / n_t).mean(axis=1)
        # pooled SD of the synthetic responses (all paired cells)
        n_r = M.sum(1); s1 = s.sum(1); s2 = (s * s).sum(1)
        N = W @ n_r; mean = (W @ s1) / N
        var = ((W @ s2) / N - mean ** 2) * N / (N - 1)
        sd = np.sqrt(var)
        sh, ss = W @ h, W @ s
        shh, sss, shs = W @ (h * h), W @ (s * s), W @ (h * s)
        cov = shs / n_t - (sh / n_t) * (ss / n_t)
        vh = shh / n_t - (sh / n_t) ** 2
        vs = sss / n_t - (ss / n_t) ** 2
        corr = (cov / np.sqrt(vh * vs)).mean(axis=1)
    return {"mae": mae, "sd": sd, "sd_pct": 100 * sd / human_sd, "corr": corr}


CONTRASTS = [("A", "B", "all"), ("A", "C", "all"), ("B", "C", "all")]   # D (mirror) is excluded from fidelity: it is a counterfactual, see a6_mirror.py
METRICS = ("mae", "sd_pct", "corr")


def _joint(S1: np.ndarray, S2: np.ndarray):
    """Keep only cells valid in BOTH conditions (identical denominators)."""
    j = np.isnan(S1) | np.isnan(S2)
    return np.where(j, np.nan, S1), np.where(j, np.nan, S2)


def run(resp: pd.DataFrame, b: int = 1000, seed: int = 42, models=None) -> dict:
    """Tables:
      table      : metrics per condition using all available cells of that condition (descriptive).
      diffs      : PAIRED contrasts, both conditions evaluated on the cells valid in both (primary).
      cc_table / cc_diffs : complete-case sensitivity, only respondents with valid A, B and C on all 11 thermometers.
    """
    rng = np.random.default_rng(seed)
    hw, attrs = load_human_and_attrs()
    keys = hw.index
    human_sd = float(np.nanstd(hw.values, ddof=1))
    elig = keys.isin(eligible_keys(attrs))
    models = [m for m in (models or MODELS) if (resp.model == m).any()]
    H = hw.values
    S = {(m, c): wide(resp, m, c, 0, "F", keys, block="full").values for m in models for c in "ABC"}
    # complete case: A, B and C all complete (11/11) for the respondent
    cc = {m: np.all([~np.isnan(S[(m, c)]).any(1) for c in "ABC"], axis=0) for m in models}
    n = len(keys)

    def sub(arr, msk): return arr[msk]
    all_ = np.ones(n, bool)

    def individual(W, wcc=False):
        out = {}
        for m in models:
            for c in "ABC":
                for ss in ("all", "elig"):
                    msk = elig if ss == "elig" else all_
                    Wm = W[:, msk]
                    if wcc:
                        Wm = Wm * cc[m][msk][None, :]
                    out[(m, c, ss)] = metrics_weighted(sub(H, msk), sub(S[(m, c)], msk), Wm, human_sd)
        return out

    def contrasts(W, wcc=False):
        out = {}
        for m in models:
            for c1, c2, ss in CONTRASTS:
                msk = elig if ss == "elig" else all_
                s1, s2 = _joint(sub(S[(m, c1)], msk), sub(S[(m, c2)], msk))
                Wm = W[:, msk] * (cc[m][msk][None, :] if wcc else 1.0)
                r1 = metrics_weighted(sub(H, msk), s1, Wm, human_sd)
                r2 = metrics_weighted(sub(H, msk), s2, Wm, human_sd)
                out[(m, f"{c1}-{c2}", ss)] = {k: r1[k] - r2[k] for k in METRICS}
        return out

    W1 = np.ones((1, n), dtype=np.float32)
    pt_ind, pt_con = individual(W1), contrasts(W1)
    pt_ind_cc, pt_con_cc = individual(W1, True), contrasts(W1, True)
    acc_ind, acc_con, acc_ind_cc, acc_con_cc = {}, {}, {}, {}
    for W in boot_weights(n, b, rng):
        for acc, res in ((acc_ind, individual(W)), (acc_con, contrasts(W)), (acc_ind_cc, individual(W, True)), (acc_con_cc, contrasts(W, True))):
            for key, d in res.items():
                for k, v in d.items():
                    acc.setdefault(key, {}).setdefault(k, []).append(v)

    def table(pt, acc, names):
        rows = []
        for key, d in pt.items():
            row = dict(zip(names, key))
            for k in METRICS:
                v = float(d[k][0]) if hasattr(d[k], "__len__") else float(d[k])
                lo, hi = pct_ci(np.concatenate(acc[key][k]))
                row.update({k: v, f"{k}_lo": lo, f"{k}_hi": hi})
            rows.append(row)
        return pd.DataFrame(rows)

    t_ind = table(pt_ind, acc_ind, ["model", "cond", "subset"])
    t_ind_cc = table(pt_ind_cc, acc_ind_cc, ["model", "cond", "subset"])
    t_con = table(pt_con, acc_con, ["model", "contrast", "subset"])
    t_con_cc = table(pt_con_cc, acc_con_cc, ["model", "contrast", "subset"])

    def add_mean_across_models(tc, pt, acc):
        rows = []
        for (c1, c2, ss) in sorted({(k[1].split("-")[0], k[1].split("-")[1], k[2]) for k in pt}):
            keys_ = [(m, f"{c1}-{c2}", ss) for m in models if (m, f"{c1}-{c2}", ss) in pt]
            if not keys_:
                continue
            row = {"model": "MEAN_ACROSS_MODELS", "contrast": f"{c1}-{c2}", "subset": ss}
            for k in METRICS:
                dm = np.mean([np.concatenate(acc[kk][k]) for kk in keys_], axis=0)
                p = float(np.mean([pt[kk][k] for kk in keys_]))
                lo, hi = pct_ci(dm)
                row.update({k: p, f"{k}_lo": lo, f"{k}_hi": hi})
                row[f"{k}_between_model_sd"] = float(np.std([pt[kk][k] for kk in keys_], ddof=1)) if len(keys_) > 1 else float("nan")
            rows.append(row)
        return pd.concat([tc, pd.DataFrame(rows)], ignore_index=True)
    # pt_con values are arrays of shape (1,) -> floats
    pt_con_f = {k: {m_: float(np.ravel(v)[0]) for m_, v in d.items()} for k, d in pt_con.items()}
    pt_con_cc_f = {k: {m_: float(np.ravel(v)[0]) for m_, v in d.items()} for k, d in pt_con_cc.items()}
    return {"table": t_ind, "diffs": add_mean_across_models(t_con, pt_con_f, acc_con), "cc_table": t_ind_cc,
            "cc_diffs": add_mean_across_models(t_con_cc, pt_con_cc_f, acc_con_cc),
            "human_sd": human_sd, "n_all": int(n), "n_elig": int(elig.sum()),
            "n_complete_case": {m: int(cc[m].sum()) for m in models}}
