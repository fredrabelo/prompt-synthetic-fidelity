"""Output validity, truncation and token counts, by model x condition x block x draw.

Nothing is discarded here: invalid responses are excluded from the response statistics (malformed outputs are never
retried) and are reported in these tables. A run is flagged (`flag_below_98`) when a condition has fewer than 98% valid
outputs. `balanced_complete_case` counts respondents with valid A, B and C responses on all 11 thermometers, the sample
used by the complete-case sensitivity analyses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import THERMS, wide


def run(calls: pd.DataFrame, resp: pd.DataFrame | None = None) -> dict:
    c = calls.copy()
    c["truncated"] = c.invalid_reason.fillna("").str.contains("truncated")
    g = c.groupby(["model", "block", "cond", "draw", "order"], dropna=False)
    t = g.agg(n=("valid", "size"), valid=("valid", "sum"), truncated=("truncated", "sum"),
              in_tok=("input_tokens", "sum"), out_tok=("output_tokens", "sum")).reset_index()
    t["valid_rate"] = t.valid / t.n
    tm = c.groupby(["model", "cond"]).agg(n=("valid", "size"), valid=("valid", "sum"), truncated=("truncated", "sum"),
                                          in_tok=("input_tokens", "sum"), out_tok=("output_tokens", "sum")).reset_index()
    tm["valid_rate"] = tm.valid / tm.n
    tm["flag_below_98"] = tm.valid_rate < 0.98
    reasons = (c[~c.valid].assign(reason=lambda d: d.invalid_reason.fillna("").str.slice(0, 50))
               .groupby(["model", "cond", "reason"]).size().rename("n").reset_index())
    out = {"by_cell": t, "by_model_cond": tm, "invalid_reasons": reasons}
    if resp is not None:
        from common import load_human_and_attrs
        hw, _ = load_human_and_attrs()
        rows = []
        for m in sorted(c.model.unique()):
            ok = None
            for cond in "ABC":
                w = wide(resp, m, cond, 0, "F", hw.index, block="full")
                v = w.notna().all(axis=1)
                ok = v if ok is None else (ok & v)
            rows.append({"model": m, "n_resp_full": len(hw), "balanced_complete_case": int(ok.sum()), "share": float(ok.mean())})
        out["balanced_complete_case"] = pd.DataFrame(rows)
    return out
