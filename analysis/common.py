"""Shared loading and bootstrap utilities for the analysis pipeline.

Data model
- ``calls``: one row per model call (model, block, condition, external_key, draw, order, valid, invalid_reason, token counts).
- ``resp``:  one row per (valid call x thermometer): model, block, external_key, cond, draw, order, thermometer, value.
- ``wide(resp, model, cond, draw, order)``: DataFrame (external_key x 11 thermometers), NaN where invalid or missing.

Design checks
- The loader refuses incomplete or inconsistent data: every model must contain exactly the pre-specified units of the
  ``full``, ``cons`` and ``rev`` blocks, with no duplicates.
- Invalid responses are excluded from response statistics (NaN) but remain in ``calls`` for the validity tables.
- Nothing in this module calls an API or writes to ``data/``.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("P1_DATA_DIR") or ROOT / "data")  # tests point this at a synthetic data directory
MODELS = ["haiku", "ministral", "qwen", "gptoss"]
CONDS = ["A", "B", "C", "D"]
THERMS = [
    "thermometer_republican_party", "thermometer_democratic_party", "thermometer_liberals", "thermometer_conservatives",
    "thermometer_muslims", "thermometer_gays_lesbians", "thermometer_christians", "thermometer_asian_americans",
    "thermometer_black_americans", "thermometer_white_americans", "thermometer_jews",
]
# Units expected per block and model (final design)
EXPECTED_UNITS = {"full": 27693, "cons": 4000, "rev": 1000}
FINAL_BLOCKS = ("full", "cons", "rev")


class DataError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Loaded:
    calls: pd.DataFrame
    resp: pd.DataFrame
    inputs: dict          # SHA-256 of the data files read (traceability)
    warnings: list


def expected_unit_set(block: str) -> set:
    """Exact set of (external_key, cond, draw, order) that each block must contain (pre-specified design)."""
    hw, attrs = load_human_and_attrs()
    if block == "full":
        el = set(eligible_keys(attrs))
        return {(k, c, 0, "F") for k in hw.index for c in "ABC"} | {(k, "D", 0, "F") for k in el}
    sub = sub250_keys()
    if block == "cons":
        return {(k, c, d, "F") for k in sub for c in CONDS for d in (1, 2, 3, 4)}
    if block == "rev":
        return {(k, c, 0, "R") for k in sub for c in CONDS}
    raise ValueError(block)


def load_responses(models=None, allow_partial=False) -> Loaded:
    """Load ``data/calls.csv``.

    allow_partial=False (default): requires complete full/cons/rev blocks for every requested model.
    allow_partial=True: skips the completeness checks (development and tests on synthetic data only).
    """
    models = models or MODELS
    path = DATA / "calls.csv"
    raw = pd.read_csv(path, dtype={"invalid_reason": "string", "stop_reason": "string"})
    raw["valid"] = raw["valid"].astype(bool)
    calls = raw.rename(columns={"condition": "cond"})
    calls = calls[calls.model.isin(models)].copy()
    calls["block"] = calls["block"].astype(str)
    missing = [m for m in models if m not in set(calls.model)]
    if missing and not allow_partial:
        raise DataError(f"no calls for models: {missing}")
    dup = calls.duplicated(["model", "block", "external_key", "cond", "draw", "order"])
    if dup.any():
        raise DataError(f"duplicate units (model x block x respondent x cond x draw x order): {int(dup.sum())}")
    if not allow_partial:
        for m in models:
            cm = calls[calls.model == m]
            for blk in FINAL_BLOCKS:
                got = set(map(tuple, cm[cm.block == blk][["external_key", "cond", "draw", "order"]].itertuples(index=False)))
                if len(got) != EXPECTED_UNITS[blk]:
                    raise DataError(f"{m}/{blk}: {len(got)} units, expected {EXPECTED_UNITS[blk]}")
                exp = expected_unit_set(blk)
                if got != exp:
                    raise DataError(f"{m}/{blk}: unexpected composition; missing {len(exp - got)}, extra {len(got - exp)}")
            modes = cm.groupby("block")["serving_mode"].nunique()
            if (modes > 1).any():
                raise DataError(f"{m}: more than one serving mode within a block")
    valid = calls[calls.valid]
    long = valid.melt(id_vars=["model", "block", "external_key", "cond", "draw", "order"], value_vars=THERMS,
                      var_name="thermometer", value_name="value").dropna(subset=["value"])
    long["value"] = long["value"].astype(float)
    if ((long.value < 0) | (long.value > 100)).any() or (long.value % 1 != 0).any():
        raise DataError("valid responses must be integers between 0 and 100")
    per_call = long.groupby(["model", "block", "external_key", "cond", "draw", "order"]).size()
    if (per_call != len(THERMS)).any():
        raise DataError("valid calls must have exactly eleven responses")
    inputs = {str(p.relative_to(DATA.parent)) if DATA.parent in p.parents else str(p): sha256_file(p)
              for p in (path, DATA / "respondents.csv", DATA / "human_benchmark.csv", DATA / "subsample_250.json") if p.exists()}
    resp = long[["model", "block", "external_key", "cond", "draw", "order", "thermometer", "value"]].reset_index(drop=True)
    calls = calls.drop(columns=THERMS + ["prompt_sha256"], errors="ignore").reset_index(drop=True)
    return Loaded(calls=calls, resp=resp, inputs=inputs, warnings=[])


def wide(resp: pd.DataFrame, model: str, cond: str, draw: int = 0, order: str = "F", keys=None, block=None) -> pd.DataFrame:
    """external_key x THERMS (NaN = missing/invalid). ``keys`` fixes and orders the rows."""
    q = (resp.model == model) & (resp.cond == cond) & (resp.draw == draw) & (resp.order == order)
    if block is not None:
        q &= resp.block == block
    w = resp[q].pivot(index="external_key", columns="thermometer", values="value").reindex(columns=THERMS)
    if keys is not None:
        w = w.reindex(keys)
    return w


def load_human_and_attrs():
    human = pd.read_csv(DATA / "human_benchmark.csv")
    hw = human.pivot(index="external_key", columns="thermometer", values="human_value").reindex(columns=THERMS)
    attrs = pd.read_csv(DATA / "respondents.csv").set_index("external_key")
    hw = hw.reindex(attrs.index)
    return hw, attrs


def eligible_keys(attrs: pd.DataFrame):
    """Partisans (Democrat/Republican): eligible for the mirror condition (all have a mappable ideology)."""
    return attrs.index[attrs.pid.isin(["Democrat", "Republican"])]


def sub250_keys():
    return json.loads((DATA / "subsample_250.json").read_text())["externalKeys"]


# -- Respondent bootstrap --------------------------------------------------
def boot_weights(n: int, b: int, rng: np.random.Generator, chunk: int = 250):
    """Multinomial weights (B x n): how many times each respondent enters the resample."""
    done = 0
    while done < b:
        k = min(chunk, b - done)
        idx = rng.integers(0, n, size=(k, n))
        w = np.zeros((k, n), dtype=np.float32)
        rows = np.repeat(np.arange(k), n)
        np.add.at(w, (rows, idx.ravel()), 1.0)
        yield w
        done += k


def pct_ci(x: np.ndarray, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return (float(np.percentile(x, lo)), float(np.percentile(x, hi))) if len(x) else (float("nan"), float("nan"))


def fmt_ci(point, ci, nd=2):
    return f"{point:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
