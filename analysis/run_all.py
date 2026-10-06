"""Run the full analysis pipeline and write traceable results.

Usage: python3 run_all.py [--b 1000] [--out results] [--models haiku,qwen] [--allow-partial] [--skip a4,a6]
Output in <out>/ (relative to the repository root): one CSV per analysis table, summary.json, aapor_table.csv and
results_manifest.json (SHA-256 of the input data files and of the analysis code, parameters, seed). Unless --allow-partial is
given (development only), any missing or inconsistent data aborts the run with an error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a1_validity, a2_fidelity, a3_agreement, a4_reliability, a5_order, a6_mirror, a8_summary  # noqa: E402
from common import load_responses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="development only: skip the data completeness checks")
    ap.add_argument("--skip", default="", help="analyses to skip, e.g. a4,a7")
    ap.add_argument("--b-cond", type=int, default=1000, help="bootstrap resamples for a7 (R); use at least 500, ideally 1000")
    a = ap.parse_args()
    out = (HERE.parent / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = a.models.split(",") if a.models else None
    skip = set(a.skip.split(",")) if a.skip else set()
    t0 = time.time()
    L = load_responses(models=models, allow_partial=a.allow_partial)
    for w in L.warnings:
        print("WARNING:", w)
    print(f"loaded: {len(L.calls)} calls, {L.resp.model.nunique()} models")
    res = {}
    steps = [("validity", "a1", lambda: a1_validity.run(L.calls, L.resp)),
             ("fidelity", "a2", lambda: a2_fidelity.run(L.resp, a.b, a.seed, models)),
             ("agreement", "a3", lambda: a3_agreement.run(L.resp, a.b, a.seed, models)),
             ("reliability", "a4", lambda: a4_reliability.run(L.resp, a.b, a.seed, models)),
             ("order", "a5", lambda: a5_order.run(L.resp, a.b, a.seed, models)),
             ("mirror", "a6", lambda: a6_mirror.run(L.resp, a.b, a.seed, models))]
    for name, tag, fn in steps:
        if tag in skip:
            continue
        t = time.time()
        res[name] = fn()
        for k, v in res[name].items():
            if hasattr(v, "to_csv"):
                v.to_csv(out / f"{tag}_{name}_{k}.csv", index=False)
        print(f"{name}: ok ({time.time() - t:.0f}s)")
    if "a7" not in skip:
        # conditional fidelity: export long format -> Rscript -> read the summary for summary.json / AAPOR table
        import pandas as pd
        import export_long
        t = time.time()
        long_dir = out / "long_for_R"
        used = export_long.main(long_dir, allow_partial=a.allow_partial, models=models)
        cond_dir = out / "conditional"
        cmd = ["Rscript", str(HERE / "a7_conditional.R"), str(long_dir), str(cond_dir), str(a.b_cond), ",".join(used)]
        print("R:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        (out / "a7_R_log.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"a7_conditional.R failed (see {out / 'a7_R_log.txt'})")
        res["conditional"] = {"summary": pd.read_csv(cond_dir / "conditional_fidelity_summary.csv"), "points": pd.read_csv(cond_dir / "conditional_fidelity_points.csv")}
        print(f"conditional: ok ({time.time() - t:.0f}s)")
    summary = a8_summary.build(res)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    a8_summary.aapor_table(res).to_csv(out / "aapor_table.csv", index=False)
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py")) + [HERE / "a7_conditional.R"]}
    (out / "results_manifest.json").write_text(json.dumps({
        "bootstrap_resamples": a.b, "seed": a.seed, "models": models, "allow_partial": a.allow_partial,
        "skipped": sorted(skip), "bootstrap_resamples_conditional": a.b_cond, "warnings": L.warnings, "n_calls": int(len(L.calls)), "input_file_sha256": L.inputs, "analysis_code_sha256": code,
        "elapsed_seconds": round(time.time() - t0)}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
