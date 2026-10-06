"""Export the long format (one CSV per model) for the conditional-fidelity analysis in R: block `full`, draw 0, original order, A/B/C.

Usage: python3 export_long.py <out_dir> [--allow-partial]
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import load_responses


def main(out_dir: Path, allow_partial=False, models=None):
    L = load_responses(models=models, allow_partial=allow_partial)
    r = L.resp
    r = r[(r.block == "full") & (r.draw == 0) & (r.order == "F") & (r.cond.isin(list("ABC")))]
    out_dir.mkdir(parents=True, exist_ok=True)
    for m, d in r.groupby("model"):
        d[["model", "cond", "external_key", "thermometer", "value"]].rename(columns={"cond": "condition", "value": "synthetic_value"}).to_csv(out_dir / f"synthetic_long_{m}.csv", index=False)
    return sorted(r.model.unique())


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), allow_partial="--allow-partial" in sys.argv))
