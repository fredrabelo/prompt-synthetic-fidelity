# When Does Prompt Structure Matter? A Multi-Model Evaluation of Synthetic Survey Respondents

Replication materials for the paper of the same name (*Survey Practice*, Special Issue on Artificial Intelligence
Assisted Surveys; full citation to be added on publication). The study asks whether the way respondent information
is written into a prompt changes how well LLM-based synthetic respondents reproduce human survey answers. It uses
7,526 matched person-wave records (6,027 distinct people) from the ANES 2016/2020 cumulative file (the ground truth of
Bisbee et al. 2024), eleven feeling thermometers, and four models (Claude Haiku 4.5, Ministral 3B, Qwen3 32B,
GPT-OSS-120B).

Prompt conditions (same facts, same questions; only the respondent block changes):

| Condition | Respondent information | Respondents |
|---|---|---|
| A | none (question template only) | 7,526 |
| B | eleven attributes plus country, as a narrative persona | 7,526 |
| C | the same facts in the same order, one labelled field per line | 7,526 |
| D | B with party identification and ideology mirrored (Democrat/Republican respondents) | 5,115 |

Design blocks per model (32,693 calls, 130,772 in total): `full` (draw 0, A/B/C for everyone, D for partisans),
`cons` (four more identical draws for a fixed stratified subsample of 250 partisans, A–D) and `rev` (the same subsample
with the eleven items in reversed order).

## Repository layout

```
data/        ANES respondent attributes and benchmark, person linkage, instrument, protocol, and every model call
prompts/     Exact prompt construction (build_prompts.py) with hash verification against data/calls.csv
analysis/    reanalyze.py (main analysis), build_figures.py, build_person_linkage.py, and the diagnostics pipeline
results/     Output tables, person-level validation folds, prediction arrays, and run manifest
figures/     Figures
tests/       Integrity checks for the linkage, person-grouped folds, and variance partition
```

## Reproduce the reported analyses

Requires Python 3.10 or later with NumPy, pandas, and matplotlib. From the repository root:

```bash
python -m pip install -r requirements.txt
python prompts/build_prompts.py verify         # 130,772 calls, 28,693 distinct prompts, zero hash mismatches
OPENBLAS_NUM_THREADS=1 python analysis/reanalyze.py     # all main tables -> results/
python analysis/build_figures.py               # -> figures/
python -m unittest discover -s tests -v
```

The analysis uses only the included frozen calls and data and makes no model API requests. It writes the reported
tables and bootstrap intervals to `results/` using seed `20261111` and 1,000 person-cluster resamples. Validation
folds are assigned to people, and every bootstrap resample draws people with all of their observed waves.
Confidence intervals for the out-of-fold benchmarks condition on the fitted out-of-fold predictions (no nested refit).

## Rebuild the cross-wave person linkage

`data/person_linkage.csv` is the linkage used in the analysis. To regenerate it, download the May 19, 2026 ANES
2016–2020–2024 Panel Merged File from the
[ANES data center](https://electionstudies.org/data-center/2016-2020-2024-panel-merged-study/) and run:

```bash
python analysis/build_person_linkage.py --panel /path/to/anes_mergedfile_2016-2020-2024panel.csv
```

The script reads the public case identifiers `V160001` and `V200001` and checks the expected totals: 7,526
person-wave records, 6,027 people, and 1,499 people linked across 2016 and 2020. The raw ANES file is not
redistributed here; the generated mapping is included.

## Output-validity and conditional-coefficient diagnostics

The diagnostics reported in Supplement S8 (output validity by model and condition, source-by-attribute
conditional coefficients) come from a separate pipeline:

```bash
pip install -r requirements.txt
Rscript -e 'install.packages(c("fixest", "dplyr", "tidyr", "readr"))'   # conditional coefficients
python analysis/tests/test_pipeline.py        # pipeline checked against synthetic data with known answers
python analysis/run_all.py --b 1000           # -> results/a*_*.csv, results/conditional/ (the R step takes ~35 min)
python analysis/build_tables.py results       # results/tables.md and results/facts.json
python analysis/make_figures.py results
```

`run_all.py` refuses incomplete or inconsistent data, writes `results/results_manifest.json` with the SHA-256 of every
input file and analysis script, and uses a fixed seed (`--seed`, default 42). `analysis/README_ANALYSIS.md` defines
the statistics of this pipeline. Bootstrap bounds of the conditional-coefficient analysis (`results/conditional/`)
come from a parallel R bootstrap and a rerun changes them by less than 0.5 percentage points; descriptive point
estimates are exact. The tables in this pipeline resample records, not people, and are descriptive.

The ANES-derived fields remain subject to ANES terms and citation requirements. See [`data/README.md`](data/README.md)
for file formats, provenance, and licenses.
