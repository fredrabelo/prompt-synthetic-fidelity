# Second-revision analyses

Frozen-data reanalysis; no LLM calls. Reads `../data/` (respondents, benchmark, calls, subsample) and
`../analysis/common.py`, `../analysis/a4_reliability.py` from the first-version package.

```bash
pip install -r ../requirements.txt
OPENBLAS_NUM_THREADS=1 python3 analysis/reanalyze.py      # all tables -> results/
python3 analysis/test_reanalysis.py                       # numerical checks
python3 analysis/make_figures.py                          # figures/
```

Folds are assigned to distinct people (both waves of the 1,499 two-wave people share a fold), and every bootstrap
resamples people with all their waves; seed 20261111, 1,000 resamples. Confidence intervals for the out-of-fold
benchmarks condition on the fitted out-of-fold predictions (no nested refit).

## Results (`results/`)

| File | Content |
|---|---|
| `fidelity.csv`, `fidelity_by_item.csv`, `fidelity_by_year.csv` | MAE, per-item correlation, within-item SD ratio, mean bias for year means, OLS (out of fold) and each model x condition |
| `paired_comparisons.csv` | Paired predictor-minus-predictor contrasts (columns `left`, `right`) with bootstrap intervals, on identical valid cells |
| `five_draw_fidelity.csv` | Mean of five calls vs. grouped-fold OLS on matching cells, 250-partisan subsample |
| `agreement.csv`, `agreement_by_item.csv` | Rank correlation, median absolute difference, signed-difference quantiles, thresholded agreement |
| `mse_decomposition.csv` | Per-item squared bias + SD mismatch + imperfect association = MSE |
| `icc_summary.csv`, `icc_by_item.csv` | Item-specific ICC(1), conditions A-D |
| `format_decomposition.csv` | Corrected C-B contrast components (grand, respondent, item, respondent x item) |
| `response_variance_decomposition.csv` | Eight-component response-scale variance partition (condition x respondent etc.) |
| `order_effects.csv`, `order_available_cases.csv`, `order_decomposition_sensitivity.csv` | Reversed-order shift; available-case version; noise-covariance multipliers 0.5, 1, 2 |
| `adjusted_mirror.csv`, `adjusted_mirror_by_item.csv`, `human_mirror_fit.csv` (design rank per item), `mirror_interaction_sensitivity.csv`, `party_ideology_support.csv` | Regression-adjusted human party-and-ideology contrast, ratios, spillovers, specification check, cell counts |
| `cv_folds.csv` | Fold of each record (grouped by person) |
| `predictions.npz`, `mirror_design.npz` | Item-level predictions (benchmarks and every model x condition, records x 11 items); mirror design matrices |
| `manifest.json` | Seed, resamples, call and person counts, SHA-256 of inputs, linkage file and script |

`data/person_clusters.csv` maps each record to its person; `data/source_provenance.json` records the ANES panel file it was
derived from (see `data/README.md`). The conditional-coefficient results of the first version remain in `../results/conditional/`.
