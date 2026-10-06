# Analysis pipeline

Everything here reads `../data/` (`calls.csv`, `respondents.csv`, `human_benchmark.csv`, `subsample_250.json`). Nothing calls an API.
Run: `python3 run_all.py --b 1000` (writes `../results/`). Tests with synthetic ground truth: `python3 tests/test_pipeline.py`.

## Files
| File | Purpose |
|---|---|
| `common.py` | Loader (refuses incomplete/inconsistent data), `wide()` pivots, cluster-bootstrap weights, shared constants |
| `a1_validity.py` | Validity, truncation, invalid reasons, tokens per model × condition × block × draw; balanced complete-case counts |
| `a2_fidelity.py` | MAE, SD-as-%-of-human, per-item correlation vs ANES; paired contrasts A−B, A−C, B−C (N=7,526) evaluated on the SAME valid cells + complete-case sensitivity; between-model SD. D is excluded (counterfactual → a6) |
| `a3_agreement.py` | Within-respondent agreement for B–C (main), A–B, A–C, B–D: identical/≤5/≤10, MAD, median, signed difference, Pearson, Spearman |
| `a4_reliability.py` | 250 partisans × 5 draws: noise (agreement, MAD, draw SD, entropy, ICC), cross-condition distance vs noise, JSD/KL vs split-half baseline |
| `a5_order.py` | Reversed vs original order (same 250): bias, MAD, position slope, order-to-noise ratio |
| `a6_mirror.py` | Partisan mirror D vs B: strata, direction-aligned shifts, spillovers, exaggeration vs observational human contrast |
| `a7_conditional.R` + `export_long.py` | Conditional-fidelity regressions (Bisbee et al. design) per model × condition A/B/C, cluster bootstrap |
| `a8_summary.py` | `summary.json` (headline numbers) and `aapor_table.csv` |
| `build_tables.py`, `make_figures.py` | Paper tables (`tables.md`, `facts.json`) and figures from a results directory |
| `run_all.py` | Orchestrator; writes `results_manifest.json` with SHA-256 of every input file and analysis file, seed, B |

## Data rules
- `full` 27,693 · `cons` 4,000 · `rev` 1,000 units per model must be complete and match the pre-specified design exactly (`--allow-partial` exists for development only and is recorded in the manifest). Duplicate units, wrong composition, out-of-range values, or a valid call without exactly eleven answers raise errors.
- Invalid responses (malformed output is never retried) are NaN in all response statistics and counted in a1. Every pairwise statistic uses pairs where both responses are valid. Sensitivity to differential missingness: `balanced_complete_case`.
- Draw 0 = block `full`; draws 1–4 = block `cons`; reversed order = block `rev` (draw 0).

## Definitions (all bootstrap CIs resample RESPONDENTS, percentile 2.5–97.5; paired contrasts reuse the same resample across conditions and models)
**Fidelity (a2).** `mae`: per-thermometer mean |human − synthetic| over valid pairs, then mean over the 11 thermometers. `sd_pct`: pooled SD of the
matched synthetic responses ÷ SD of all ANES responses × 100 (the ANES SD is a fixed constant over the full benchmark). `corr`: per-thermometer Pearson, mean over 11.
A, B, C are also reported on the 5,115 eligible partisans (reference for the mirror). Pooled "MEAN_ACROSS_MODELS" is the simple mean of the four models' differences
(CI from the joint resample) with the between-model SD next to it. D is intentionally absent from fidelity tables: a mirrored persona compared with the original respondent's human answer is not a fidelity estimate.

**Within-respondent agreement (a3).** Same respondent, same thermometer, draw 0, original order. Spearman uses ranks computed once on the full sample (bootstrap reuses them: approximation).
Without repeated draws this mixes representation effect and sampling noise — interpret next to a4's noise benchmark.

**Reliability (a4).** Pairwise agreement/MAD over all 10 draw pairs; `draw_sd` mean SD across draws; `entropy_bits` empirical entropy (bits) of the 5 integer draws (max log2 5 = 2.32),
`icc1` one-way ICC with respondents as groups (mean over thermometers); entropy and ICC use only cells with all 5 draws valid (`complete_share`).
`cross_mad` = mean |X_d − Y_d| for draw index d (different conditions); `excess_mad` = cross_mad − mean(noise_mad of the two conditions); ratio = cross/noise.
Divergence: JSD (bits) and KL (add-0.5 smoothing) between pooled 10-bin histograms (B–C, A–B), with split-half baseline (draws {0,1} vs {2,3,4}) of each condition. B–D omitted (the 50/50 mirror leaves the aggregate histogram nearly unchanged).
Entropy/ICC describe the generation regime used (temperature 0.3; gpt-oss reasoning effort low), not a universal model property.

**Order (a5).** For the same 250 respondents and condition: reversed − original. `position_slope`: slope of (reversed − original) on (position_rev − position_orig); `order_to_noise_ratio` = MAD(order)/MAD(draw 0 vs draws 1–4 original order).

**Mirror (a6).** Δ = D − B for the same respondent. Directions: Democrat→Republican, Republican→Democrat. `mean_gap_shift` is signed and only meaningful within one direction;
use the *aligned* versions (positive = in the direction of the mirror) for mixed strata. `expected_dir_share`: share whose g = T_Dem − T_Rep moves the expected way (ties separate).
Exaggeration: model_t = (E_Rep[Δ_t] − E_Dem[Δ_t])/2 (average paired effect of switching Rep→Dem) vs human_t = mean_human(t | Democrat) − mean_human(t | Republican) (OBSERVATIONAL, unadjusted);
`exaggeration_all/party` = mean|model_t| / mean|human_t| over 11 / 2 thermometers. The human contrast is not a causal effect.
Pre-specified strata: all 5,115; non-moderates 4,165 (party and ideology change); moderates 950 (only party changes); each direction; excluding cross-pressured profiles (255).

**Conditional fidelity (a7).** Replicates the per-group, per-year regression with Data × covariate interactions, clustered by respondent; % of interaction coefficients with p<.05 and % of those sign-flipped.
Points are computed from the data; interval = point ± 1.96 × bootstrap SE (the p<.05 threshold makes the statistic discontinuous). Conditions A, B, C only. Number of bootstrap resamples: `--b-cond` (default 1000).

## Known limitations to check
1. Bootstrap for eligible-only (D) contrasts reweights the eligible respondents within the full-sample resample (conditions on membership).
2. Draw independence is an assumption of the sampling mechanism; low overlap rules out caching, it does not prove independence.
3. Reliability/order subsample = 250 partisans stratified from the 5,115 eligible; not representative of independents.
4. JSD/KL use 10 bins of integer 0–100 responses; results depend on binning.
5. Spearman bootstrap holds ranks fixed.
6. gpt-oss temperature 0.3 is accepted by the API but may be ignored by the reasoning model.
