# When Does Prompt Structure Matter? A Multi-Model Evaluation of Synthetic Survey Respondents

Replication package for the paper of the same name (*Survey Practice*, Special Issue on Artificial Intelligence
Assisted Surveys). The study asks whether the way respondent information is written into a prompt changes how well
LLM-based synthetic respondents reproduce human survey answers. It uses 7,526 matched respondents from the ANES
2016/2020 cumulative file (the ground truth of Bisbee et al. 2024), eleven feeling thermometers, and four models
(Claude Haiku 4.5, Ministral 3B, Qwen3 32B, GPT-OSS-120B).

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
data/        ANES respondent attributes and benchmark, instrument, protocol, and every model call
prompts/     Exact prompt construction (build_prompts.py) with hash verification against data/calls.csv
analysis/    Pipeline that turns data/ into every table and figure in the paper (+ tests/)
results/     Output of the pipeline (CSV tables, summary.json, results_manifest.json)
figures/     Figures used in the paper
```

See [`data/README.md`](data/README.md) for file formats and licenses.

## Reproducing the results

```bash
pip install -r requirements.txt
Rscript -e 'install.packages(c("fixest", "dplyr", "tidyr", "readr"))'   # conditional fidelity

python3 prompts/build_prompts.py verify        # every call's prompt hash is reproduced from respondents.csv
python3 analysis/tests/test_pipeline.py        # pipeline checked against synthetic data with known answers
python3 analysis/run_all.py --b 1000           # all analyses -> results/ (conditional fidelity step takes ~35 min)
python3 analysis/build_tables.py results       # results/tables.md and results/facts.json
python3 analysis/make_figures.py results       # -> figures/
```

`run_all.py` refuses incomplete or inconsistent data (unit counts, duplicates, composition of each block), writes
`results/results_manifest.json` with the SHA-256 of every input file and analysis script, and uses a fixed seed
(`--seed`, default 42). `analysis/README_ANALYSIS.md` defines every statistic.

All tables in `results/` regenerate exactly from the stored draws, except the bootstrap bounds of the conditional-fidelity
analysis (`results/conditional/`): they come from a parallel R bootstrap and a rerun changes them by less than 0.5
percentage points. The files shipped here are the ones reported in the paper; point estimates are exact.

## Reproducibility scope

The responses in `data/calls.csv` were generated through Amazon Bedrock (Haiku 4.5, Qwen3 32B and GPT-OSS-120B in
batch mode; Ministral 3B synchronously) at temperature 0.3 (GPT-OSS with reasoning effort `low`) with a forced
tool call, no conversational memory and no retries of malformed outputs. The execution harness itself is not
distributed. Everything needed to repeat the calls is: `prompts/build_prompts.py` (byte-exact system prompt,
user messages and item order), `data/protocol.json` (model identifiers, parameters, tool schema) and
`data/respondents.csv`. Re-running the models will not reproduce the stored draws (sampling is stochastic and hosted
models change); the analysis reproduces from the stored draws.

## License

Code: MIT (`LICENSE`). Newly generated data, figures and tables: CC BY 4.0. ANES-derived fields remain subject to the
ANES Terms of Use; details in `data/README.md`.

## Citation

Please cite the paper (details to be added on publication) and the original study:

```bibtex
@article{bisbee2024synthetic,
  title={Synthetic Replacements for Human Survey Data? The Perils of Large Language Models},
  author={Bisbee, James and Clinton, Joshua D and Dorff, Cassy and Kenkel, Brenton and Larson, Jennifer M},
  journal={Political Analysis},
  volume={32}, number={4}, pages={401--416}, year={2024},
  doi={10.1017/pan.2024.5}
}
```
