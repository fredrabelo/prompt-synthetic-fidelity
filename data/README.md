# Data

| File | Content |
|---|---|
| `respondents.csv` | One row per ANES respondent (N = 7,526): `external_key, year, age, raceth, gender, marst, education, income, ideo, regis, pid, interest`. The eleven attributes used to build each persona, recoded from the ANES 2016/2020 cumulative file. |
| `human_benchmark.csv` | Long format `external_key, thermometer, human_value`: the real ANES feeling thermometer (0–100) for each respondent and group, where answered. |
| `instrument.json` | The eleven items: key and exact question text. |
| `subsample_250.json` | The fixed subsample of 250 Democrat/Republican respondents (stratified by year × party × ideology, seed 20261001) used for repeated draws and reversed order. |
| `protocol.json` | System prompt, narrative template, tool schema, block definitions, and per-model identifier, serving mode, temperature, reasoning effort and token limit. |
| `calls.csv` | One row per model call (130,772 rows), described below. |
| `explanations/<model>.jsonl.gz` | The one-sentence rationale each model returned for each answer, keyed like `calls.csv`. |

## `calls.csv`

| Column | Meaning |
|---|---|
| `model` | `haiku`, `ministral`, `qwen`, `gptoss` |
| `condition` | `A`, `B`, `C`, `D` |
| `external_key` | Respondent id (joins `respondents.csv`, `human_benchmark.csv`) |
| `block` | `full` (draw 0), `cons` (draws 1–4), `rev` (reversed item order) |
| `draw` | Draw index (0–4) |
| `order` | `F` original item order, `R` reversed |
| `serving_mode` | `batch` or `sync` |
| `valid` | Exactly eleven integer answers in 0–100 were returned. Invalid outputs are kept and never retried |
| `invalid_reason`, `stop_reason` | Why an output was invalid; the model's stop reason |
| `input_tokens`, `output_tokens`, `reasoning_chars` | Usage reported by the provider |
| `prompt_sha256` | SHA-256 of the user message, reproduced by `prompts/build_prompts.py verify` |
| `thermometer_*` | The eleven answers (blank when `valid` is false) |

## Source

Ground truth: ANES Time Series Cumulative Data File (American National Election Studies, University of Michigan and
Stanford University). Original study design: Bisbee, Clinton, Dorff, Kenkel & Larson (2024), *Political Analysis*
32(4):401–416; please cite it and the original Dataverse release (DOI `10.7910/DVN/VPN481`).

## Licensing and terms of use

- The Bisbee et al. replication package, DOI `10.7910/DVN/VPN481`, is released under the CC0 1.0 Public Domain Dedication.
- The ANES-derived respondent attributes and human benchmark values come only from ANES public-release data. ANES
  permits dissemination of derived public-release datasets and asks redistributors to link to the original dataset.
  These fields remain subject to the [ANES Terms of Use](https://electionstudies.org/data-center/): research or
  statistical use only; no identification or investigation of individual respondents; citation of the original ANES
  data and documentation; and acknowledgment that ANES, the original collectors, and funders are not responsible for
  downstream uses or interpretations.
- The newly generated model outputs (`calls.csv`, `explanations/`), figures and tables are released under CC BY 4.0.
  The analysis code (`../analysis/`, `../prompts/`) is released under the MIT License. These licenses apply only to the
  contributors' original material and do not supersede the ANES terms above.
