# Data

## Files

- **`respondents.csv`** — one row per ANES respondent (N=7,526). Columns:
  `external_key, year, age, raceth, gender, marst, education, income, ideo, regis,
  pid, interest`. These are the ten attributes used to construct each synthetic
  persona, recoded from the ANES 2016/2020 cumulative data file exactly as
  described in the paper's Methodology section.
- **`human_benchmark.csv`** — long format: `external_key, thermometer, human_value`.
  The real ANES feeling-thermometer response (0–100) for each respondent × group,
  where available (not every respondent answered every thermometer).
- **`synthetic_responses.csv`** — long format: `condition, external_key,
  thermometer, draw_index, synthetic_value, explanation`. `condition` is one of
  `A_generic_llm`, `B_persona_demographic`, `C_population_grounded`,
  `D_evidence_grounded`, matching conditions A–D in the paper. `explanation` is
  the model's one-sentence stated reasoning for that response.

## Source

Ground truth: ANES Time Series Cumulative Data File (American National Election
Studies, University of Michigan and Stanford University). Original study design:
Bisbee, Clinton, Dorff, Kenkel & Larson (2024), "Synthetic Replacements for Human
Survey Data? The Perils of Large Language Models," *Political Analysis* 32(4):401–416
— please cite this paper and the original Dataverse release (DOI
`10.7910/DVN/VPN481`) as a matter of scientific attribution.

## Licensing and terms of use

- The Bisbee et al. replication package, DOI `10.7910/DVN/VPN481`, is released
  under the **CC0 1.0 Public Domain Dedication**.
- The ANES-derived respondent attributes and human benchmark values come only from
  ANES public-release data. ANES permits dissemination of derived public-release
  datasets and asks redistributors to link to the original dataset. These fields
  remain subject to the [ANES Terms of Use](https://electionstudies.org/data-center/):
  research or statistical use only; no identification or investigation of individual
  respondents; citation of the original ANES data and documentation; and acknowledgment
  that ANES, the original collectors, and funders are not responsible for downstream
  uses or interpretations.
- The newly generated synthetic responses and original figures/tables are released
  under **CC BY 4.0**. Original analysis code (`../analysis/`) is released under the
  **MIT License**. These licenses apply only to the contributors' original material
  and do not supersede the ANES terms above.
