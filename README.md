# Does Prompt Structure Improve Synthetic Respondent Fidelity? — Replication Package

Data, analysis code, and figures for the paper *"Does Prompt Structure Improve
Synthetic Respondent Fidelity? A Controlled Test of Persona Design"* (submitted,
2026). The paper replicates Bisbee, Clinton, Dorff, Kenkel & Larson (2024),
*"Synthetic Replacements for Human Survey Data? The Perils of Large Language
Models,"* *Political Analysis* 32(4):401–416, and tests whether restructuring
LLM-based synthetic respondents' prompts — rather than adding new information —
recovers the heterogeneity that simple persona prompting loses.

This repository does **not** include the underlying agent/prompting platform used
to generate the synthetic responses — see [Reproducibility scope](#reproducibility-scope)
below. It includes everything needed to independently verify every number and
figure reported in the paper from the released data.

## Structure

```
data/         Respondent attributes, human benchmark, and all synthetic responses (CSV)
analysis/     Scripts that turn data/ into every statistic and figure in the paper
figures/      Output of the analysis scripts — the exact PNGs used in the paper
```

## Reproducing the paper's results

```bash
cd analysis
pip install pandas numpy matplotlib
python3 bootstrap_analysis.py          # -> bootstrap_summary.json (Table 1, Table 2, Appendix)
python3 make_figure1_headline.py       # -> ../figures/figure1_headline.png

Rscript conditional_fidelity_analysis.R  # -> conditional_fidelity_results.csv (RQ4 / Table 1)
# requires: install.packages(c("fixest","dplyr","tidyr","readr","stringr","forcats"))
python3 make_figure3_coefficients.py   # -> ../figures/figure3_coefficients.png

python3 make_figure2_distributions.py  # -> ../figures/figure2_distributions.png
```

Each script is self-contained and reads only from `../data/`. No database, API
key, or external service is required — this is exactly the pipeline used to
produce every number in the paper.

## Reproducibility scope

The synthetic responses in `data/synthetic_responses.csv` were generated using an
internal, not-publicly-released research platform, calling `claude-haiku-4-5` via
the Anthropic Message Batches API. We do not release that platform's source code.
The paper's Appendix reproduces the complete, literal prompt text (system prompt,
all four condition templates, question template, and output schema) used to
generate every response, which — combined with the data in this repository — is
sufficient to (a) verify every statistic and figure in the paper, and (b) regenerate
equivalent synthetic responses against the same model using only the Appendix's
prompts and any standard LLM API client.

## License

See `data/README.md` for the applicable licenses (they differ by file: ANES-derived
fields carry ANES's own terms of use; newly generated data, figures, and analysis
code are CC BY 4.0 / MIT respectively).

## Citation

If you use this data or code, please cite both this paper (details to be added on
acceptance) and the original study:

```bibtex
@article{bisbee2024synthetic,
  title={Synthetic Replacements for Human Survey Data? The Perils of Large Language Models},
  author={Bisbee, James and Clinton, Joshua D and Dorff, Cassy and Kenkel, Brenton and Larson, Jennifer M},
  journal={Political Analysis},
  volume={32}, number={4}, pages={401--416}, year={2024},
  doi={10.1017/pan.2024.5}
}
```
