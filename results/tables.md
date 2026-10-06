<!-- TABLE:models_execution -->
| Model | Identifier | Mode | Parameters |
|---|---|---|---|
| Claude Haiku 4.5 | us.anthropic.claude-haiku-4-5-20251001-v1:0 | batch | temperature 0.3; max tokens 1,500 |
| Ministral 3B | mistral.ministral-3-3b-instruct | synchronous | temperature 0.3; max tokens 8,000 |
| Qwen3 32B | qwen.qwen3-32b-v1:0 | batch | temperature 0.3; max tokens 8,000; thinking inactive |
| GPT-OSS-120B | openai.gpt-oss-120b-1:0 | batch | reasoning effort low; temperature 0.3; max tokens 8,000 |

<!-- TABLE:validity -->
| Model | A | B | C | D |
|---|---|---|---|---|
| Claude Haiku 4.5 | 100.0% | 100.0% | 100.0% | 100.0% |
| Ministral 3B | 98.8% | 99.8% | 99.8% | 99.8% |
| Qwen3 32B | 100.0% | 100.0% | 100.0% | 100.0% |
| GPT-OSS-120B | 99.5% | 99.7% | 99.5% | 99.6% |

<!-- TABLE:fidelity -->
| Model | Condition | MAE | Correlation | SD (% of human) |
|---|---|---|---|---|
| Claude Haiku 4.5 | A  No information | 22.61 [22.42, 22.81] | 0.002 [-0.007, 0.012] | 53.4 [53.3, 53.5] |
| Claude Haiku 4.5 | B  Narrative persona | 16.74 [16.60, 16.87] | 0.471 [0.464, 0.478] | 66.2 [65.8, 66.6] |
| Claude Haiku 4.5 | C  Structured fields | 16.86 [16.72, 16.99] | 0.468 [0.461, 0.475] | 67.2 [66.8, 67.6] |
| Ministral 3B | A  No information | 26.59 [26.38, 26.81] | 0.007 [-0.004, 0.017] | 54.7 [54.3, 55.1] |
| Ministral 3B | B  Narrative persona | 21.93 [21.76, 22.10] | 0.436 [0.428, 0.444] | 104.5 [103.9, 105.1] |
| Ministral 3B | C  Structured fields | 22.32 [22.15, 22.50] | 0.439 [0.431, 0.446] | 103.8 [103.3, 104.4] |
| Qwen3 32B | A  No information | 24.72 [24.50, 24.91] | 0.003 [-0.007, 0.015] | 69.6 [69.4, 69.8] |
| Qwen3 32B | B  Narrative persona | 17.75 [17.61, 17.89] | 0.425 [0.417, 0.433] | 76.0 [75.6, 76.4] |
| Qwen3 32B | C  Structured fields | 18.05 [17.90, 18.19] | 0.424 [0.416, 0.432] | 75.0 [74.7, 75.4] |
| GPT-OSS-120B | A  No information | 25.52 [25.29, 25.77] | -0.004 [-0.013, 0.005] | 77.1 [77.0, 77.2] |
| GPT-OSS-120B | B  Narrative persona | 18.45 [18.30, 18.60] | 0.438 [0.431, 0.446] | 86.3 [85.8, 86.8] |
| GPT-OSS-120B | C  Structured fields | 19.16 [19.01, 19.31] | 0.434 [0.426, 0.441] | 89.3 [88.8, 89.8] |

<!-- TABLE:contrasts -->
| Contrast | Model | ΔMAE | ΔCorrelation | ΔSD (pp) |
|---|---|---|---|---|
| A − B | Claude Haiku 4.5 | 5.87 [5.71, 6.04] | -0.469 [-0.481, -0.457] | -12.8 [-13.2, -12.4] |
| A − B | Ministral 3B | 4.65 [4.40, 4.90] | -0.428 [-0.442, -0.414] | -49.8 [-50.5, -49.1] |
| A − B | Qwen3 32B | 6.97 [6.77, 7.16] | -0.422 [-0.436, -0.408] | -6.4 [-6.9, -6.0] |
| A − B | GPT-OSS-120B | 7.08 [6.87, 7.31] | -0.441 [-0.453, -0.430] | -9.2 [-9.7, -8.7] |
| A − B | Mean across models | 6.14 [6.01, 6.29] | -0.440 [-0.448, -0.432] | -19.6 [-20.0, -19.1] |
| A − C | Claude Haiku 4.5 | 5.75 [5.58, 5.93] | -0.466 [-0.478, -0.454] | -13.8 [-14.2, -13.4] |
| A − C | Ministral 3B | 4.26 [4.01, 4.50] | -0.431 [-0.445, -0.417] | -49.1 [-49.8, -48.4] |
| A − C | Qwen3 32B | 6.67 [6.49, 6.85] | -0.421 [-0.434, -0.408] | -5.5 [-5.9, -5.0] |
| A − C | GPT-OSS-120B | 6.34 [6.13, 6.55] | -0.437 [-0.448, -0.425] | -12.2 [-12.7, -11.7] |
| A − C | Mean across models | 5.75 [5.62, 5.89] | -0.439 [-0.447, -0.430] | -20.1 [-20.6, -19.7] |
| B − C | Claude Haiku 4.5 | -0.12 [-0.15, -0.08] | 0.003 [-0.001, 0.007] | -1.0 [-1.1, -0.9] |
| B − C | Ministral 3B | -0.40 [-0.50, -0.29] | -0.002 [-0.007, 0.002] | 0.6 [0.3, 0.9] |
| B − C | Qwen3 32B | -0.30 [-0.38, -0.21] | 0.001 [-0.007, 0.009] | 0.9 [0.7, 1.2] |
| B − C | GPT-OSS-120B | -0.71 [-0.79, -0.65] | 0.004 [0.001, 0.009] | -3.0 [-3.3, -2.8] |
| B − C | Mean across models | -0.38 [-0.42, -0.34] | 0.001 [-0.001, 0.004] | -0.6 [-0.7, -0.5] |

<!-- TABLE:agreement -->
| Model | Pair | Identical | Within 5 pts | Within 10 pts | Mean absolute difference | Per-thermometer r | Within-respondent r (mean) |
|---|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | B–C | 38.9% [38.4%, 39.3%] | 86.7% | 98.6% | 2.88 [2.85, 2.91] | 0.897 | 0.973 |
| Claude Haiku 4.5 | A–B | 10.0% [9.7%, 10.3%] | 37.5% | 58.4% | 13.92 [13.73, 14.13] | 0.001 | 0.312 |
| Ministral 3B | B–C | 27.8% [27.4%, 28.2%] | 52.6% | 75.8% | 8.74 [8.65, 8.82] | 0.853 | 0.863 |
| Ministral 3B | A–B | 6.8% [6.6%, 7.1%] | 17.2% | 28.9% | 25.98 [25.74, 26.22] | 0.002 | 0.166 |
| Qwen3 32B | B–C | 32.4% [32.0%, 32.9%] | 65.5% | 84.9% | 6.53 [6.44, 6.60] | 0.712 | 0.907 |
| Qwen3 32B | A–B | 11.3% [11.0%, 11.6%] | 30.0% | 46.3% | 18.27 [18.06, 18.50] | 0.004 | 0.360 |
| GPT-OSS-120B | B–C | 37.1% [36.6%, 37.5%] | 68.5% | 87.5% | 5.65 [5.59, 5.72] | 0.888 | 0.927 |
| GPT-OSS-120B | A–B | 11.5% [11.1%, 11.8%] | 29.5% | 45.4% | 22.09 [21.74, 22.45] | -0.013 | 0.184 |

<!-- TABLE:reliability -->
| Model | Condition | Identical across draws | Mean absolute difference across draws | Entropy (bits) | ICC(1) |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | B  Narrative persona | 59.6% [58.0%, 61.5%] | 1.64 [1.56, 1.72] | 0.74 | 0.93 |
| Claude Haiku 4.5 | C  Structured fields | 60.9% [59.2%, 62.6%] | 1.74 [1.65, 1.83] | 0.71 | 0.91 |
| Ministral 3B | B  Narrative persona | 39.9% [38.5%, 41.3%] | 6.23 [6.02, 6.43] | 1.17 | 0.90 |
| Ministral 3B | C  Structured fields | 36.8% [35.5%, 38.2%] | 7.18 [6.96, 7.39] | 1.24 | 0.87 |
| Qwen3 32B | B  Narrative persona | 38.8% [37.3%, 40.2%] | 5.69 [5.44, 5.96] | 1.17 | 0.76 |
| Qwen3 32B | C  Structured fields | 39.8% [38.5%, 41.2%] | 5.47 [5.24, 5.71] | 1.15 | 0.74 |
| GPT-OSS-120B | B  Narrative persona | 47.8% [46.0%, 49.5%] | 4.18 [4.00, 4.35] | 0.99 | 0.91 |
| GPT-OSS-120B | C  Structured fields | 46.9% [45.1%, 48.8%] | 4.35 [4.16, 4.54] | 1.01 | 0.92 |

<!-- TABLE:cross_vs_noise -->
| Model | Pair | Between-condition mean absolute difference | Same-prompt mean absolute difference (noise) | Excess over noise | Ratio | Excess JSD (bits) |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | B–C | 2.92 | 1.69 | 1.23 [1.12, 1.34] | 1.72 | 0.023 [0.021, 0.029] |
| Claude Haiku 4.5 | A–B | 14.71 | 1.81 | 12.89 [11.85, 13.95] | 8.12 | 0.405 [0.373, 0.439] |
| Ministral 3B | B–C | 8.57 | 6.70 | 1.87 [1.64, 2.11] | 1.28 | 0.028 [0.025, 0.034] |
| Ministral 3B | A–B | 28.29 | 10.07 | 18.22 [17.31, 19.08] | 2.81 | 0.376 [0.350, 0.404] |
| Qwen3 32B | B–C | 6.28 | 5.58 | 0.70 [0.56, 0.84] | 1.13 | 0.015 [0.012, 0.019] |
| Qwen3 32B | A–B | 19.44 | 7.61 | 11.82 [10.69, 12.94] | 2.55 | 0.276 [0.250, 0.306] |
| GPT-OSS-120B | B–C | 5.48 | 4.26 | 1.22 [1.04, 1.43] | 1.29 | 0.020 [0.017, 0.025] |
| GPT-OSS-120B | A–B | 24.28 | 4.00 | 20.28 [18.11, 22.39] | 6.07 | 0.447 [0.413, 0.484] |

<!-- TABLE:order -->
| Model | Condition | Mean absolute reversed–original difference | Same-prompt mean absolute difference | Order/noise ratio | Signed mean (rev − orig) | Position slope (diagnostic) |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | A  No information | 10.12 [9.77, 10.48] | 2.00 | 5.07 [4.80, 5.34] | 3.31 | 0.11 |
| Claude Haiku 4.5 | B  Narrative persona | 3.75 [3.63, 3.87] | 1.63 | 2.31 [2.18, 2.47] | 0.47 | -0.08 |
| Claude Haiku 4.5 | C  Structured fields | 3.63 [3.49, 3.77] | 1.74 | 2.09 [1.96, 2.23] | 1.12 | -0.11 |
| Claude Haiku 4.5 | D  Partisan mirror | 3.79 [3.65, 3.93] | 1.67 | 2.27 [2.13, 2.46] | 1.12 | -0.10 |
| Ministral 3B | A  No information | 18.19 [17.59, 18.77] | 13.90 | 1.31 [1.25, 1.36] | 7.76 | -0.21 |
| Ministral 3B | B  Narrative persona | 9.54 [9.14, 9.94] | 6.17 | 1.55 [1.47, 1.62] | 2.28 | -0.34 |
| Ministral 3B | C  Structured fields | 10.89 [10.42, 11.34] | 7.07 | 1.54 [1.47, 1.61] | 5.52 | -0.80 |
| Ministral 3B | D  Partisan mirror | 9.78 [9.39, 10.15] | 6.49 | 1.51 [1.45, 1.56] | 2.22 | -0.33 |
| Qwen3 32B | A  No information | 16.48 [15.68, 17.29] | 9.25 | 1.78 [1.67, 1.91] | 2.36 | 0.59 |
| Qwen3 32B | B  Narrative persona | 8.13 [7.74, 8.53] | 5.74 | 1.42 [1.34, 1.50] | 1.80 | 0.20 |
| Qwen3 32B | C  Structured fields | 8.13 [7.72, 8.58] | 5.42 | 1.50 [1.41, 1.60] | 1.69 | 0.29 |
| Qwen3 32B | D  Partisan mirror | 7.92 [7.51, 8.36] | 5.72 | 1.38 [1.30, 1.47] | 2.33 | 0.19 |
| GPT-OSS-120B | A  No information | 12.05 [11.68, 12.37] | 3.86 | 3.12 [2.97, 3.28] | 10.37 | 0.56 |
| GPT-OSS-120B | B  Narrative persona | 7.11 [6.85, 7.39] | 4.23 | 1.68 [1.59, 1.78] | 3.94 | -0.17 |
| GPT-OSS-120B | C  Structured fields | 7.80 [7.49, 8.11] | 4.35 | 1.79 [1.69, 1.89] | 4.72 | -0.24 |
| GPT-OSS-120B | D  Partisan mirror | 7.20 [6.93, 7.47] | 4.29 | 1.68 [1.59, 1.76] | 4.30 | -0.15 |

<!-- TABLE:mirror -->
| Model | Stratum | n | Aligned shift in Dem−Rep gap | Moves in expected direction | Mean absolute spillover (other 9 groups) |
|---|---|---|---|---|---|
| Claude Haiku 4.5 | All eligible (5,115) | 5,115 | 102.1 [101.2, 103.1] | 97.8% | 21.9 |
| Claude Haiku 4.5 | Non-moderates (party and ideology change) | 4,165 | 109.2 [108.3, 110.3] | 97.3% | 24.1 |
| Claude Haiku 4.5 | Moderates (party only) | 950 | 71.0 [70.4, 71.6] | 100.0% | 12.4 |
| Claude Haiku 4.5 | Democrat → Republican | 2,572 | 96.6 [95.4, 98.1] | 97.0% | 20.4 |
| Claude Haiku 4.5 | Republican → Democrat | 2,543 | 107.7 [106.5, 108.9] | 98.7% | 23.4 |
| Claude Haiku 4.5 | Excluding cross-pressured profiles | 4,860 | 107.2 [106.5, 107.8] | 100.0% | 22.6 |
| Ministral 3B | All eligible (5,115) | 5,115 | 144.3 [143.2, 145.4] | 99.1% | 44.9 |
| Ministral 3B | Non-moderates (party and ideology change) | 4,165 | 151.2 [149.9, 152.3] | 98.9% | 48.9 |
| Ministral 3B | Moderates (party only) | 950 | 114.2 [113.1, 115.4] | 100.0% | 27.7 |
| Ministral 3B | Democrat → Republican | 2,572 | 138.5 [136.8, 140.1] | 98.8% | 42.6 |
| Ministral 3B | Republican → Democrat | 2,543 | 150.2 [148.9, 151.5] | 99.4% | 47.3 |
| Ministral 3B | Excluding cross-pressured profiles | 4,860 | 150.2 [149.4, 150.9] | 100.0% | 46.2 |
| Qwen3 32B | All eligible (5,115) | 5,115 | 107.8 [106.9, 108.7] | 99.2% | 24.7 |
| Qwen3 32B | Non-moderates (party and ideology change) | 4,165 | 115.6 [114.6, 116.6] | 99.1% | 26.7 |
| Qwen3 32B | Moderates (party only) | 950 | 73.6 [72.8, 74.3] | 100.0% | 15.8 |
| Qwen3 32B | Democrat → Republican | 2,572 | 102.7 [101.4, 104.0] | 99.0% | 23.4 |
| Qwen3 32B | Republican → Democrat | 2,543 | 113.0 [111.7, 114.3] | 99.4% | 26.0 |
| Qwen3 32B | Excluding cross-pressured profiles | 4,860 | 111.7 [110.9, 112.5] | 100.0% | 25.3 |
| GPT-OSS-120B | All eligible (5,115) | 5,115 | 117.7 [116.6, 118.7] | 98.1% | 37.0 |
| GPT-OSS-120B | Non-moderates (party and ideology change) | 4,165 | 125.6 [124.5, 126.7] | 97.7% | 39.8 |
| GPT-OSS-120B | Moderates (party only) | 950 | 82.9 [81.9, 84.0] | 99.8% | 24.8 |
| GPT-OSS-120B | Democrat → Republican | 2,572 | 111.4 [109.9, 113.0] | 97.3% | 35.0 |
| GPT-OSS-120B | Republican → Democrat | 2,543 | 124.0 [122.6, 125.2] | 99.0% | 39.1 |
| GPT-OSS-120B | Excluding cross-pressured profiles | 4,860 | 122.7 [121.9, 123.4] | 100.0% | 38.0 |

<!-- TABLE:exaggeration -->
| Model | Exaggeration, 11 groups | Exaggeration, 2 party groups | Pattern correlation with human contrast |
|---|---|---|---|
| Claude Haiku 4.5 | 1.02 [1.00, 1.04] | 0.92 [0.91, 0.94] | 0.995 |
| Ministral 3B | 1.85 [1.82, 1.89] | 1.31 [1.29, 1.33] | 0.950 |
| Qwen3 32B | 1.10 [1.08, 1.12] | 0.98 [0.96, 0.99] | 0.985 |
| GPT-OSS-120B | 1.54 [1.51, 1.57] | 1.07 [1.05, 1.08] | 0.950 |

<!-- TABLE:rq4 -->
| Model | Condition | % coefficients differing from ANES | % of those with opposite sign |
|---|---|---|---|
| Claude Haiku 4.5 | A  No information | 43.4 [39.7, 47.1] | 41.9 [34.3, 49.5] |
| Claude Haiku 4.5 | B  Narrative persona | 34.5 [29.6, 39.5] | 38.2 [28.1, 48.2] |
| Claude Haiku 4.5 | C  Structured fields | 34.1 [29.6, 38.6] | 36.0 [26.8, 45.2] |
| Ministral 3B | A  No information | 38.4 [34.6, 42.2] | 58.0 [49.7, 66.3] |
| Ministral 3B | B  Narrative persona | 45.0 [40.6, 49.4] | 35.9 [28.5, 43.3] |
| Ministral 3B | C  Structured fields | 41.8 [38.0, 45.6] | 34.8 [28.3, 41.3] |
| Qwen3 32B | A  No information | 43.4 [39.4, 47.5] | 56.5 [47.8, 65.3] |
| Qwen3 32B | B  Narrative persona | 37.0 [32.4, 41.7] | 37.4 [28.2, 46.7] |
| Qwen3 32B | C  Structured fields | 32.7 [28.4, 37.0] | 29.2 [20.6, 37.8] |
| GPT-OSS-120B | A  No information | 43.0 [39.3, 46.7] | 56.1 [47.9, 64.3] |
| GPT-OSS-120B | B  Narrative persona | 41.8 [37.1, 46.5] | 36.4 [28.2, 44.6] |
| GPT-OSS-120B | C  Structured fields | 42.7 [38.3, 47.1] | 36.7 [29.0, 44.4] |
