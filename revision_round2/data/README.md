# Longitudinal cluster identifiers

`person_clusters.csv` maps each of the 7,526 analytic person-wave keys to one of 6,027 distinct-person clusters. It does not alter outcomes, attributes, sample selection or model calls. There are 1,499 two-wave clusters; the remaining clusters have one observed wave. The 250-record repeated-call subsample contains 247 clusters.

Source: ANES 2016–2020–2024 Panel Study Merged File, May 19, 2026 version, https://electionstudies.org/data-center/2016-2020-2024-panel-merged-study/ . Match `anes_` + integer(20160000 + V160001) to the 2016 sample key and `anes_` + integer(2020000000 + V200001) to the 2020 sample key. Retain only links present on both sides of the analytic sample; both source IDs must be unique. Use the 2016 key as the common cluster key; unmatched records retain their own keys. Demographic disagreement does not override the official identifier linkage.

The supplied mapping suffices to reproduce all analyses without downloading the larger panel file. Its hash is recorded in results/manifest.json.
