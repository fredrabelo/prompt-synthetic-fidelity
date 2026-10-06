################################################################################
## Conditional fidelity: replicates the design of MS_Figure3.R in Bisbee et al. (2024). For each group and year, a regression with
## Data (human/synthetic) x covariate interactions and respondent-clustered SEs, run per MODEL and condition (A, B, C; D has no
## matched human counterpart). Statistics: % of coefficients significantly different from the human ones and % of the significant
## ones with a sign flip. Uncertainty: respondent-level bootstrap (external_key is the cluster; draw_id only pairs attributes with
## outcomes) and a Wald interval (point +/- 1.96 x bootstrap SE), because a statistic built on a p < .05 threshold is discontinuous.
## Points are computed here from the data.
## Input : <long_dir>/synthetic_long_<model>.csv (export_long.py) and ../data/{respondents,human_benchmark}.csv
## Output: <out_dir>/conditional_fidelity_{points,bootstrap,summary}.csv
## Usage : Rscript a7_conditional.R <long_dir> <out_dir> [n_boot=200] [models comma-sep]
################################################################################
suppressMessages({library(dplyr); library(tidyr); library(readr); library(fixest)})
# fixest uses OpenMP; forking (mclapply) after OpenMP threads exist in the parent can DEADLOCK. One thread per regression; parallelism comes from the workers.
setFixest_nthreads(1); Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1")
args <- commandArgs(trailingOnly = TRUE)
long_dir <- args[1]; out_dir <- args[2]
n_boot <- if (length(args) >= 3) as.integer(args[3]) else 200
models <- if (length(args) >= 4) strsplit(args[4], ",")[[1]] else sub("synthetic_long_(.*)\\.csv", "\\1", list.files(long_dir, "synthetic_long_.*\\.csv"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(42)
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
data_dir <- file.path(here, "..", "data")
respondents <- read_csv(file.path(data_dir, "respondents.csv"), show_col_types = FALSE)
human <- read_csv(file.path(data_dir, "human_benchmark.csv"), show_col_types = FALSE)

attrs_base <- respondents %>% transmute(
  external_key, year,
  income = scale(as.numeric(factor(income, levels = c("$30,000", "$50,000", "$80,000", "$100,000", "more than $150,000"))))[, 1],
  age = scale(age)[, 1],
  ideo3 = factor(ifelse(grepl("conser", ideo), "Conservative", ifelse(grepl("liber", ideo), "Liberal", "Moderate")), levels = c("Liberal", "Moderate", "Conservative")),
  education = factor(ifelse(grepl("bachel", education), "BA or more", ifelse(grepl("coll", education), "Some college", "No college")), levels = c("No college", "Some college", "BA or more")),
  interest = factor(interest, levels = c("never", "sometimes", "frequently", "regularly", "always")),
  marst = factor(marst, levels = c("married", "single", "divorced", "widowed", "separated")),
  raceth = factor(gsub("non-Hispanic ", "", raceth), levels = c("white", "black", "Hispanic")),
  gender = factor(gender), pid = factor(pid), regis = factor(regis))
groups <- unique(human$thermometer)
all_keys <- unique(respondents$external_key); n_resp <- length(all_keys)

run_one <- function(attrs, human_all, synth_all, cond, grp, yr) {
  hg <- human_all %>% filter(thermometer == grp) %>% select(draw_id, ANES = human_value)
  sg <- synth_all %>% filter(condition == cond, thermometer == grp) %>% select(draw_id, LLM = synthetic_value)
  d <- attrs %>% filter(year == yr) %>% inner_join(hg, by = "draw_id") %>% inner_join(sg, by = "draw_id") %>%
    pivot_longer(c(ANES, LLM), names_to = "Data", values_to = "FT") %>% mutate(Data = factor(Data, levels = c("ANES", "LLM")), FT = as.numeric(FT))
  if (nrow(d) < 20) return(NULL)
  m <- tryCatch(feols(FT ~ raceth * Data + age * Data + gender * Data + ideo3 * Data + pid * Data + income * Data + education * Data + regis * Data + interest * Data + marst * Data, d, cluster = "external_key"), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  ct <- as.data.frame(m$coeftable); ct$term <- rownames(ct); names(ct) <- c("est", "se", "tstat", "pval", "term")
  inter <- ct %>% filter(grepl(":", term)) %>% mutate(base_term = gsub("DataLLM:|:DataLLM", "", term))
  mains <- ct %>% filter(!grepl(":", term)) %>% select(base_term = term, main_est = est)
  inter %>% left_join(mains, by = "base_term") %>% mutate(sig = pval < 0.05, llm_est = main_est + est, sign_flip = sig & (sign(main_est) != sign(llm_est)) & main_est != 0) %>%
    summarise(n_sig = sum(sig), n = n(), n_flip = sum(sign_flip))
}
pipeline <- function(attrs, human_r, synth_r, conds) {
  bind_rows(lapply(conds, function(cond) {
    s <- 0L; n <- 0L; f <- 0L
    for (grp in groups) for (yr in c(2016, 2020)) { r <- run_one(attrs, human_r, synth_r, cond, grp, yr); if (!is.null(r)) { s <- s + r$n_sig; n <- n + r$n; f <- f + r$n_flip } }
    tibble(condition = cond, pct_significant = 100 * s / n, pct_sign_flipped_of_significant = 100 * f / max(s, 1))
  }))
}
pts <- list(); boots <- list()
for (mdl in models) {
  synth <- read_csv(file.path(long_dir, sprintf("synthetic_long_%s.csv", mdl)), show_col_types = FALSE)
  conds <- sort(unique(synth$condition))
  key_map0 <- tibble(external_key = all_keys, draw_id = seq_along(all_keys))
  point <- pipeline(key_map0 %>% inner_join(attrs_base, by = "external_key"), key_map0 %>% inner_join(human, by = "external_key"), key_map0 %>% inner_join(synth, by = "external_key"), conds)
  point$model <- mdl; pts[[mdl]] <- point
  cat(sprintf("[%s] point estimates done; bootstrap %d ...\n", mdl, n_boot))
  worker <- function(b) {
    km <- tibble(external_key = sample(all_keys, n_resp, replace = TRUE)); km$draw_id <- seq_len(nrow(km))
    res <- pipeline(km %>% inner_join(attrs_base, by = "external_key"), km %>% inner_join(human, by = "external_key"), km %>% inner_join(synth, by = "external_key"), conds)
    res$resample <- b; res
  }
  boots[[mdl]] <- bind_rows(parallel::mclapply(seq_len(n_boot), worker, mc.cores = min(8L, parallel::detectCores(logical = FALSE)), mc.set.seed = TRUE)) %>% mutate(model = mdl)
}
points <- bind_rows(pts); boot <- bind_rows(boots)
write_csv(points, file.path(out_dir, "conditional_fidelity_points.csv")); write_csv(boot, file.path(out_dir, "conditional_fidelity_bootstrap.csv"))
summ <- boot %>% group_by(model, condition) %>% summarise(se_sig = sd(pct_significant), se_flip = sd(pct_sign_flipped_of_significant), .groups = "drop") %>%
  left_join(points, by = c("model", "condition")) %>% transmute(model, condition, pct_significant, pct_significant_lo = pct_significant - 1.96 * se_sig, pct_significant_hi = pct_significant + 1.96 * se_sig,
    pct_flip = pct_sign_flipped_of_significant, pct_flip_lo = pct_flip - 1.96 * se_flip, pct_flip_hi = pct_flip + 1.96 * se_flip)
write_csv(summ, file.path(out_dir, "conditional_fidelity_summary.csv")); print(summ)
