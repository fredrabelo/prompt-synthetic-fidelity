################################################################################
##
## Purpose: RQ3 conditional/subgroup fidelity — reproduces the Bisbee et al.
##          (2024) MS_Figure3.R design (per-group, per-year regression with a
##          Data(human/synthetic) x covariate interaction, clustered SE by
##          respondent), run separately for each of our 4 grounding
##          conditions (A/B/C/D), to compute the "% coefficients
##          significantly different from human" / "% of those sign-flipped"
##          headline statistic for each condition.
##
## Adapted from: docs/papers/bisbee/PA_replication/code/MS_Figure3.R
## Input: ../replication_data/{respondents,human_benchmark,synthetic_responses}.csv
## Output: prints the summary table; writes conditional_fidelity_results.csv
##
################################################################################

.libPaths(c("~/R/library", .libPaths()))
suppressMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(forcats)
  library(fixest)
})

here <- dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)))
if (length(here) == 0 || here == "") here <- "."
data_dir <- file.path(here, "..", "data")

respondents <- read_csv(file.path(data_dir, "respondents.csv"), show_col_types = FALSE)
human <- read_csv(file.path(data_dir, "human_benchmark.csv"), show_col_types = FALSE)
synthetic <- read_csv(file.path(data_dir, "synthetic_responses.csv"), show_col_types = FALSE)

# ---- Recode attributes exactly as MS_Figure3.R does (lines 62-76) ---------
attrs <- respondents %>%
  transmute(
    external_key,
    year,
    income = scale(as.numeric(factor(income,
      levels = c("$30,000", "$50,000", "$80,000", "$100,000", "more than $150,000"))))[, 1],
    age = scale(age)[, 1],
    ideo3 = factor(
      ifelse(grepl("conser", ideo), "Conservative", ifelse(grepl("liber", ideo), "Liberal", "Moderate")),
      levels = c("Liberal", "Moderate", "Conservative")
    ),
    education = factor(
      ifelse(grepl("bachel", education), "BA or more", ifelse(grepl("coll", education), "Some college", "No college")),
      levels = c("No college", "Some college", "BA or more")
    ),
    interest = factor(interest, levels = c("never", "sometimes", "frequently", "regularly", "always")),
    marst = factor(marst, levels = c("married", "single", "divorced", "widowed", "separated")),
    raceth = factor(gsub("non-Hispanic ", "", raceth), levels = c("white", "black", "Hispanic")),
    gender = factor(gender),
    pid = factor(pid),
    regis = factor(regis)
  )

conditions <- unique(synthetic$condition)
groups <- unique(human$thermometer)

run_one <- function(cond, grp, yr) {
  human_g <- human %>% filter(thermometer == grp) %>% select(external_key, ANES = human_value)
  synth_g <- synthetic %>% filter(condition == cond, thermometer == grp) %>%
    select(external_key, LLM = synthetic_value)
  toanal3 <- attrs %>% filter(year == yr) %>%
    inner_join(human_g, by = "external_key") %>%
    inner_join(synth_g, by = "external_key") %>%
    pivot_longer(cols = c(ANES, LLM), names_to = "Data", values_to = "FT") %>%
    mutate(Data = factor(Data, levels = c("ANES", "LLM")), FT = as.numeric(FT))

  if (nrow(toanal3) < 20) return(NULL)

  m <- tryCatch(
    feols(FT ~ raceth * Data + age * Data + gender * Data + ideo3 * Data +
      pid * Data + income * Data + education * Data + regis * Data +
      interest * Data + marst * Data,
    toanal3,
    cluster = "external_key"
    ),
    error = function(e) NULL
  )
  if (is.null(m)) return(NULL)

  ct <- as.data.frame(m$coeftable)
  ct$term <- rownames(ct)
  names(ct) <- c("est", "se", "tstat", "pval", "term")

  interactions <- ct %>% filter(grepl(":", term)) %>%
    mutate(base_term = gsub("DataLLM:|:DataLLM", "", term))
  mains <- ct %>% filter(!grepl(":", term)) %>% select(base_term = term, main_est = est)

  interactions %>%
    left_join(mains, by = "base_term") %>%
    mutate(
      condition = cond, group = grp, year = yr,
      llm_est = main_est + est,
      sig = pval < 0.05,
      sign_flip = sig & (sign(main_est) != sign(llm_est)) & main_est != 0
    ) %>%
    select(condition, group, year, term = base_term, human_coef = main_est, llm_coef = llm_est,
      interaction_est = est, pval, sig, sign_flip)
}

results <- list()
i <- 1
for (cond in conditions) {
  for (grp in groups) {
    for (yr in c(2016, 2020)) {
      r <- run_one(cond, grp, yr)
      if (!is.null(r)) { results[[i]] <- r; i <- i + 1 }
    }
  }
}
all_results <- bind_rows(results)

write_csv(all_results, file.path(here, "conditional_fidelity_results.csv"))

summary_tab <- all_results %>%
  group_by(condition) %>%
  summarise(
    n_coefficients = n(),
    n_significant = sum(sig),
    pct_significant = round(100 * mean(sig), 1),
    pct_sign_flipped_of_significant = round(100 * sum(sign_flip) / pmax(sum(sig), 1), 1)
  )

cat("\n=== Conditional/subgroup fidelity (Bisbee-style), per condition ===\n")
print(summary_tab)
cat("\nBisbee et al. (2024) benchmark (GPT-3.5, persona condition): 48% significantly different, 32% of those sign-flipped.\n")
