## Descriptives for the analysis sample (complete cases) compared with the excluded participants,
## participant flow counts, and a check of the -1 codes used for missing values.
## Complements descriptive_statistics.R, which describes all rows of catss_final_data.csv.
## Date: 2026-10-23
## Project: Deep Cross Modal Fusion Model for predicting schizophrenia
##
## Usage: run from the folder containing catss_final_data.csv.
##   Rscript analysis_sample_descriptives.R
## Output (aggregated only; counts below 5 are shown as "<5"):
##   analysis_sample_flow.csv            : rows and twin pairs remaining after each exclusion step
##   table1_included_vs_excluded.csv     : sex, outcomes, SUD15 items and modality summaries per group, with SMD
##   missing_code_check.csv              : -1 codes and other negative values in the questionnaire items
##   analysis_sample_twin_structure.csv  : twin pairs with both twins vs one twin in the analysis sample

library(readr)
library(dplyr)

SMALL_CELL <- 5

catss <- read_csv("catss_final_data.csv", show_col_types = FALSE)
catss$cmpair <- as.character(catss$cmpair)
included <- complete.cases(catss)
cat(sprintf("Total rows: %d | complete cases: %d | excluded: %d\n",
            nrow(catss), sum(included), sum(!included)))

MODALITIES <- c(
  "SUD15" = "^SUD15", "PRS" = "^PRS", "SCZ15" = "^SCZ15", "ADHD9" = "^ADHD9", "ASD9" = "^ASD9",
  "ACE15" = "^ACE15", "ACE18" = "^ACE18", "SUD18" = "^SUD18", "SES" = "^SES", "SEX" = "^SEX",
  "batch*PC" = "^batch_.*_x_PC"
)
OUTCOMES <- c("SCZ18_Pos_Norm", "SCZ18_Neg_Norm")
modality_cols <- lapply(MODALITIES, function(p) grep(p, names(catss), value = TRUE))

# Counts of 1 to 4 are suppressed; zero is shown as 0
fmt_count <- function(n) ifelse(n > 0 & n < SMALL_CELL, sprintf("<%d", SMALL_CELL), as.character(n))
fmt_n_pct <- function(n, total) {
  ifelse(n > 0 & n < SMALL_CELL, sprintf("<%d", SMALL_CELL), sprintf("%d (%.1f%%)", n, 100 * n / total))
}
fmt_mean_sd <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < SMALL_CELL) return(sprintf("n<%d", SMALL_CELL))
  sprintf("%.3f (%.3f)", mean(x), sd(x))
}
fmt_median_iqr <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < SMALL_CELL) return(sprintf("n<%d", SMALL_CELL))
  q <- quantile(x, c(0.25, 0.5, 0.75))
  sprintf("%.3f [%.3f, %.3f]", q[2], q[1], q[3])
}
# Standardised mean difference (included vs excluded), pooled SD
smd <- function(x_in, x_out) {
  x_in <- x_in[!is.na(x_in)]
  x_out <- x_out[!is.na(x_out)]
  if (length(x_in) < 2 || length(x_out) < 2) return(NA)
  pooled <- sqrt((var(x_in) + var(x_out)) / 2)
  if (pooled == 0) return(NA)
  (mean(x_in) - mean(x_out)) / pooled
}


# ══════════════════════════════════════════
# 1. PARTICIPANT FLOW
# ══════════════════════════════════════════
# Sequential exclusion: outcomes first, then genotype-based inputs (PRS, batch*PC), then the questionnaires.

flow_steps <- list(
  "Both outcomes observed (SCZ18)" = OUTCOMES,
  "PRS available (genotyped)" = modality_cols[["PRS"]],
  "Batch x PC available" = modality_cols[["batch*PC"]],
  "SEX available" = modality_cols[["SEX"]],
  "SUD15 complete" = modality_cols[["SUD15"]],
  "SCZ15 complete" = modality_cols[["SCZ15"]],
  "ADHD9 complete" = modality_cols[["ADHD9"]],
  "ASD9 complete" = modality_cols[["ASD9"]],
  "ACE15 complete" = modality_cols[["ACE15"]],
  "ACE18 complete" = modality_cols[["ACE18"]],
  "SUD18 complete" = modality_cols[["SUD18"]],
  "SES complete" = modality_cols[["SES"]]
)

keep <- rep(TRUE, nrow(catss))
flow <- data.frame(Step = "All rows in catss_final_data.csv", Rows = nrow(catss),
                   Twin_pairs = length(unique(catss$cmpair)), Excluded_at_step = NA)
for (step in names(flow_steps)) {
  cols <- flow_steps[[step]]
  step_ok <- if (length(cols) > 0) complete.cases(catss[, cols]) else rep(TRUE, nrow(catss))
  excluded_now <- sum(keep & !step_ok)
  keep <- keep & step_ok
  flow <- rbind(flow, data.frame(Step = step, Rows = sum(keep),
                                 Twin_pairs = length(unique(catss$cmpair[keep])),
                                 Excluded_at_step = excluded_now))
}
# Any remaining columns (e.g. identifiers) with missing values
other_ok <- complete.cases(catss)
flow <- rbind(flow, data.frame(Step = "Complete cases (analysis sample)", Rows = sum(other_ok),
                               Twin_pairs = length(unique(catss$cmpair[other_ok])),
                               Excluded_at_step = sum(keep & !other_ok)))
if (sum(keep & !other_ok) > 0) {
  cat("Note: rows excluded at the last step have missing values only in columns outside the modalities.\n")
}
flow_out <- flow %>% mutate(Excluded_at_step = ifelse(is.na(Excluded_at_step), "", fmt_count(Excluded_at_step)))
write_csv(flow_out, "analysis_sample_flow.csv")
print(flow_out)
cat("Saved: analysis_sample_flow.csv\n")

# Twin structure of the analysis sample
twins_per_pair <- table(catss$cmpair[included])
twin_structure <- data.frame(
  Twin_pairs = length(twins_per_pair),
  Pairs_with_both_twins = fmt_count(sum(twins_per_pair == 2)),
  Pairs_with_one_twin = fmt_count(sum(twins_per_pair == 1)),
  Pairs_with_more_than_two_rows = fmt_count(sum(twins_per_pair > 2))
)
write_csv(twin_structure, "analysis_sample_twin_structure.csv")
print(twin_structure)
cat("Saved: analysis_sample_twin_structure.csv\n")


# ══════════════════════════════════════════
# 2. TABLE 1: INCLUDED vs EXCLUDED
# ══════════════════════════════════════════

groups <- list(Included = catss[included, ], Excluded = catss[!included, ], All = catss)
rows <- list()
add_row <- function(variable, statistic, values, smd_value = NA) {
  rows[[length(rows) + 1]] <<- data.frame(
    Variable = variable, Statistic = statistic,
    Included = values[["Included"]], Excluded = values[["Excluded"]], All = values[["All"]],
    SMD = ifelse(is.na(smd_value), "", sprintf("%.3f", smd_value)),
    stringsAsFactors = FALSE
  )
}

add_row("N", "rows", sapply(groups, function(g) as.character(nrow(g))))
add_row("Twin pairs", "n", sapply(groups, function(g) as.character(length(unique(g$cmpair)))))

# Sex (coded 1 = male, 2 = female, as in descriptive_statistics.R)
sex_col <- modality_cols[["SEX"]][1]
sex_smd <- smd(as.numeric(groups$Included[[sex_col]] == 2), as.numeric(groups$Excluded[[sex_col]] == 2))
for (code in c(1, 2)) {
  label <- ifelse(code == 1, "Sex: male", "Sex: female")
  add_row(label, "n (%)", sapply(groups, function(g) fmt_n_pct(sum(g[[sex_col]] == code, na.rm = TRUE), nrow(g))),
          if (code == 2) sex_smd else NA)
}
add_row("Sex: missing", "n (%)", sapply(groups, function(g) fmt_n_pct(sum(is.na(g[[sex_col]])), nrow(g))))

# Outcomes
for (outcome in OUTCOMES) {
  s <- smd(groups$Included[[outcome]], groups$Excluded[[outcome]])
  add_row(outcome, "mean (SD)", sapply(groups, function(g) fmt_mean_sd(g[[outcome]])), s)
  add_row(outcome, "median [IQR]", sapply(groups, function(g) fmt_median_iqr(g[[outcome]])))
  add_row(outcome, "missing n (%)", sapply(groups, function(g) fmt_n_pct(sum(is.na(g[[outcome]])), nrow(g))))
}

# SUD15 items (the anchor exposure), reported item by item
for (item in modality_cols[["SUD15"]]) {
  values <- sort(unique(na.omit(catss[[item]])))
  s <- smd(groups$Included[[item]], groups$Excluded[[item]])
  if (all(values %in% c(0, 1))) {
    add_row(item, "any use, n (%)", sapply(groups, function(g) fmt_n_pct(sum(g[[item]] == 1, na.rm = TRUE), sum(!is.na(g[[item]])))), s)
  } else {
    add_row(item, "mean (SD)", sapply(groups, function(g) fmt_mean_sd(g[[item]])), s)
  }
}

# Other modalities: mean of the items per participant (available items only), and missingness
for (m in setdiff(names(MODALITIES), c("SUD15", "SEX"))) {
  cols <- modality_cols[[m]]
  if (length(cols) == 0) next
  score <- function(g) {
    s <- rowMeans(as.matrix(g[, cols]), na.rm = TRUE)
    s[is.nan(s)] <- NA
    s
  }
  add_row(sprintf("%s (%d items): item mean", m, length(cols)), "mean (SD)",
          sapply(groups, function(g) fmt_mean_sd(score(g))), smd(score(groups$Included), score(groups$Excluded)))
  add_row(sprintf("%s: any item missing", m), "n (%)",
          sapply(groups, function(g) fmt_n_pct(sum(!complete.cases(g[, cols])), nrow(g))))
}

table1 <- bind_rows(rows)
write_csv(table1, "table1_included_vs_excluded.csv")
cat("Saved: table1_included_vs_excluded.csv\n")
cat("SMD: standardised mean difference, included minus excluded (|SMD| > 0.1 is commonly read as imbalance).\n")


# ══════════════════════════════════════════
# 3. MISSING-VALUE CODES
# ══════════════════════════════════════════
# feature_selection.py recodes NaN to -1 in other_bullying15 and other_abuse18, so these rows are kept by
# complete.cases()/dropna(). Count them, and look for negative values in any other questionnaire item.
# PRS and batch x PC columns are continuous and can be negative, so they are not checked.

item_cols <- unlist(modality_cols[c("SUD15", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX")],
                    use.names = FALSE)
code_rows <- lapply(item_cols, function(col) {
  x_all <- catss[[col]]
  x_in <- catss[[col]][included]
  data.frame(
    Variable = col,
    Minus1_all_rows = fmt_count(sum(x_all == -1, na.rm = TRUE)),
    Minus1_analysis_sample = fmt_count(sum(x_in == -1, na.rm = TRUE)),
    Minus1_pct_analysis_sample = sprintf("%.1f", 100 * mean(x_in == -1, na.rm = TRUE)),
    Other_negative_analysis_sample = fmt_count(sum(x_in < 0 & x_in != -1, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
})
code_check <- bind_rows(code_rows) %>%
  filter(Minus1_all_rows != "0" | Other_negative_analysis_sample != "0")
write_csv(code_check, "missing_code_check.csv")
if (nrow(code_check) == 0) {
  cat("No -1 or other negative values found in the questionnaire items.\n")
} else {
  print(code_check)
}
cat("Saved: missing_code_check.csv\n")
