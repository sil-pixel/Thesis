## Descriptive statistics table — one row per modality group
## Reports missing values per group
## Author: Silpa Soni Nallacheruvu
## Date: 12/05/2026
## Project: Deep Cross Modal Fusion Model for predicting schizophrenia

library(readr)
library(dplyr)
library(tidyr)
library(kableExtra)

# ── Load data (no complete cases filter) ──
catss <- read_csv("catss_final_data.csv")
cat(sprintf("Total rows: %d\n", nrow(catss)))
cat(sprintf("Complete cases: %d\n", sum(complete.cases(catss))))
cat(sprintf("Rows with any missing: %d\n", sum(!complete.cases(catss))))

# ── Define modality groups ──
modality_groups <- list(
  "SUD15 (Substance Use, age 15)"         = "^SUD15",
  "PRS (Polygenic Risk Scores)"           = "^PRS",
  "SCZ15 (Psychotic Experiences, age 15)" = "^SCZ15",
  "ADHD9 (ADHD Symptoms, age 9)"          = "^ADHD9",
  "ASD9 (ASD Traits, age 9)"              = "^ASD9",
  "ACE15 (Adverse Experiences, age 15)"    = "^ACE15",
  "ACE18 (Adverse Experiences, age 18)"    = "^ACE18",
  "SUD18 (Substance Use, age 18)"         = "^SUD18",
  "SES (Socioeconomic Status)"            = "^SES",
  "SEX"                                    = "^SEX",
  "SCZ18 Positive (Outcome)"              = "^SCZ18_Pos_Norm",
  "SCZ18 Negative (Outcome)"              = "^SCZ18_Neg_Norm"
)


# ── Compute summary per group ──
summarise_group <- function(df, cols, group_name) {
  if (length(cols) == 0) return(NULL)
  vals <- as.matrix(df[, cols])
  all_vals <- as.numeric(vals)
  
  unique_vals <- sort(unique(na.omit(all_vals)))
  n_vars <- length(cols)
  n_obs <- nrow(df)
  n_total_cells <- length(all_vals)
  n_missing_cells <- sum(is.na(all_vals))
  
  # Count rows with any missing in this group
  n_rows_missing <- sum(apply(vals, 1, function(r) any(is.na(r))))
  pct_rows_missing <- n_rows_missing / n_obs * 100
  
  # Special case: SEX coded as 1/2
  if (grepl("^SEX", group_name)) {
    n_male <- sum(all_vals == 1, na.rm = TRUE)
    n_female <- sum(all_vals == 2, na.rm = TRUE)
    n_na <- sum(is.na(all_vals))
    data.frame(
      N_Variables = n_vars, N_Observations = n_obs,
      N_Missing_Rows = n_rows_missing, Pct_Missing_Rows = round(pct_rows_missing, 1),
      Type = "Categorical",
      Mean = NA, SD = NA, Median = NA, Min = 1, Max = 2,
      Pct_Positive = NA,
      Note = sprintf("Male: %d (%.1f%%), Female: %d (%.1f%%), Missing: %d (%.1f%%)",
                     n_male, n_male / n_obs * 100,
                     n_female, n_female / n_obs * 100,
                     n_na, n_na / n_obs * 100),
      stringsAsFactors = FALSE
    )
  } else if (all(unique_vals %in% c(0, 1))) {
    # Binary
    prevalence <- mean(all_vals, na.rm = TRUE) * 100
    data.frame(
      N_Variables = n_vars, N_Observations = n_obs,
      N_Missing_Rows = n_rows_missing, Pct_Missing_Rows = round(pct_rows_missing, 1),
      Type = "Binary",
      Mean = round(prevalence, 1),
      SD = round(sd(rowMeans(vals, na.rm = TRUE), na.rm = TRUE), 3),
      Median = NA, Min = 0, Max = 1,
      Pct_Positive = round(prevalence, 1), Note = NA,
      stringsAsFactors = FALSE
    )
  } else {
    # Continuous/ordinal
    data.frame(
      N_Variables = n_vars, N_Observations = n_obs,
      N_Missing_Rows = n_rows_missing, Pct_Missing_Rows = round(pct_rows_missing, 1),
      Type = ifelse(all(unique_vals == floor(unique_vals)) & length(unique_vals) <= 10,
                    "Ordinal", "Continuous"),
      Mean = round(mean(all_vals, na.rm = TRUE), 3),
      SD = round(sd(all_vals, na.rm = TRUE), 3),
      Median = round(median(all_vals, na.rm = TRUE), 3),
      Min = round(min(all_vals, na.rm = TRUE), 3),
      Max = round(max(all_vals, na.rm = TRUE), 3),
      Pct_Positive = NA, Note = NA,
      stringsAsFactors = FALSE
    )
  }
}

results <- list()
for (group_name in names(modality_groups)) {
  pattern <- modality_groups[[group_name]]
  cols <- grep(pattern, colnames(catss), value = TRUE)
  
  if (length(cols) == 0) {
    cat(sprintf("  WARNING: No columns matched for '%s' (pattern: %s)\n", group_name, pattern))
    next
  }
  
  cat(sprintf("  %s: %d columns matched\n", group_name, length(cols)))
  row <- summarise_group(catss, cols, group_name)
  row$Modality <- group_name
  results[[group_name]] <- row
}

summary_df <- bind_rows(results) %>%
  select(Modality, N_Variables, N_Observations, N_Missing_Rows, Pct_Missing_Rows,
         Type, Mean, SD, Median, Min, Max, Pct_Positive, Note)


# ── Print ──
cat("\n")
print(summary_df, row.names = FALSE)


# ── Save as CSV ──
write_csv(summary_df, "descriptive_statistics.csv")
cat("\nSaved: descriptive_statistics.csv\n")


# ── Generate formatted HTML table ──
html_df <- summary_df %>%
  mutate(
    `Mean ± SD` = case_when(
      Type == "Categorical" ~ Note,
      Type == "Binary" ~ sprintf("%.1f%%", Pct_Positive),
      TRUE ~ sprintf("%.3f ± %.3f", Mean, SD)
    ),
    Range = case_when(
      Type == "Categorical" ~ "1, 2",
      Type == "Binary" ~ "0, 1",
      TRUE ~ sprintf("%.3f – %.3f", Min, Max)
    ),
    Missing = sprintf("%d (%.1f%%)", N_Missing_Rows, Pct_Missing_Rows)
  ) %>%
  select(Modality, `N Vars` = N_Variables, Type, `Mean ± SD`, Median, Range, Missing)

html_table <- html_df %>%
  kable(format = "html", align = c("l", "c", "c", "c", "c", "c", "c")) %>%
  kable_styling(
    bootstrap_options = c("striped", "hover", "condensed"),
    full_width = FALSE, font_size = 13
  ) %>%
  row_spec(0, bold = TRUE, background = "#f0f0f0")

writeLines(as.character(html_table), "descriptive_statistics.html")
cat("Saved: descriptive_statistics.html\n")


# ── LaTeX version for thesis ──
latex_df <- summary_df %>%
  mutate(
    `Mean ± SD` = case_when(
      Type == "Categorical" ~ gsub("%", "\\\\%", Note),
      Type == "Binary" ~ sprintf("%.1f\\%%", Pct_Positive),
      TRUE ~ sprintf("%.3f $\\pm$ %.3f", Mean, SD)
    ),
    Range = case_when(
      Type == "Categorical" ~ "1, 2",
      Type == "Binary" ~ "0, 1",
      TRUE ~ sprintf("%.3f--%.3f", Min, Max)
    ),
    Missing = sprintf("%d (%.1f\\%%)", N_Missing_Rows, Pct_Missing_Rows)
  ) %>%
  select(Modality, `N Vars` = N_Variables, Type, `Mean ± SD`, Median, Range, Missing)

latex_table <- latex_df %>%
  kable(format = "latex", booktabs = TRUE, escape = FALSE,
        caption = "Descriptive statistics of input modalities and outcome variables. Missing reports the number and percentage of rows with any missing value in that modality group.") %>%
  kable_styling(latex_options = c("hold_position", "scale_down"))

writeLines(as.character(latex_table), "descriptive_statistics.tex")
cat("Saved: descriptive_statistics.tex\n")