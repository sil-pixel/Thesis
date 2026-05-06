## Forest plots and interaction bar plots for GLMM significant coefficients
## Author: Silpa Soni Nallacheruvu
## Date: 2026-05-04


library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(stringr)
library(readr)

# ── Load coefficient tables ──
# These should be the CSV files saved from your GLMM analysis
pos_coefs <- read_csv("GLMM_Pos_significant_table.csv")
neg_coefs <- read_csv("GLMM_Neg_significant_table.csv")

# Add row names back as a column if needed
if (!"Variable" %in% names(pos_coefs)) {
  pos_coefs$Variable <- rownames(pos_coefs)
}
if (!"Variable" %in% names(neg_coefs)) {
  neg_coefs$Variable <- rownames(neg_coefs)
}

# ── Helper: classify variables ──
classify_variable <- function(var_name) {
  if (grepl(":", var_name)) {
    return("Interaction")
  } else if (grepl("^SUD15", var_name)) {
    return("SUD15")
  } else if (grepl("^SCZ15", var_name)) {
    return("SCZ15")
  } else if (grepl("^ACE", var_name)) {
    return("ACE")
  } else if (grepl("^PRS", var_name)) {
    return("PRS")
  } else if (grepl("^ADHD9", var_name)) {
    return("ADHD9")
  } else if (grepl("^ASD9", var_name)) {
    return("ASD9")
  } else if (grepl("^SES", var_name)) {
    return("SES")
  } else if (grepl("^SEX", var_name)) {
    return("SEX")
  } else if (grepl("factor\\(batch\\)", var_name)) {
    return("Batch×PC")
  } else {
    return("Other")
  }
}

# ── Helper: clean variable names for display ──
clean_name <- function(var_name) {
  name <- var_name
  name <- gsub("factor\\(batch\\)", "batch", name)
  # Shorten long interaction names
  if (nchar(name) > 50) {
    parts <- strsplit(name, ":")[[1]]
    parts <- sapply(parts, function(p) {
      if (nchar(p) > 25) substr(p, 1, 25) else p
    })
    name <- paste(parts, collapse = ":")
  }
  return(name)
}

# ── Prepare data for forest plot ──
prepare_forest_data <- function(coefs_df, model_tag) {
  df <- coefs_df %>%
    mutate(
      Model = model_tag,
      CI_lower = Estimate - 1.96 * `Std. Error`,
      CI_upper = Estimate + 1.96 * `Std. Error`,
      Category = sapply(Variable, classify_variable),
      Display_name = sapply(Variable, clean_name)
    ) %>%
    arrange(Estimate)
  
  # Order factor levels by estimate
  df$Display_name <- factor(df$Display_name, levels = df$Display_name)
  return(df)
}

pos_forest <- prepare_forest_data(pos_coefs, "Positive (Psychotic)")
neg_forest <- prepare_forest_data(neg_coefs, "Negative (Depressive)")


# ══════════════════════════════════════════
# 1. SIDE-BY-SIDE FOREST PLOTS
# ══════════════════════════════════════════

make_forest_plot <- function(df, title) {
  ggplot(df, aes(x = Estimate, y = Display_name, color = Category)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper),
                   height = 0.3, linewidth = 0.5) +
    geom_point(size = 2.5) +
    scale_color_manual(values = c(
      "Interaction" = "#e74c3c",
      "SUD15" = "#e67e22",
      "SCZ15" = "#3498db",
      "ACE" = "#9b59b6",
      "PRS" = "#2ecc71",
      "ADHD9" = "#1abc9c",
      "ASD9" = "#f39c12",
      "SES" = "#95a5a6",
      "SEX" = "#e91e63",
      "Batch×PC" = "#607d8b",
      "Other" = "#bdc3c7"
    )) +
    labs(
      title = title,
      x = "Estimate (95% CI)",
      y = NULL,
      color = "Variable Type"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      plot.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom"
    )
}

p_pos <- make_forest_plot(pos_forest, "Positive SCZ Symptoms (Psychotic)")
p_neg <- make_forest_plot(neg_forest, "Negative SCZ Symptoms (Depressive)")

# Side by side
combined_forest <- p_pos + p_neg +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave("GLMM_forest_plots_side_by_side.png", combined_forest,
       width = 20, height = 12, dpi = 300)
cat("Saved: GLMM_forest_plots_side_by_side.png\n")


# ══════════════════════════════════════════
# 2. TOP 10 TERMS FOREST PLOT (cleaner)
# ══════════════════════════════════════════

make_top_forest <- function(df, title, n = 10) {
  # Select top N by absolute estimate
  top_df <- df %>%
    arrange(desc(abs(Estimate))) %>%
    head(n) %>%
    arrange(Estimate)
  top_df$Display_name <- factor(top_df$Display_name, levels = top_df$Display_name)
  
  ggplot(top_df, aes(x = Estimate, y = Display_name, color = Category)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper),
                   height = 0.3, linewidth = 0.6) +
    geom_point(size = 3) +
    scale_color_manual(values = c(
      "Interaction" = "#e74c3c",
      "SUD15" = "#e67e22",
      "SCZ15" = "#3498db",
      "ACE" = "#9b59b6",
      "PRS" = "#2ecc71",
      "ADHD9" = "#1abc9c",
      "ASD9" = "#f39c12",
      "SES" = "#95a5a6",
      "SEX" = "#e91e63",
      "Batch×PC" = "#607d8b",
      "Other" = "#bdc3c7"
    )) +
    labs(
      title = title,
      x = "Estimate (95% CI)",
      y = NULL,
      color = "Variable Type"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 10),
      plot.title = element_text(size = 12, face = "bold"),
      legend.position = "right"
    )
}

p_top_pos <- make_top_forest(pos_forest, "Positive SCZ — Top 10 Predictors")
p_top_neg <- make_top_forest(neg_forest, "Negative SCZ — Top 10 Predictors")

combined_top <- p_top_pos / p_top_neg +
  plot_layout(guides = "collect") &
  theme(legend.position = "right")

ggsave("GLMM_top10_forest_plots.png", combined_top,
       width = 14, height = 12, dpi = 300)
cat("Saved: GLMM_top10_forest_plots.png\n")


# ══════════════════════════════════════════
# 3. INTERACTION EFFECTS GROUPED BY SUBSTANCE
# ══════════════════════════════════════════

extract_substance <- function(var_name) {
  if (!grepl("^SUD15_", var_name) || !grepl(":", var_name)) return(NA)
  sub_part <- strsplit(var_name, ":")[[1]][1]
  sub_part <- gsub("^SUD15_", "", sub_part)
  sub_part <- gsub("15$", "", sub_part)
  return(sub_part)
}

extract_interacting_modality <- function(var_name) {
  if (!grepl(":", var_name)) return(NA)
  parts <- strsplit(var_name, ":")[[1]]
  return(parts[2])
}

prepare_interaction_data <- function(coefs_df, model_tag) {
  df <- coefs_df %>%
    filter(grepl("^SUD15_.*:", Variable)) %>%
    mutate(
      Model = model_tag,
      Substance = sapply(Variable, extract_substance),
      Interacting_with = sapply(Variable, extract_interacting_modality),
      CI_lower = Estimate - 1.96 * `Std. Error`,
      CI_upper = Estimate + 1.96 * `Std. Error`,
      Direction = ifelse(Estimate > 0, "Risk", "Protective")
    )
  return(df)
}

pos_interactions <- prepare_interaction_data(pos_coefs, "Positive (Psychotic)")
neg_interactions <- prepare_interaction_data(neg_coefs, "Negative (Depressive)")
all_interactions <- bind_rows(pos_interactions, neg_interactions)

if (nrow(all_interactions) > 0) {
  # Clean up long names for display
  all_interactions$Interacting_short <- sapply(all_interactions$Interacting_with, function(x) {
    if (nchar(x) > 30) substr(x, 1, 30) else x
  })
  
  p_interactions <- ggplot(all_interactions,
                           aes(x = reorder(Interacting_short, Estimate),
                               y = Estimate, fill = Direction)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width = 0.3, linewidth = 0.4) +
    coord_flip() +
    facet_grid(Substance ~ Model, scales = "free_y", space = "free_y") +
    scale_fill_manual(values = c("Risk" = "#e74c3c", "Protective" = "#3498db")) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    labs(
      title = "SUD15 Interaction Effects by Substance Type",
      subtitle = "Grouped by substance, faceted by symptom dimension",
      x = "Interacting Variable",
      y = "Estimate (95% CI)",
      fill = "Effect Direction"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(size = 13, face = "bold")
    )
  
  ggsave("GLMM_interaction_effects_by_substance.png", p_interactions,
         width = 16, height = 10, dpi = 300)
  cat("Saved: GLMM_interaction_effects_by_substance.png\n")
}


# ══════════════════════════════════════════
# 4. SHARED vs UNIQUE PREDICTORS VENN-STYLE
# ══════════════════════════════════════════

pos_vars <- pos_coefs$Variable
neg_vars <- neg_coefs$Variable

shared <- intersect(pos_vars, neg_vars)
pos_only <- setdiff(pos_vars, neg_vars)
neg_only <- setdiff(neg_vars, pos_vars)

cat("\n═══════════════════════════════════════\n")
cat("  Shared vs Unique Significant Predictors\n")
cat("═══════════════════════════════════════\n")
cat(sprintf("  Positive only: %d terms\n", length(pos_only)))
cat(sprintf("  Negative only: %d terms\n", length(neg_only)))
cat(sprintf("  Shared:        %d terms\n", length(shared)))

if (length(shared) > 0) {
  cat("\n  Shared terms (direction comparison):\n")
  for (var in shared) {
    pos_est <- pos_coefs$Estimate[pos_coefs$Variable == var]
    neg_est <- neg_coefs$Estimate[neg_coefs$Variable == var]
    same_dir <- ifelse(sign(pos_est) == sign(neg_est), "same direction", "OPPOSITE")
    cat(sprintf("    %s: Pos=%.3f, Neg=%.3f (%s)\n", var, pos_est, neg_est, same_dir))
  }
}

# Save summary
summary_df <- data.frame(
  Category = c("Positive only", "Negative only", "Shared"),
  Count = c(length(pos_only), length(neg_only), length(shared)),
  Variables = c(
    paste(head(pos_only, 5), collapse = ", "),
    paste(head(neg_only, 5), collapse = ", "),
    paste(shared, collapse = ", ")
  )
)
write_csv(summary_df, "GLMM_shared_unique_predictors.csv")
cat("\nSaved: GLMM_shared_unique_predictors.csv\n")