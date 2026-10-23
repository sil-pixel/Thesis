## Forest plots and interaction bar plots for the updated GLMM significant coefficients
## Updated version of GLMM_Analysis.R (2026-10-22) for the outputs of GLMM_updated_split.R. Changes:
##   1. Inputs: reads Pos/Neg_GLMM_updated_significant_table.csv.
##   2. Interaction plot: the updated interactions are SUD15_score x one partner per fused modality
##      (SCZ15, ADHD9, ASD9, ACE15, ACE18, SUD18, SES scores, SEX) and SUD15_score x each cell-type PRS.
##      Terms are grouped by partner modality instead of by substance item.
##   3. Modality colours: SUD18 added, batch_<chip>_x_PC<k> columns classified as Batch×PC.
##   4. Outputs are saved with the suffix _updated, so the original figures are not overwritten.
## Coefficients come from the first split (seed 42), as saved by GLMM_updated_split.R.
##
## Usage: run from the folder containing the GLMM_updated_split.R outputs.
##   Rscript GLMM_Analysis_updated.R

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(stringr)
library(readr)

# ── Load coefficient tables ──
pos_coefs <- read_csv("Pos_GLMM_updated_significant_table.csv", show_col_types = FALSE)
neg_coefs <- read_csv("Neg_GLMM_updated_significant_table.csv", show_col_types = FALSE)


# ── Helper: modality of a single (non-interaction) term ──
term_modality <- function(term) {
  if (grepl("^SUD15", term)) return("SUD15")
  if (grepl("^SUD18", term)) return("SUD18")
  if (grepl("^SCZ15", term)) return("SCZ15")
  if (grepl("^ACE", term)) return("ACE")
  if (grepl("^PRS", term)) return("PRS")
  if (grepl("^ADHD9", term)) return("ADHD9")
  if (grepl("^ASD9", term)) return("ASD9")
  if (grepl("^SES", term)) return("SES")
  if (grepl("^SEX", term)) return("SEX")
  if (grepl("^batch_.*_x_PC", term)) return("Batch×PC")
  return("Other")
}

# ── Helper: classify variables ──
classify_variable <- function(var_name) {
  if (grepl(":", var_name)) return("Interaction")
  return(term_modality(var_name))
}

# ── Helper: clean variable names for display ──
clean_name <- function(var_name) {
  name <- var_name
  if (nchar(name) > 50) {
    parts <- strsplit(name, ":")[[1]]
    parts <- sapply(parts, function(p) {
      if (nchar(p) > 25) substr(p, 1, 25) else p
    })
    name <- paste(parts, collapse = ":")
  }
  return(name)
}


# ── Color palette for modality categories ──
category_colors <- c(
  "Interaction" = "#e74c3c",
  "SUD15" = "#e67e22",
  "SUD18" = "#a0522d",
  "SCZ15" = "#3498db",
  "ACE" = "#9b59b6",
  "PRS" = "#2ecc71",
  "ADHD9" = "#1abc9c",
  "ASD9" = "#f39c12",
  "SES" = "#95a5a6",
  "SEX" = "#e91e63",
  "Batch×PC" = "#607d8b",
  "Other" = "#bdc3c7"
)

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
  df$Display_name <- factor(df$Display_name, levels = unique(df$Display_name))
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
    scale_color_manual(values = category_colors) +
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

combined_forest <- p_pos + p_neg +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave("GLMM_updated_forest_plots_side_by_side.png", combined_forest,
       width = 20, height = 12, dpi = 300)
cat("Saved: GLMM_updated_forest_plots_side_by_side.png\n")


# ══════════════════════════════════════════
# 2. TOP 10 TERMS FOREST PLOT
# ══════════════════════════════════════════

make_top_forest <- function(df, title, n = 10) {
  # Select top N by absolute estimate
  top_df <- df %>%
    arrange(desc(abs(Estimate))) %>%
    head(n) %>%
    arrange(Estimate)
  top_df$Display_name <- factor(top_df$Display_name, levels = unique(top_df$Display_name))

  ggplot(top_df, aes(x = Estimate, y = Display_name, color = Category)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper),
                   height = 0.3, linewidth = 0.6) +
    geom_point(size = 3) +
    scale_color_manual(values = category_colors) +
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

ggsave("GLMM_updated_top10_forest_plots.png", combined_top,
       width = 14, height = 12, dpi = 300)
cat("Saved: GLMM_updated_top10_forest_plots.png\n")


# ══════════════════════════════════════════
# 3. INTERACTION EFFECTS GROUPED BY PARTNER MODALITY
# ══════════════════════════════════════════

# Partner of SUD15_score in each interaction, e.g. "SCZ15_score", "SEX", "PRS_Astrocyte_z"
extract_partner <- function(var_name) {
  parts <- strsplit(var_name, ":")[[1]]
  partner <- parts[parts != "SUD15_score"]
  if (length(partner) != 1) return(NA)
  return(partner)
}

# Display label: modality name for scores, cell type for PRS
partner_label <- function(partner) {
  label <- gsub("_score$", "", partner)
  label <- gsub("^PRS_", "", label)
  label <- gsub("_z$", "", label)
  return(label)
}

prepare_interaction_data <- function(coefs_df, model_tag) {
  df <- coefs_df %>%
    filter(grepl("SUD15_score", Variable) & grepl(":", Variable)) %>%
    mutate(
      Model = model_tag,
      Partner = sapply(Variable, extract_partner),
      CI_lower = Estimate - 1.96 * `Std. Error`,
      CI_upper = Estimate + 1.96 * `Std. Error`,
      Direction = ifelse(Estimate > 0, "Risk", "Protective")
    ) %>%
    mutate(
      Partner_group = ifelse(sapply(Partner, term_modality) == "PRS",
                             "Cell-type PRS", "Phenotypic modalities"),
      Partner_label = sapply(Partner, partner_label)
    )
  return(df)
}

pos_interactions <- prepare_interaction_data(pos_coefs, "Positive SCZ")
neg_interactions <- prepare_interaction_data(neg_coefs, "Negative SCZ")
all_interactions <- bind_rows(pos_interactions, neg_interactions)

if (nrow(all_interactions) > 0) {
  all_interactions$Model <- factor(all_interactions$Model,
                                   levels = c("Positive SCZ", "Negative SCZ"))
  all_interactions$Partner_group <- factor(all_interactions$Partner_group,
                                           levels = c("Phenotypic modalities", "Cell-type PRS"))

  p_interactions <- ggplot(all_interactions,
                           aes(x = reorder(Partner_label, Estimate),
                               y = Estimate, fill = Direction)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width = 0.3, linewidth = 0.4) +
    coord_flip() +
    facet_grid(Partner_group ~ Model, scales = "free_y", space = "free_y") +
    scale_fill_manual(values = c("Risk" = "#e74c3c", "Protective" = "#3498db")) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    labs(
      title = "SUD15 Interaction Effects by Partner Modality",
      subtitle = "SUD15 score × fused modality; significant terms only (p < 0.05, seed 42 fit)",
      x = "Interacting with SUD15 score",
      y = "Estimate (95% CI)",
      fill = "Effect Direction"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      strip.text = element_text(size = 10, face = "bold"),
      plot.title = element_text(size = 13, face = "bold")
    )

  ggsave("GLMM_updated_interaction_effects_by_modality.png", p_interactions,
         width = 16, height = 10, dpi = 300)
  cat("Saved: GLMM_updated_interaction_effects_by_modality.png\n")
} else {
  cat("No significant SUD15 interactions; interaction plot skipped.\n")
}


# ══════════════════════════════════════════
# 4. MAIN EFFECTS ONLY (excluding interactions)
# ══════════════════════════════════════════

make_main_effects_plot <- function(df, title) {
  main_df <- df %>% filter(Category != "Interaction")

  if (nrow(main_df) == 0) {
    cat(sprintf("  No main effects to plot for: %s\n", title))
    return(NULL)
  }

  main_df <- main_df %>% arrange(Estimate)
  main_df$Display_name <- factor(main_df$Display_name, levels = unique(main_df$Display_name))

  ggplot(main_df, aes(x = Estimate, y = Display_name, color = Category)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper),
                   height = 0.3, linewidth = 0.6) +
    geom_point(size = 3) +
    scale_color_manual(values = category_colors) +
    labs(
      title = title,
      x = "Estimate (95% CI)",
      y = NULL,
      color = "Modality"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 9),
      plot.title = element_text(size = 12, face = "bold"),
      legend.position = "right"
    )
}

p_main_pos <- make_main_effects_plot(pos_forest, "GLMM — Positive SCZ Main Effects")
p_main_neg <- make_main_effects_plot(neg_forest, "GLMM — Negative SCZ Main Effects")

if (!is.null(p_main_pos) && !is.null(p_main_neg)) {
  combined_main <- p_main_pos / p_main_neg +
    plot_layout(guides = "collect") &
    theme(legend.position = "right")

  ggsave("GLMM_updated_main_effects_forest.png", combined_main,
         width = 14, height = 12, dpi = 300)
  cat("Saved: GLMM_updated_main_effects_forest.png\n")
}


# ══════════════════════════════════════════
# 5. SHARED vs UNIQUE PREDICTORS
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
    paste(pos_only, collapse = ", "),
    paste(neg_only, collapse = ", "),
    paste(shared, collapse = ", ")
  )
)
write_csv(summary_df, "GLMM_updated_shared_unique_predictors.csv")
cat("\nSaved: GLMM_updated_shared_unique_predictors.csv\n")
