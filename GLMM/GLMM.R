## GLMM model for Psychotic Risk prediction from Substance Use and other factors in CATSS
## Author: Silpa Soni Nallacheruvu
## Date: 2026-05-06
## Converted from GLMM.Rmd to standalone R script with file outputs.

library(readr)
library(lme4)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(kableExtra)
library(patchwork)


# ── Load and clean data ──
catss <- read_csv("catss_final_data.csv")
catss <- catss[complete.cases(catss), ]
cat(sprintf("Complete cases: %d\n", nrow(catss)))


# ── Prepare formula components ──
SUD15_vars <- grep("^SUD15", names(catss), value = TRUE)
SCZ15_vars <- grep("^SCZ15", names(catss), value = TRUE)
PRS_vars   <- grep("^PRS", names(catss), value = TRUE)
ADHD9_vars <- grep("^ADHD9", names(catss), value = TRUE)
ASD9_vars  <- grep("^ASD9", names(catss), value = TRUE)
ACE_vars   <- grep("^ACE", names(catss), value = TRUE)
SES_vars   <- grep("^SES", names(catss), value = TRUE)
PCA_vars   <- grep("^PC", names(catss), value = TRUE)[1:2]
Sex_var    <- grep("^SEX", names(catss), value = TRUE)
Batch_var  <- grep("^batch", names(catss), value = TRUE)[1]

covariate_terms <- paste(
  c(SUD15_vars, SCZ15_vars, ADHD9_vars, ASD9_vars, ACE_vars, SES_vars, PRS_vars, Sex_var),
  collapse = " + "
)

genetic_confounding_terms <- paste(
  paste0("factor(", Batch_var, ") : ", PCA_vars),
  collapse = " + "
)

interaction_terms <- paste(
  c(
    paste0(SUD15_vars, ":", SCZ15_vars),
    paste0(SUD15_vars, ":", ADHD9_vars),
    paste0(SUD15_vars, ":", ASD9_vars),
    paste0(SUD15_vars, ":", ACE_vars),
    paste0(SUD15_vars, ":", SES_vars),
    paste0(SUD15_vars, ":", PRS_vars),
    paste0(SUD15_vars, ":", Sex_var)
  ),
  collapse = " + "
)

# Build formulas
pos_glmm_formula <- as.formula(paste(
  "SCZ18_Pos_Norm ~", covariate_terms, "+",
  interaction_terms, "+", genetic_confounding_terms, "+ (1 | cmpair)"
))

neg_glmm_formula <- as.formula(paste(
  "SCZ18_Neg_Norm ~", covariate_terms, "+",
  interaction_terms, "+", genetic_confounding_terms, "+ (1 | cmpair)"
))


# ══════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════

extract_significant_coefs <- function(model, model_tag) {
  coefs <- summary(model)$coefficients
  df_resid <- df.residual(model)
  p_values <- 2 * (1 - pt(abs(coefs[, "t value"]), df = df_resid))
  coefs <- cbind(as.data.frame(coefs), p_value = p_values)
  coefs$significance <- cut(
    coefs$p_value,
    breaks = c(-Inf, 0.001, 0.01, 0.05, 0.1, Inf),
    labels = c("***", "**", "*", ".", "")
  )
  coefs$Variable <- rownames(coefs)
  
  # Full table (all coefficients)
  full_table <- coefs %>%
    filter(Variable != "(Intercept)") %>%
    arrange(p_value) %>%
    select(Variable, Estimate, `Std. Error`, `t value`, p_value, significance) %>%
    mutate(
      Estimate = round(Estimate, 4),
      `Std. Error` = round(`Std. Error`, 4),
      `t value` = round(`t value`, 3),
      p_value = signif(p_value, 3)
    )
  
  # Significant only (p < 0.05)
  sig_table <- full_table %>% filter(p_value < 0.05)
  
  # Save both
  write_csv(full_table, sprintf("%s_GLMM_all_coefs.csv", model_tag))
  write_csv(sig_table, sprintf("%s_GLMM_significant_table.csv", model_tag))
  cat(sprintf("  Saved: %s_GLMM_all_coefs.csv (%d terms)\n", model_tag, nrow(full_table)))
  cat(sprintf("  Saved: %s_GLMM_significant_table.csv (%d significant terms)\n", model_tag, nrow(sig_table)))
  
  return(list(full = full_table, significant = sig_table))
}


plot_predictions <- function(y_true, y_pred, model_tag, seed) {
  df <- data.frame(actual = y_true, predicted = y_pred)
  df$residuals <- df$actual - df$predicted
  
  # Metrics
  r2 <- cor(df$actual, df$predicted)^2
  rmse <- sqrt(mean((df$actual - df$predicted)^2))
  mae <- mean(abs(df$actual - df$predicted))
  pearson_r <- cor(df$actual, df$predicted)
  spearman_r <- cor(df$actual, df$predicted, method = "spearman")
  
  metrics_text <- sprintf(
    "R² = %.4f\nRMSE = %.4f\nMAE = %.4f\nSpearman ρ = %.4f\nPearson r = %.4f",
    r2, rmse, mae, spearman_r, pearson_r
  )
  
  # Predicted vs Actual
  p1 <- ggplot(df, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.4, color = "#2C7FB8", size = 1.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linewidth = 0.8) +
    annotate("text",
             x = quantile(df$actual, 0.95),
             y = quantile(df$predicted, 0.05),
             label = metrics_text,
             hjust = 1, vjust = 0, size = 3.5,
             fontface = "italic") +
    labs(
      title = sprintf("%s GLMM — Predicted vs Actual (seed %d)", model_tag, seed),
      x = "Actual Values",
      y = "Predicted Values"
    ) +
    theme_minimal()
  
  # Residual plot
  p2 <- ggplot(df, aes(x = predicted, y = residuals)) +
    geom_point(alpha = 0.4, color = "#74A9CF", size = 1.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_smooth(method = "loess", se = TRUE, color = "#e74c3c", alpha = 0.2, linewidth = 0.6) +
    labs(
      title = sprintf("%s GLMM — Residual Plot (seed %d)", model_tag, seed),
      x = "Predicted Values",
      y = "Residuals (Actual − Predicted)"
    ) +
    theme_minimal()
  
  combined <- p1 + p2 + plot_layout(ncol = 2)
  
  filename <- sprintf("%s_GLMM_pred_vs_actual_seed_%d.png", model_tag, seed)
  ggsave(filename, combined, width = 14, height = 6, dpi = 150)
  cat(sprintf("  Saved: %s\n", filename))
  
  return(list(r2 = r2, rmse = rmse, mae = mae,
              spearman_rho = spearman_r, pearson_r = pearson_r))
}


plot_random_effect_qq <- function(model, model_tag, seed) {
  re <- ranef(model)$cmpair[, 1]
  qq <- qqnorm(re, plot.it = FALSE)
  
  df <- data.frame(theoretical = qq$x, observed = qq$y)
  
  p <- ggplot(df, aes(x = theoretical, y = observed)) +
    geom_point(alpha = 0.3, size = 1) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.8) +
    labs(
      title = sprintf("%s GLMM — Twin Random Effect QQ Plot (seed %d)", model_tag, seed),
      x = "Theoretical Quantiles",
      y = "Random Effects"
    ) +
    theme_minimal()
  
  filename <- sprintf("%s_GLMM_random_effect_qq_seed_%d.png", model_tag, seed)
  ggsave(filename, p, width = 7, height = 6, dpi = 150)
  cat(sprintf("  Saved: %s\n", filename))
}


# ══════════════════════════════════════════
# Multi-seed evaluation
# ══════════════════════════════════════════

seeds <- c(42, 43, 44, 45, 46)
test_size <- 0.25

pos_results <- list()
neg_results <- list()

for (i in seq_along(seeds)) {
  set.seed(seeds[i])
  cat(sprintf("\n══ Seed %d ══\n", seeds[i]))
  
  # Split
  unique_ids <- unique(catss$cmpair)
  n_test <- floor(length(unique_ids) * test_size)
  test_ids <- sample(unique_ids, size = n_test)
  
  test_df  <- catss[catss$cmpair %in% test_ids, ]
  train_df <- catss[!catss$cmpair %in% test_ids, ]
  train_df$cmpair <- as.factor(train_df$cmpair)
  
  cat(sprintf("  Train: %d rows, Test: %d rows\n", nrow(train_df), nrow(test_df)))
  
  # ── Positive GLMM ──
  cat("  Fitting Positive GLMM...\n")
  pos_model <- glmer(pos_glmm_formula, data = train_df)
  
  # Save coefficients (only for first seed to avoid overwriting)
  if (i == 1) {
    cat("  Extracting Positive coefficients...\n")
    pos_coef_tables <- extract_significant_coefs(pos_model, "Pos")
    plot_random_effect_qq(pos_model, "Pos", seeds[i])
  }
  
  pos_pred <- predict(pos_model, newdata = test_df, allow.new.levels = TRUE, type = "response")
  pos_metrics <- plot_predictions(test_df$SCZ18_Pos_Norm, pos_pred, "Pos", seeds[i])
  pos_metrics$seed <- seeds[i]
  pos_results[[i]] <- as.data.frame(pos_metrics)
  
  cat(sprintf("  Pos — RMSE: %.4f, R²: %.4f, Spearman: %.4f\n",
              pos_metrics$rmse, pos_metrics$r2, pos_metrics$spearman_rho))
  
  # ── Negative GLMM ──
  cat("  Fitting Negative GLMM...\n")
  neg_model <- glmer(neg_glmm_formula, data = train_df)
  
  if (i == 1) {
    cat("  Extracting Negative coefficients...\n")
    neg_coef_tables <- extract_significant_coefs(neg_model, "Neg")
    plot_random_effect_qq(neg_model, "Neg", seeds[i])
  }
  
  neg_pred <- predict(neg_model, newdata = test_df, allow.new.levels = TRUE, type = "response")
  neg_metrics <- plot_predictions(test_df$SCZ18_Neg_Norm, neg_pred, "Neg", seeds[i])
  neg_metrics$seed <- seeds[i]
  neg_results[[i]] <- as.data.frame(neg_metrics)
  
  cat(sprintf("  Neg — RMSE: %.4f, R²: %.4f, Spearman: %.4f\n",
              neg_metrics$rmse, neg_metrics$r2, neg_metrics$spearman_rho))
}


# ══════════════════════════════════════════
# Summary across seeds
# ══════════════════════════════════════════

pos_df <- bind_rows(pos_results)
neg_df <- bind_rows(neg_results)

summarise_results <- function(df, model_tag) {
  metrics <- c("rmse", "mae", "r2", "spearman_rho", "pearson_r")
  row <- data.frame(Model = model_tag)
  for (m in metrics) {
    row[[paste0(m, "_mean")]] <- mean(df[[m]])
    row[[paste0(m, "_std")]]  <- sd(df[[m]])
  }
  return(row)
}

glmm_summary <- bind_rows(
  summarise_results(pos_df, "Pos_GLMM"),
  summarise_results(neg_df, "Neg_GLMM")
)

cat("\n══════════════════════════════════════════\n")
cat("  GLMM Summary across", length(seeds), "seeds\n")
cat("══════════════════════════════════════════\n")
print(glmm_summary)

# Save everything
write_csv(glmm_summary, "GLMM_summary.csv")
write_csv(pos_df, "GLMM_Pos_per_seed.csv")
write_csv(neg_df, "GLMM_Neg_per_seed.csv")

cat("\nSaved: GLMM_summary.csv\n")
cat("Saved: GLMM_Pos_per_seed.csv\n")
cat("Saved: GLMM_Neg_per_seed.csv\n")