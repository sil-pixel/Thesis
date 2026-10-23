## GAMM model for Psychotic Risk prediction from Substance Use and other factors in CATSS
## Updated version of GAMM.R (2026-10-22). Same changes as GLMM_updated_split.R; PRS main effects remain smooth terms.
##   1. Splits: reads the twin-pair splits exported by Models/split_data.py, so GAMM is evaluated on the
##      same train/test partitions as DCMFNet and the ML benchmark (GAMM.R drew its own splits with sample()).
##   2. Interactions: mirror the DCMFNet fusion design. SUD15 is the anchor, and there is one interaction per
##      fused modality (SCZ15, ADHD9, ASD9, ACE15, ACE18, SUD18, SES, SEX) plus one per cell-type PRS.
##      GAMM.R built its interactions with paste0(), which pairs items by position and recycles the shorter
##      list. That gave an arbitrary subset of item pairs.
##   3. Main effects: the same inputs as DCMFNet (ACE15 and ACE18 separate, SUD18 added, batch_x_PC columns).
##   4. R²: 1 - SSE/SST, the same definition as sklearn r2_score used for DCMFNet and the ML models.
##      The squared Pearson correlation used in GAMM.R is kept as r2_cor2.
##   5. SD across seeds: population SD (ddof = 0), as in the Python summaries.
##
## Usage: run from the folder containing catss_final_data.csv and the split_seed_<seed>.csv files
## written by split_data.py. All outputs are saved in the same folder.
##   python split_data.py
##   Rscript GAMM_updated_split.R

library(readr)
library(mgcv)
library(dplyr)
library(ggplot2)


# GAMM uses s() for PRS smooth terms; the twin structure is a random effect smooth per twin pair
MAIN_EFFECT_TERMS <- function(main_effect_vars, PRS_vars) {
  ifelse(main_effect_vars %in% PRS_vars, paste0("s(", main_effect_vars, ")"), main_effect_vars)
}
RANDOM_EFFECT_TERM <- "+ s(cmpair, bs='re')"


# ══════════════════════════════════════════
# Model specification (mirrors DCMFNet modalities)
# ══════════════════════════════════════════

# Returns the variables per modality, the modality scores used for interactions and the formulas.
# Kept separate from the fitting code so tests can check the specification without data
# (tests/test_glmm_gamm_dcmfnet_alignment.py).
build_model_spec <- function(column_names) {
  # Same prefixes as DCMFNet (dcmfnet/training/data.py MODALITY_PATTERNS)
  SUD15_vars <- grep("^SUD15", column_names, value = TRUE)   # anchor modality
  PRS_vars   <- grep("^PRS", column_names, value = TRUE)
  SCZ15_vars <- grep("^SCZ15", column_names, value = TRUE)
  ADHD9_vars <- grep("^ADHD9", column_names, value = TRUE)
  ASD9_vars  <- grep("^ASD9", column_names, value = TRUE)
  ACE15_vars <- grep("^ACE15", column_names, value = TRUE)
  ACE18_vars <- grep("^ACE18", column_names, value = TRUE)
  SUD18_vars <- grep("^SUD18", column_names, value = TRUE)
  SES_vars   <- grep("^SES", column_names, value = TRUE)
  Sex_var    <- grep("^SEX", column_names, value = TRUE)
  BatchPC_vars <- grep("^batch_.*_x_PC", column_names, value = TRUE)
  
  # Questionnaire modalities summarised by one score each for the interaction terms
  score_blocks <- list(
    SUD15 = SUD15_vars, SCZ15 = SCZ15_vars, ADHD9 = ADHD9_vars, ASD9 = ASD9_vars,
    ACE15 = ACE15_vars, ACE18 = ACE18_vars, SUD18 = SUD18_vars, SES = SES_vars
  )
  # Drop groups without columns (e.g. ACE18 and SUD18 in the age-15-only models), so no empty score is built
  score_blocks <- score_blocks[lengths(score_blocks) > 0]
  
  main_effect_vars <- c(SUD15_vars, PRS_vars, SCZ15_vars, ADHD9_vars, ASD9_vars, ACE15_vars,
                        ACE18_vars, SUD18_vars, SES_vars, Sex_var, BatchPC_vars)
  main_effect_terms <- paste(MAIN_EFFECT_TERMS(main_effect_vars, PRS_vars), collapse = " + ")
  
  # One interaction per DCMFNet fusion block: SUD15 score x modality score, SUD15 score x SEX,
  # and SUD15 score x each cell-type PRS (kept separate, as each PRS is its own input in DCMFNet).
  # Item-level main effects already span the score main effects, so scores enter only through interactions.
  interaction_terms <- c(
    paste0("SUD15_score:", setdiff(names(score_blocks), "SUD15"), "_score"),
    paste0("SUD15_score:", Sex_var),
    paste0("SUD15_score:", PRS_vars, "_z")
  )
  
  rhs <- paste(main_effect_terms, "+", paste(interaction_terms, collapse = " + "), RANDOM_EFFECT_TERM)
  list(
    score_blocks = score_blocks,
    PRS_vars = PRS_vars,
    Sex_var = Sex_var,
    BatchPC_vars = BatchPC_vars,
    main_effect_vars = main_effect_vars,
    interaction_terms = interaction_terms,
    pos_formula = as.formula(paste("SCZ18_Pos_Norm ~", rhs)),
    neg_formula = as.formula(paste("SCZ18_Neg_Norm ~", rhs))
  )
}

# ══════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════

# Standardise with training-set mean and SD only (no information from the test set)
standardise <- function(x_train, x_test) {
  m <- mean(x_train)
  s <- sd(x_train)
  if (is.na(s) || s == 0) s <- 1
  list(train = (x_train - m) / s, test = (x_test - m) / s)
}

# Modality score = mean of the standardised items, re-standardised. PRS are standardised individually.
add_interaction_scores <- function(train_df, test_df, spec) {
  for (block in names(spec$score_blocks)) {
    vars <- spec$score_blocks[[block]]
    z_train <- sapply(vars, function(v) standardise(train_df[[v]], test_df[[v]])$train)
    z_test  <- sapply(vars, function(v) standardise(train_df[[v]], test_df[[v]])$test)
    score <- standardise(rowMeans(as.matrix(z_train)), rowMeans(as.matrix(z_test)))
    train_df[[paste0(block, "_score")]] <- score$train
    test_df[[paste0(block, "_score")]]  <- score$test
  }
  for (v in spec$PRS_vars) {
    z <- standardise(train_df[[v]], test_df[[v]])
    train_df[[paste0(v, "_z")]] <- z$train
    test_df[[paste0(v, "_z")]]  <- z$test
  }
  list(train = train_df, test = test_df)
}


read_split <- function(seed, catss) {
  split_file <- sprintf("split_seed_%d.csv", seed)
  split <- read_csv(split_file, col_types = cols(cmpair = col_character(), split = col_character()))
  missing_ids <- setdiff(unique(catss$cmpair), split$cmpair)
  extra_ids <- setdiff(split$cmpair, unique(catss$cmpair))
  if (length(missing_ids) > 0 || length(extra_ids) > 0) {
    stop(sprintf("Split file %s does not match the data (%d pairs missing, %d extra). Re-run split_data.py on the same data file.",
                 split_file, length(missing_ids), length(extra_ids)))
  }
  split
}


extract_gamm_coefs <- function(model, model_tag) {
  # Parametric coefficients
  param_coefs <- as.data.frame(summary(model)$p.table)
  param_coefs$Variable <- rownames(param_coefs)
  param_coefs <- param_coefs %>%
    rename(p_value = `Pr(>|t|)`) %>%
    mutate(significance = cut(
      p_value,
      breaks = c(-Inf, 0.001, 0.01, 0.05, 0.1, Inf),
      labels = c("***", "**", "*", ".", "")
    ))
  
  # Full table
  full_table <- param_coefs %>%
    filter(Variable != "(Intercept)") %>%
    arrange(p_value) %>%
    select(Variable, Estimate, `Std. Error`, `t value`, p_value, significance) %>%
    mutate(
      Estimate = round(Estimate, 4),
      `Std. Error` = round(`Std. Error`, 4),
      `t value` = round(`t value`, 3),
      p_value = signif(p_value, 3)
    )
  
  # Significant only
  sig_table <- full_table %>% filter(p_value < 0.05)
  
  # Smooth terms
  smooth_coefs <- as.data.frame(summary(model)$s.table)
  smooth_coefs$Variable <- rownames(smooth_coefs)
  smooth_coefs <- smooth_coefs %>%
    rename(p_value = `p-value`) %>%
    mutate(
      significance = cut(
        p_value,
        breaks = c(-Inf, 0.001, 0.01, 0.05, 0.1, Inf),
        labels = c("***", "**", "*", ".", "")
      ),
      p_value = signif(p_value, 3)
    )
  
  # Save all
  write_csv(full_table, sprintf("GAMM_updated_%s_all_parametric_coefs.csv", model_tag))
  write_csv(sig_table, sprintf("GAMM_updated_%s_significant_table.csv", model_tag))
  write_csv(smooth_coefs, sprintf("GAMM_updated_%s_smooth_terms.csv", model_tag))
  
  cat(sprintf("  Saved: GAMM_updated_%s_all_parametric_coefs.csv (%d terms)\n", model_tag, nrow(full_table)))
  cat(sprintf("  Saved: GAMM_updated_%s_significant_table.csv (%d significant terms)\n", model_tag, nrow(sig_table)))
  cat(sprintf("  Saved: GAMM_updated_%s_smooth_terms.csv (%d smooth terms)\n", model_tag, nrow(smooth_coefs)))
  
  return(list(full = full_table, significant = sig_table, smooth = smooth_coefs))
}

plot_predictions <- function(y_true, y_pred, model_tag, seed) {
  df <- data.frame(actual = y_true, predicted = y_pred)
  
  r2 <- 1 - sum((df$actual - df$predicted)^2) / sum((df$actual - mean(df$actual))^2)
  r2_cor2 <- cor(df$actual, df$predicted)^2
  rmse <- sqrt(mean((df$actual - df$predicted)^2))
  mae <- mean(abs(df$actual - df$predicted))
  pearson_r <- cor(df$actual, df$predicted)
  spearman_r <- cor(df$actual, df$predicted, method = "spearman")
  
  metrics_text <- sprintf(
    "R² = %.4f\nRMSE = %.4f\nMAE = %.4f\nSpearman ρ = %.4f\nPearson r = %.4f",
    r2, rmse, mae, spearman_r, pearson_r
  )
  
  # Calibration plot only
  p <- ggplot(df, aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.3, color = "#2C7FB8", size = 1.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                color = "red", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, color = "blue", linewidth = 0.8) +
    annotate("label",
             x = max(df$actual) * 0.98,
             y = min(df$predicted) + (max(df$predicted) - min(df$predicted)) * 0.02,
             label = metrics_text,
             hjust = 1, vjust = 0,
             size = 4, fontface = "bold",
             fill = "white", alpha = 0.85,
             label.size = 0.5, label.r = unit(0.3, "lines")) +
    labs(
      title = sprintf("%s GAMM (updated) — Predicted vs Actual (seed %d)", model_tag, seed),
      x = "Actual Values",
      y = "Predicted Values"
    ) +
    theme_minimal(base_size = 13)
  
  filename <- sprintf("%s_GAMM_updated_calibration_seed_%d.png", model_tag, seed)
  ggsave(filename, p, width = 8, height = 7, dpi = 300)
  cat(sprintf("  Saved: %s\n", filename))
  
  # Residual plot saved separately
  df$residuals <- df$actual - df$predicted
  p_resid <- ggplot(df, aes(x = predicted, y = residuals)) +
    geom_point(alpha = 0.3, color = "#74A9CF", size = 1.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 0.8) +
    geom_smooth(method = "loess", se = TRUE, color = "#e74c3c",
                alpha = 0.2, linewidth = 0.6) +
    labs(
      title = sprintf("%s GAMM (updated) — Residual Plot (seed %d)", model_tag, seed),
      x = "Predicted Values",
      y = "Residuals (Actual − Predicted)"
    ) +
    theme_minimal(base_size = 13)
  
  resid_filename <- sprintf("%s_GAMM_updated_residual_seed_%d.png", model_tag, seed)
  ggsave(resid_filename, p_resid, width = 8, height = 7, dpi = 300)
  cat(sprintf("  Saved: %s\n", resid_filename))
  
  return(list(r2 = r2, r2_cor2 = r2_cor2, rmse = rmse, mae = mae,
              spearman_rho = spearman_r, pearson_r = pearson_r))
}


plot_gamm_random_effect <- function(model, model_tag, seed) {
  # Find which smooth term is the random effect
  smooth_names <- names(model$smooth)
  re_index <- which(sapply(model$smooth, function(s) s$label) == "s(cmpair)")
  
  if (length(re_index) == 0) {
    # Fallback: use the last smooth term
    re_index <- length(model$smooth)
  }
  
  filename <- sprintf("%s_GAMM_updated_random_effect_seed_%d.png", model_tag, seed)
  png(filename, width = 1600, height = 1200, res = 300)
  plot(model, select = re_index,
       main = sprintf("%s GAMM (updated) — Twin Random Effect (seed %d)", model_tag, seed))
  dev.off()
  cat(sprintf("  Saved: %s\n", filename))
}


plot_gamm_smooth_terms <- function(model, model_tag, seed) {
  # Plot all PRS smooth terms (exclude the random effect)
  n_smooth <- length(model$smooth)
  prs_indices <- which(sapply(model$smooth, function(s) grepl("^s\\(PRS", s$label)))
  
  if (length(prs_indices) > 0) {
    filename <- sprintf("%s_GAMM_updated_PRS_smooth_terms_seed_%d.png", model_tag, seed)
    n_plots <- length(prs_indices)
    ncol <- min(n_plots, 3)
    nrow <- ceiling(n_plots / ncol)
    
    png(filename, width = 800 * ncol, height = 700 * nrow, res = 300)
    par(mfrow = c(nrow, ncol))
    for (idx in prs_indices) {
      plot(model, select = idx, shade = TRUE, shade.col = "lightblue",
           main = model$smooth[[idx]]$label)
    }
    dev.off()
    cat(sprintf("  Saved: %s\n", filename))
  }
}


main <- function() {
  # ── Load and clean data ──
  catss <- read_csv("catss_final_data.csv", show_col_types = FALSE)
  catss <- catss[complete.cases(catss), ]
  catss$cmpair <- as.character(catss$cmpair)
  cat(sprintf("Complete cases: %d\n", nrow(catss)))
  
  spec <- build_model_spec(names(catss))
  cat("Variables per modality:\n")
  print(sapply(c(spec$score_blocks, list(PRS = spec$PRS_vars, SEX = spec$Sex_var, batch_x_PC = spec$BatchPC_vars)), length))
  cat(sprintf("Interaction terms: %d\n", length(spec$interaction_terms)))
  pos_formula <- spec$pos_formula
  neg_formula <- spec$neg_formula
  
  
  # ══════════════════════════════════════════
  # Multi-seed evaluation
  # ══════════════════════════════════════════
  
  seeds <- c(42, 43, 44, 45, 46)
  
  pos_results <- list()
  neg_results <- list()
  
  for (i in seq_along(seeds)) {
    cat(sprintf("\n══ Seed %d ══\n", seeds[i]))
    
    # Split (shared with DCMFNet and the ML benchmark)
    split <- read_split(seeds[i], catss)
    test_ids <- split$cmpair[split$split == "test"]
    
    test_df  <- catss[catss$cmpair %in% test_ids, ]
    train_df <- catss[!catss$cmpair %in% test_ids, ]
    
    scored <- add_interaction_scores(train_df, test_df, spec)
    train_df <- scored$train
    test_df  <- scored$test
    train_df$cmpair <- as.factor(train_df$cmpair)
    
    cat(sprintf("  Train: %d rows (%d IDs), Test: %d rows (%d IDs)\n",
                nrow(train_df), length(unique(train_df$cmpair)),
                nrow(test_df), length(unique(test_df$cmpair))))
    
    # Test twin pairs are never in training: predict with the random effect excluded (population level).
    # cmpair is set to a training level only so that predict() accepts the factor; s(cmpair) is excluded.
    test_df$cmpair <- factor(levels(train_df$cmpair)[1], levels = levels(train_df$cmpair))
    
    # ── Positive GAMM ──
    cat("  Fitting Positive GAMM...\n")
    pos_model <- gam(pos_formula, data = train_df, family = gaussian(), method = "REML")
    
    # Save coefficients and smooth plots only for first seed
    if (i == 1) {
      cat("  Extracting Positive coefficients...\n")
      pos_coef_tables <- extract_gamm_coefs(pos_model, "Pos")
      plot_gamm_random_effect(pos_model, "Pos", seeds[i])
      plot_gamm_smooth_terms(pos_model, "Pos", seeds[i])
    }
    
    pos_pred <- as.numeric(predict(pos_model, newdata = test_df, type = "response", exclude = "s(cmpair)"))
    pos_metrics <- plot_predictions(test_df$SCZ18_Pos_Norm, pos_pred, "Pos", seeds[i])
    pos_metrics$seed <- seeds[i]
    pos_results[[i]] <- as.data.frame(pos_metrics)
    
    cat(sprintf("  Pos — RMSE: %.4f, R²: %.4f, Spearman: %.4f\n",
                pos_metrics$rmse, pos_metrics$r2, pos_metrics$spearman_rho))
    
    # ── Negative GAMM ──
    cat("  Fitting Negative GAMM...\n")
    neg_model <- gam(neg_formula, data = train_df, family = gaussian(), method = "REML")
    
    if (i == 1) {
      cat("  Extracting Negative coefficients...\n")
      neg_coef_tables <- extract_gamm_coefs(neg_model, "Neg")
      plot_gamm_random_effect(neg_model, "Neg", seeds[i])
      plot_gamm_smooth_terms(neg_model, "Neg", seeds[i])
    }
    
    neg_pred <- as.numeric(predict(neg_model, newdata = test_df, type = "response", exclude = "s(cmpair)"))
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
  
  # Population SD (ddof = 0), matching np.std in the Python summaries
  pop_sd <- function(x) sqrt(mean((x - mean(x))^2))
  
  summarise_results <- function(df, model_tag) {
    metrics <- c("rmse", "mae", "r2", "r2_cor2", "spearman_rho", "pearson_r")
    row <- data.frame(Model = model_tag)
    for (m in metrics) {
      row[[paste0(m, "_mean")]] <- mean(df[[m]])
      row[[paste0(m, "_std")]]  <- pop_sd(df[[m]])
    }
    return(row)
  }
  
  gamm_summary <- bind_rows(
    summarise_results(pos_df, "Pos_GAMM_updated"),
    summarise_results(neg_df, "Neg_GAMM_updated")
  )
  
  cat("\n══════════════════════════════════════════\n")
  cat("  Updated GAMM Summary across", length(seeds), "seeds\n")
  cat("══════════════════════════════════════════\n")
  print(gamm_summary)
  
  # Save everything
  write_csv(gamm_summary, "GAMM_updated_summary.csv")
  write_csv(pos_df, "GAMM_updated_Pos_per_seed.csv")
  write_csv(neg_df, "GAMM_updated_Neg_per_seed.csv")
  
  cat("\nSaved: GAMM_updated_summary.csv\n")
  cat("Saved: GAMM_updated_Pos_per_seed.csv\n")
  cat("Saved: GAMM_updated_Neg_per_seed.csv\n")
}


# Run only when executed with Rscript, not when sourced (e.g., by tests)
if (sys.nframe() == 0) {
  main()
}
