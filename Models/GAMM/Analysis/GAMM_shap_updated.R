## SHAP and grouped permutation importance per modality for the updated GAMM
## Companion of Models/GLMM/Analysis/GLMM_shap_updated.R and Models/DCMFNet/Analysis/shap_analysis.py, so the
## GAMM, GLMM and DCMFNet attributions can be compared (attribution_agreement.py). Date: 2026-10-24
##
## The model is the one fitted by GAMM_updated_split.R (same splits, same main effects and interactions, PRS main
## effects as smooth terms); its functions are reused by sourcing that script. Attributions use the
## population-level prediction (s(cmpair) excluded), which is the prediction evaluated on the test set.
##
## SHAP: exact interventional Shapley values per modality, with the training set as the reference distribution.
## The GAMM is additive, so the same decomposition as for the GLMM applies:
##   - Linear main effect of item j in modality m: beta_j * (x_j - mean_train(x_j)), added to modality m.
##   - Smooth main effect s(PRS_k): f_k(x) - mean_train(f_k), added to PRS.
##   - Interaction gamma * S * P, where S = SUD15 score and P = partner (modality score, SEX or one PRS):
##       SUD15   gets gamma/2 * (S - S_bar) * (P + P_bar)
##       partner gets gamma/2 * (P - P_bar) * (S + S_bar)
##   - Modality importance = mean over test rows of |SHAP of the modality| (same as for DCMFNet and the GLMM).
## Permutation: all items of one modality are shuffled together in the test set, the modality scores are
## recomputed from the shuffled items (with training statistics), and the drop in test R² is recorded.
##
## Usage: run from the folder containing catss_final_data.csv and the split_seed_<seed>.csv files
## (the same folder as GAMM_updated_split.R).
##   Rscript GAMM_shap_updated.R
## Output: GAMM_attribution_per_seed.csv (aggregated over test rows; no participant-level values are saved)
##
## The attributions describe what the fitted model relies on for its predictions. They are associations
## within the model, not causal effects of the predictors.

library(readr)
library(mgcv)
library(dplyr)

# ── Reuse the model specification and helpers from GAMM_updated_split.R (main() is not run when sourced) ──
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) dirname(sub("^--file=", "", script_arg[1])) else "."
candidates <- c("GAMM_updated_split.R", file.path(script_dir, "GAMM_updated_split.R"))
model_script <- candidates[file.exists(candidates)][1]
if (is.na(model_script)) stop("GAMM_updated_split.R not found in the working folder or in script dir")
source(model_script)

SEEDS <- c(42, 43, 44, 45, 46)
N_PERMUTATION_REPEATS <- 10

# Same modality labels as DCMFNet (importance_analysis_avg.py MODALITY_LABELS)
MODALITY_LABELS <- c("SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX", "batch*PC")


# Items of each modality, named with the DCMFNet labels
modality_items <- function(spec) {
  blocks <- spec$score_blocks
  list(
    "SUD15" = blocks$SUD15, "PRS" = spec$PRS_vars, "SCZ15" = blocks$SCZ15, "ADHD9" = blocks$ADHD9,
    "ASD9" = blocks$ASD9, "ACE15" = blocks$ACE15, "ACE18" = blocks$ACE18, "SUD18" = blocks$SUD18,
    "SES" = blocks$SES, "SEX" = spec$Sex_var, "batch*PC" = spec$BatchPC_vars
  )
}

# Modality of the partner in an interaction term (e.g. "SCZ15_score", "SEX", "PRS_Astrocyte_z")
partner_modality <- function(partner, spec) {
  if (partner %in% spec$Sex_var) return("SEX")
  if (grepl("^PRS", partner)) return("PRS")
  return(sub("_score$", "", partner))
}

# Coefficient of an interaction, whichever order mgcv used in its name
interaction_coef <- function(beta, term) {
  if (term %in% names(beta)) return(beta[[term]])
  parts <- strsplit(term, ":")[[1]]
  reversed <- paste(rev(parts), collapse = ":")
  if (reversed %in% names(beta)) return(beta[[reversed]])
  stop(sprintf("Coefficient for %s not found", term))
}

# Test twin pairs are never in training: cmpair is set to a training level only so that predict() accepts the
# factor, and s(cmpair) is excluded (as in GAMM_updated_split.R)
with_training_cmpair <- function(df, train_levels) {
  df$cmpair <- factor(train_levels[1], levels = train_levels)
  df
}

predict_population <- function(model, df) {
  as.numeric(predict(model, newdata = df, type = "response", exclude = "s(cmpair)"))
}


# Exact modality-level SHAP values for the population-level prediction. Returns an n_test x 11 matrix.
gamm_modality_shap <- function(model, train_df, test_df, spec) {
  beta <- coef(model)
  items <- modality_items(spec)
  phi <- matrix(0, nrow = nrow(test_df), ncol = length(MODALITY_LABELS),
                dimnames = list(NULL, MODALITY_LABELS))

  # Linear main effects (everything except the PRS, which are smooth terms)
  for (m in setdiff(names(items), "PRS")) {
    for (v in items[[m]]) {
      if (!v %in% names(beta)) stop(sprintf("Coefficient for %s not found", v))
      phi[, m] <- phi[, m] + beta[[v]] * (test_df[[v]] - mean(train_df[[v]]))
    }
  }

  # Smooth PRS main effects: fitted curve on the test rows minus its mean over the training rows
  smooth_labels <- paste0("s(", spec$PRS_vars, ")")
  f_test  <- predict(model, newdata = test_df,  type = "terms", terms = smooth_labels)
  f_train <- predict(model, newdata = train_df, type = "terms", terms = smooth_labels)
  for (label in smooth_labels) {
    phi[, "PRS"] <- phi[, "PRS"] + f_test[, label] - mean(f_train[, label])
  }

  # Interactions SUD15_score x partner
  S <- test_df$SUD15_score
  S_bar <- mean(train_df$SUD15_score)
  for (term in spec$interaction_terms) {
    partner <- setdiff(strsplit(term, ":")[[1]], "SUD15_score")
    gamma <- interaction_coef(beta, term)
    P <- test_df[[partner]]
    P_bar <- mean(train_df[[partner]])
    m <- partner_modality(partner, spec)
    phi[, "SUD15"] <- phi[, "SUD15"] + gamma / 2 * (S - S_bar) * (P + P_bar)
    phi[, m]       <- phi[, m]       + gamma / 2 * (P - P_bar) * (S + S_bar)
  }

  # Check: SHAP values add up to f(x) minus a constant (the expected prediction under the reference)
  pred <- predict_population(model, test_df)
  offset <- pred - rowSums(phi)
  cat(sprintf("    SHAP additivity check: SD of f(x) - sum(SHAP) = %.2e (should be ~0)\n", sd(offset)))
  phi
}


r2_score <- function(y, y_hat) 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2)


# Drop in test R² when all items of one modality are shuffled together (scores recomputed afterwards)
gamm_permutation_importance <- function(model, train_raw, test_raw, spec, outcome, seed, train_levels) {
  set.seed(seed)
  items <- modality_items(spec)
  score_test <- function(test_data) {
    with_training_cmpair(add_interaction_scores(train_raw, test_data, spec)$test, train_levels)
  }
  baseline_r2 <- r2_score(test_raw[[outcome]], predict_population(model, score_test(test_raw)))

  importance <- sapply(MODALITY_LABELS, function(m) {
    drops <- replicate(N_PERMUTATION_REPEATS, {
      test_perm <- test_raw
      test_perm[, items[[m]]] <- test_raw[sample(nrow(test_raw)), items[[m]]]
      baseline_r2 - r2_score(test_raw[[outcome]], predict_population(model, score_test(test_perm)))
    })
    mean(drops)
  })
  list(importance = importance, baseline_r2 = baseline_r2)
}


main_shap <- function() {
  catss <- read_csv("catss_final_data.csv", show_col_types = FALSE)
  catss <- catss[complete.cases(catss), ]
  catss$cmpair <- as.character(catss$cmpair)
  spec <- build_model_spec(names(catss))

  rows <- list()
  for (seed in SEEDS) {
    cat(sprintf("\n══ Seed %d ══\n", seed))
    split <- read_split(seed, catss)
    test_ids <- split$cmpair[split$split == "test"]
    train_raw <- catss[!catss$cmpair %in% test_ids, ]
    test_raw  <- catss[catss$cmpair %in% test_ids, ]

    scored <- add_interaction_scores(train_raw, test_raw, spec)
    train_df <- scored$train
    train_df$cmpair <- as.factor(train_df$cmpair)
    train_levels <- levels(train_df$cmpair)
    test_df <- with_training_cmpair(scored$test, train_levels)

    for (model_tag in c("Pos", "Neg")) {
      outcome <- if (model_tag == "Pos") "SCZ18_Pos_Norm" else "SCZ18_Neg_Norm"
      formula <- if (model_tag == "Pos") spec$pos_formula else spec$neg_formula
      cat(sprintf("  Fitting %s GAMM...\n", model_tag))
      model <- gam(formula, data = train_df, family = gaussian(), method = "REML")

      phi <- gamm_modality_shap(model, train_df, test_df, spec)
      shap_importance <- colMeans(abs(phi))

      perm <- gamm_permutation_importance(model, train_raw, test_raw, spec, outcome, seed, train_levels)
      cat(sprintf("    Test R²: %.4f\n", perm$baseline_r2))

      rows[[length(rows) + 1]] <- data.frame(model_tag = model_tag, seed = seed, source = "GAMM_SHAP",
                                             modality = MODALITY_LABELS, importance = shap_importance[MODALITY_LABELS])
      rows[[length(rows) + 1]] <- data.frame(model_tag = model_tag, seed = seed, source = "GAMM_permutation",
                                             modality = MODALITY_LABELS, importance = perm$importance[MODALITY_LABELS])
    }
  }

  attribution <- bind_rows(rows)
  write_csv(attribution, "GAMM_attribution_per_seed.csv")
  cat("\nSaved: GAMM_attribution_per_seed.csv\n")
}

if (sys.nframe() == 0) {
  main_shap()
}
