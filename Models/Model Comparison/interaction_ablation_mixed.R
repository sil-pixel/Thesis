## Interaction ablation for the GLMM and GAMM: do the 24 SUD15 interaction terms improve prediction?
## Date: 2026-10-29
## Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
##
## For each model (GLMM, GAMM), split (seeds 42-46) and outcome, the model is fitted twice on the same training set:
##   with    : the specification of GLMM_updated_split.R / GAMM_updated_split.R (105 main effects + 24 interactions)
##   without : the same specification with the 24 SUD15 interaction terms removed
## and evaluated on the same test set. Reported per split:
##   - test R² and RMSE with and without the interactions;
##   - a joint Wald test of the 24 interaction coefficients in the fit with interactions (chi-square, 24 df);
##   - GLMM only: a likelihood-ratio test of the two fits (lme4 refits both with ML for the test).
## The paired R² differences over the five splits are tested with the corrected resampled t-test in
## dcmfnet_fusion_ablation.py, together with the DCMFNet fusion ablation (one Holm family).
##
## Usage: run from the folder containing catss_final_data.csv and the split_seed_<seed>.csv files,
## with GLMM_updated_split.R and GAMM_updated_split.R in the same folder or in their repository folders
## (the same set-up as glmm_gamm_test_predictions.R).
##   Rscript interaction_ablation_mixed.R
##
## Output (aggregated only; can be exported): interaction_ablation_mixed_per_seed.csv
## The file is rewritten after every fit, so an interrupted run keeps the finished fits.

library(readr)
library(dplyr)

RUN_GAMM <- TRUE             # the GAMM is slow; set to FALSE to run the GLMM only
SEEDS <- c(42, 43, 44, 45, 46)
OUTCOMES <- c(Pos = "SCZ18_Pos_Norm", Neg = "SCZ18_Neg_Norm")
OUT_FILE <- "interaction_ablation_mixed_per_seed.csv"


# ── Load the model scripts into separate environments (their main() is not run when sourced) ──
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) dirname(sub("^--file=", "", script_arg[1])) else "."

load_model_script <- function(file_name, repo_folder) {
  candidates <- c(file_name, file.path(script_dir, "..", repo_folder, "Method", file_name),
                  file.path(script_dir, "..", repo_folder, file_name))
  path <- candidates[file.exists(candidates)][1]
  if (is.na(path)) stop(sprintf("%s not found in the working folder or in ../%s/", file_name, repo_folder))
  env <- new.env()
  sys.source(path, envir = env)
  env
}

model_envs <- list(GLMM = load_model_script("GLMM_updated_split.R", "GLMM"))
if (RUN_GAMM) model_envs$GAMM <- load_model_script("GAMM_updated_split.R", "GAMM")


# Formula without the interaction terms, built from the same pieces as build_model_spec()
formula_without_interactions <- function(env, spec, outcome) {
  main_effect_terms <- paste(env$MAIN_EFFECT_TERMS(spec$main_effect_vars, spec$PRS_vars), collapse = " + ")
  as.formula(paste(outcome, "~", main_effect_terms, env$RANDOM_EFFECT_TERM))
}

fit_model <- function(model_name, formula, train_df) {
  if (model_name == "GLMM") return(lme4::lmer(formula, data = train_df))
  mgcv::gam(formula, data = train_df, family = gaussian(), method = "REML")
}

# Population-level test predictions, as in glmm_gamm_test_predictions.R
predict_test <- function(model_name, model, train_df, test_df) {
  if (model_name == "GLMM") return(as.numeric(predict(model, newdata = test_df, re.form = NA)))
  test_df$cmpair <- factor(levels(train_df$cmpair)[1], levels = levels(train_df$cmpair))
  as.numeric(predict(model, newdata = test_df, type = "response", exclude = "s(cmpair)"))
}

# Joint Wald test of the SUD15 interaction coefficients (names such as SUD15_score:ADHD9_score or SEX:SUD15_score)
joint_wald <- function(model_name, model, n_expected) {
  b <- if (model_name == "GLMM") lme4::fixef(model) else coef(model)
  V <- as.matrix(vcov(model))
  terms <- names(b)[grepl(":", names(b)) & grepl("SUD15_score", names(b))]
  if (length(terms) != n_expected) {
    stop(sprintf("Found %d SUD15 interaction coefficients, expected %d", length(terms), n_expected))
  }
  w <- as.numeric(t(b[terms]) %*% solve(V[terms, terms], b[terms]))
  list(chisq = w, df = length(terms), p = pchisq(w, df = length(terms), lower.tail = FALSE))
}

test_metrics <- function(y, p) {
  c(r2 = 1 - sum((y - p)^2) / sum((y - mean(y))^2), rmse = sqrt(mean((y - p)^2)))
}


catss <- read_csv("catss_final_data.csv", show_col_types = FALSE)
catss <- catss[complete.cases(catss), ]
catss$cmpair <- as.character(catss$cmpair)
cat(sprintf("Complete cases: %d\n", nrow(catss)))

results <- list()
for (model_name in names(model_envs)) {
  env <- model_envs[[model_name]]
  spec <- env$build_model_spec(names(catss))
  n_int <- length(spec$interaction_terms)
  cat(sprintf("\n%s: %d main effects, %d interactions\n", model_name, length(spec$main_effect_vars), n_int))

  for (seed in SEEDS) {
    split <- env$read_split(seed, catss)
    test_ids <- split$cmpair[split$split == "test"]
    scored <- env$add_interaction_scores(catss[!catss$cmpair %in% test_ids, ],
                                         catss[catss$cmpair %in% test_ids, ], spec)
    train_df <- scored$train
    test_df <- scored$test
    train_df$cmpair <- as.factor(train_df$cmpair)

    for (tag in names(OUTCOMES)) {
      outcome <- OUTCOMES[[tag]]
      formula_with <- if (tag == "Pos") spec$pos_formula else spec$neg_formula
      formula_without <- formula_without_interactions(env, spec, outcome)
      cat(sprintf("  Seed %d, %s: fitting %s with and without interactions...\n", seed, tag, model_name))

      fit_with <- fit_model(model_name, formula_with, train_df)
      fit_without <- fit_model(model_name, formula_without, train_df)
      y <- test_df[[outcome]]
      m_with <- test_metrics(y, predict_test(model_name, fit_with, train_df, test_df))
      m_without <- test_metrics(y, predict_test(model_name, fit_without, train_df, test_df))
      wald <- joint_wald(model_name, fit_with, n_int)

      lrt <- list(chisq = NA_real_, df = NA_real_, p = NA_real_)
      if (model_name == "GLMM") {
        a <- anova(fit_without, fit_with, refit = TRUE)    # both refitted with ML
        lrt <- list(chisq = a$Chisq[2], df = a$Df[2], p = a$`Pr(>Chisq)`[2])
      }

      cat(sprintf("    Test R² with %.4f, without %.4f (difference %+.4f); Wald p = %.3g%s\n",
                  m_with[["r2"]], m_without[["r2"]], m_with[["r2"]] - m_without[["r2"]], wald$p,
                  if (is.na(lrt$p)) "" else sprintf(", LRT p = %.3g", lrt$p)))

      results[[length(results) + 1]] <- data.frame(
        model = model_name, outcome = tag, seed = seed,
        n_train = nrow(train_df), n_test = nrow(test_df), n_interactions = n_int,
        r2_with = m_with[["r2"]], r2_without = m_without[["r2"]],
        rmse_with = m_with[["rmse"]], rmse_without = m_without[["rmse"]],
        r2_difference = m_with[["r2"]] - m_without[["r2"]],
        wald_chisq = wald$chisq, wald_df = wald$df, wald_p = wald$p,
        lrt_chisq = lrt$chisq, lrt_df = lrt$df, lrt_p = lrt$p
      )
      write_csv(bind_rows(results), OUT_FILE)
    }
  }
}
cat(sprintf("\nSaved: %s (aggregated; can be exported)\n", OUT_FILE))
