## Test-set predictions of the updated GLMM and GAMM for the prediction-level evaluation
## (prediction_evaluation.py): calibration, sex-stratified and top-decile metrics, paired comparisons,
## and the age-15-only sensitivity analysis.
## Date: 2026-10-23
## Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
##
## The models are the ones in GLMM_updated_split.R and GAMM_updated_split.R (same splits, same specification);
## both scripts are sourced into separate environments, so their functions are reused as they are.
## Predictor sets:
##   full  : all DCMFNet inputs (the main analysis)
##   age15 : without the age-18 predictors (ACE18, SUD18), which are measured at the same age as the outcome.
##           Fitted on the same complete-case rows and splits as the full models.
##
## Usage: run from the folder containing catss_final_data.csv and the split_seed_<seed>.csv files.
## GLMM_updated_split.R and GAMM_updated_split.R must be in the same folder or in their repository folders.
##   Rscript glmm_gamm_test_predictions.R
##
## Output: SERVER_ONLY_predictions_GLMM.csv and SERVER_ONLY_predictions_GAMM.csv
## These files hold one row per test participant (observed and predicted outcome, sex). They are
## participant-level data: keep them on the secure server and export only the outputs of prediction_evaluation.py.
## The files are rewritten after every fit, so an interrupted run keeps the finished fits.

library(readr)
library(dplyr)

RUN_GAMM <- TRUE             # the GAMM is slow; set to FALSE to run the GLMM only
PREDICTOR_SETS <- c("full", "age15")
SEEDS <- c(42, 43, 44, 45, 46)
OUTCOMES <- c(Pos = "SCZ18_Pos_Norm", Neg = "SCZ18_Neg_Norm")
AGE18_PATTERN <- "^(ACE18|SUD18)"


# ── Load the model scripts into separate environments (their main() is not run when sourced) ──
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg) > 0) dirname(sub("^--file=", "", script_arg[1])) else "."

load_model_script <- function(file_name, repo_folder) {
  candidates <- c(file_name, file.path(script_dir, "..", repo_folder, "Method", file_name))
  path <- candidates[file.exists(candidates)][1]
  if (is.na(path)) stop(sprintf("%s not found in the working folder or in ../%s/Method", file_name, repo_folder))
  env <- new.env()
  sys.source(path, envir = env)
  env
}

model_envs <- list(GLMM = load_model_script("GLMM_updated_split.R", "GLMM"))
if (RUN_GAMM) model_envs$GAMM <- load_model_script("GAMM_updated_split.R", "GAMM")


# Fit one model and return its population-level test predictions (as in the *_updated_split.R scripts)
fit_and_predict <- function(model_name, formula, train_df, test_df) {
  if (model_name == "GLMM") {
    model <- lme4::lmer(formula, data = train_df)
    return(as.numeric(predict(model, newdata = test_df, re.form = NA)))
  }
  # GAMM: cmpair is set to a training level only so that predict() accepts the factor; s(cmpair) is excluded
  test_df$cmpair <- factor(levels(train_df$cmpair)[1], levels = levels(train_df$cmpair))
  model <- mgcv::gam(formula, data = train_df, family = gaussian(), method = "REML")
  as.numeric(predict(model, newdata = test_df, type = "response", exclude = "s(cmpair)"))
}


catss <- read_csv("catss_final_data.csv", show_col_types = FALSE)
catss <- catss[complete.cases(catss), ]
catss$cmpair <- as.character(catss$cmpair)
catss$row_id <- seq_len(nrow(catss)) - 1      # 0-based position in the complete-case data (as pandas after dropna)
cat(sprintf("Complete cases: %d\n", nrow(catss)))

sex_col <- grep("^SEX", names(catss), value = TRUE)[1]
age18_cols <- grep(AGE18_PATTERN, names(catss), value = TRUE)
predictor_columns <- list(
  full = setdiff(names(catss), "row_id"),
  age15 = setdiff(names(catss), c("row_id", age18_cols))
)
cat(sprintf("Age-15-only set removes %d columns: %s\n", length(age18_cols), paste(age18_cols, collapse = ", ")))

for (model_name in names(model_envs)) {
  env <- model_envs[[model_name]]
  out_file <- sprintf("SERVER_ONLY_predictions_%s.csv", model_name)
  results <- list()

  for (predictor_set in PREDICTOR_SETS) {
    spec <- env$build_model_spec(predictor_columns[[predictor_set]])
    cat(sprintf("\n%s, %s predictors: %d main effects, %d interactions\n", model_name, predictor_set,
                length(spec$main_effect_vars), length(spec$interaction_terms)))

    for (seed in SEEDS) {
      split <- env$read_split(seed, catss)
      test_ids <- split$cmpair[split$split == "test"]
      scored <- env$add_interaction_scores(catss[!catss$cmpair %in% test_ids, ],
                                           catss[catss$cmpair %in% test_ids, ], spec)
      train_df <- scored$train
      test_df <- scored$test
      train_df$cmpair <- as.factor(train_df$cmpair)

      for (tag in names(OUTCOMES)) {
        formula <- if (tag == "Pos") spec$pos_formula else spec$neg_formula
        cat(sprintf("  Seed %d, %s: fitting %s...\n", seed, tag, model_name))
        y_pred <- fit_and_predict(model_name, formula, train_df, test_df)
        y_true <- test_df[[OUTCOMES[[tag]]]]
        r2 <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
        cat(sprintf("    Test R²: %.4f\n", r2))

        results[[length(results) + 1]] <- data.frame(
          model = model_name, predictor_set = predictor_set, outcome = tag, seed = seed,
          row_id = test_df$row_id, y_true = y_true, y_pred = y_pred, sex = test_df[[sex_col]]
        )
        write_csv(bind_rows(results), out_file)
      }
    }
  }
  cat(sprintf("Saved: %s (keep on the secure server)\n", out_file))
}
