# title: "GLMM model for Psychotic Risk prediction from SUbstance Use and other factors in CATSS across 5 different seeds"
# author: "Silpa Soni Nallacheruvu"
# date: "2026-04-25"

library(readr)
library(lme4)
library(dplyr)

catss <- read_csv("catss_final_data.csv")
catss <- catss[complete.cases(catss), ]


# get a list of columns with their prefix
SUD15_vars <- grep("^SUD15", names(catss), value = TRUE)
SCZ15_vars <- grep("^SCZ15", names(catss), value = TRUE)
PRS_vars <- grep("^PRS", names(catss), value = TRUE)
ADHD9_vars <- grep("^ADHD9", names(catss), value = TRUE)
ASD9_vars <- grep("^ASD9", names(catss), value = TRUE)
ACE_vars <- grep("^ACE", names(catss), value = TRUE)
SES_vars <- grep("^SES", names(catss), value = TRUE)
PCA_vars <- grep("^PC", names(catss), value = TRUE)[1:2]
Sex_var <- grep("^SEX", names(catss), value = TRUE)
Batch_var <- grep("^batch", names(catss), value = TRUE)[1]

factor_terms <- paste(
  c(
    paste0("factor(", SUD15_vars, ")")
  ),
  collapse = " + "
)


covariate_terms <- paste(
  c(SUD15_vars, SCZ15_vars, ADHD9_vars, ASD9_vars, ACE_vars, SES_vars, PRS_vars, Sex_var),
  collapse = " + "
)

# PCA terms according to the genetic batch
genetic_confounding_terms <- paste(
  c(
    paste0("factor(", Batch_var, ") : ", PCA_vars)
  ),
  collapse = " + "
)

# create interaction terms between SUD15 variables and all other covariates
factor_interaction_terms <- paste(
  c(
    paste0("factor(", SUD15_vars, "):(", SCZ15_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", ADHD9_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", ASD9_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", ACE_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", SES_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", PRS_vars, ")"),
    paste0("factor(", SUD15_vars, "):(", Sex_var, ")")
  ),
  collapse = " + "
)

interaction_terms <- paste(
  c(
    paste0(SUD15_vars ,":", SCZ15_vars),
    paste0(SUD15_vars, ":", ADHD9_vars),
    paste0(SUD15_vars, ":", ASD9_vars),
    paste0(SUD15_vars, ":", ACE_vars),
    paste0(SUD15_vars, ":", SES_vars),
    paste0(SUD15_vars, ":", PRS_vars),
    paste0(SUD15_vars, ":", Sex_var)
  ),
  collapse = " + "
)

pos_glmm_formula_str <- paste(
  "SCZ18_Pos_Norm ~",
  covariate_terms, "+",
  interaction_terms, "+",
  genetic_confounding_terms,  "+",
  "(1 | cmpair)"
)

pos_glmm_formula <- as.formula(pos_glmm_formula_str)

neg_glmm_formula_str <- paste(
  "SCZ18_Neg_Norm ~",
  covariate_terms, "+",
  interaction_terms, "+",
  genetic_confounding_terms,  "+",
  "(1 | cmpair)"
)

neg_glmm_formula <- as.formula(neg_glmm_formula_str)



seeds <- c(42, 43, 44, 45, 46)
test_size <- 0.25

pos_results <- list()
neg_results <- list()

for (i in seq_along(seeds)) {
  set.seed(seeds[i])
  
  unique_ids <- unique(catss$cmpair)
  n_test <- floor(length(unique_ids) * test_size)
  test_ids <- sample(unique_ids, size = n_test)
  
  test_df <- catss[catss$cmpair %in% test_ids, ]
  train_df <- catss[!catss$cmpair %in% test_ids, ]
  train_df$cmpair <- as.factor(train_df$cmpair)
  
  # ── Positive model ──
  pos_model <- glmer(pos_glmm_formula, data = train_df)
  pos_pred <- predict(pos_model, newdata = test_df, 
                      allow.new.levels = TRUE, type = "response")
  pos_true <- test_df$SCZ18_Pos_Norm
  
  pos_results[[i]] <- data.frame(
    seed = seeds[i],
    rmse = sqrt(mean((pos_true - pos_pred)^2)),
    mae = mean(abs(pos_true - pos_pred)),
    r2 = cor(pos_true, pos_pred)^2,
    spearman_rho = cor(pos_true, pos_pred, method = "spearman"),
    pearson_r = cor(pos_true, pos_pred)
  )
  
  # ── Negative model ──
  neg_model <- glmer(neg_glmm_formula, data = train_df)
  neg_pred <- predict(neg_model, newdata = test_df, 
                      allow.new.levels = TRUE, type = "response")
  neg_true <- test_df$SCZ18_Neg_Norm
  
  neg_results[[i]] <- data.frame(
    seed = seeds[i],
    rmse = sqrt(mean((neg_true - neg_pred)^2)),
    mae = mean(abs(neg_true - neg_pred)),
    r2 = cor(neg_true, neg_pred)^2,
    spearman_rho = cor(neg_true, neg_pred, method = "spearman"),
    pearson_r = cor(neg_true, neg_pred)
  )
  
  cat(sprintf("Seed %d done\n", seeds[i]))
}

# ── Summary ──
pos_df <- bind_rows(pos_results)
neg_df <- bind_rows(neg_results)

summarise_results <- function(df, model_tag) {
  metrics <- c("rmse", "mae", "r2", "spearman_rho", "pearson_r")
  summary_row <- data.frame(Model = model_tag)
  for (m in metrics) {
    summary_row[[paste0(m, "_mean")]] <- mean(df[[m]])
    summary_row[[paste0(m, "_std")]] <- sd(df[[m]])
  }
  return(summary_row)
}

pos_summary <- summarise_results(pos_df, "Pos_GLMM")
neg_summary <- summarise_results(neg_df, "Neg_GLMM")
glmm_summary <- bind_rows(pos_summary, neg_summary)

print(glmm_summary)
write.csv(glmm_summary, "glmm_summary.csv", row.names = FALSE)
write.csv(pos_df, "glmm_pos_per_seed.csv", row.names = FALSE)
write.csv(neg_df, "glmm_neg_per_seed.csv", row.names = FALSE)