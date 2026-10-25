'''
Test-set predictions of three machine-learning comparators for the prediction-level evaluation (prediction_evaluation.py),
so that they are scored with the same metric code, calibration, sex, top-decile and paired comparisons as DCMFNet,
GLMM and GAMM.
Date: 25/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

The comparators were chosen by model class, after the default-settings screening of 40 regressors (lazy_predict.py),
not by their rank in that screening:
    LinearRegression : unpenalised linear model, no interactions, nothing to tune (reference).
    ElasticNet       : penalised linear model (lasso and ridge are special cases). Inputs standardised;
                       penalty strength and L1 ratio chosen by inner cross-validation (ElasticNetCV).
    GradientBoosting : gradient-boosted trees (HistGradientBoostingRegressor), which learn non-linear effects and
                       interactions without a designed architecture. Random search over 40 configurations.
All tuning uses the training split only, with 5 inner folds grouped by twin pair (GroupKFold on cmpair), and
mean squared error as the criterion. Each model is then refitted on the full training split and evaluated once on
the test split.

Predictor sets (same as glmm_gamm_test_predictions.R and dcmfnet_test_predictions.py):
    full  : all 105 inputs used by DCMFNet and lazy_predict.py.
    age15 : without the age-18 predictors (ACE18, SUD18); the columns are dropped, as in the mixed models.

Usage: run from the folder containing catss_final_data.csv and the split_seed_<seed>.csv files (from split_data.py).
    python ml_comparators_test_predictions.py

Output:
    SERVER_ONLY_predictions_ML.csv : one row per test participant (observed and predicted outcome, sex).
        Participant-level data: keep it on the secure server and export only the outputs of prediction_evaluation.py.
        The file is rewritten after every model, so an interrupted run keeps the finished models.
    ML_comparators_tuning.csv      : selected hyperparameters and inner-CV MSE per model, predictor set, outcome
        and seed (aggregated; can be exported).
'''

import re
import json

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44, 45, 46]
PREDICTOR_SETS = ["full", "age15"]
OUTCOMES = [("Pos", "SCZ18_Pos_Norm"), ("Neg", "SCZ18_Neg_Norm")]
AGE18_PATTERN = r"^(ACE18|SUD18)"
N_INNER_FOLDS = 5
N_BOOSTING_CONFIGS = 40
OUT_FILE = "SERVER_ONLY_predictions_ML.csv"
TUNING_FILE = "ML_comparators_tuning.csv"

# Same input columns as get_input_output_cols() in lazy_predict.py
INPUT_MODALITIES = ["SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX"]
BATCH_PC_PATTERN = r"^batch_.*_x_PC"

ELASTIC_NET_L1_RATIOS = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
BOOSTING_SPACE = {
    'learning_rate': loguniform(0.01, 0.3),
    'max_iter': randint(100, 501),
    'max_leaf_nodes': randint(4, 32),
    'max_depth': [2, 3, 4, 6, None],
    'min_samples_leaf': randint(20, 201),
    'l2_regularization': loguniform(1e-3, 10),
}


def get_input_cols(df, predictor_set):
    input_cols = [c for c in df.columns if any(c.startswith(m) for m in INPUT_MODALITIES)]
    input_cols += [c for c in df.columns if re.match(BATCH_PC_PATTERN, c)]
    if predictor_set == "age15":
        input_cols = [c for c in input_cols if not re.match(AGE18_PATTERN, c)]
    return input_cols


'''
Train/test split of one seed from split_seed_<seed>.csv, the file used by the GLMM and GAMM scripts.
'''
def read_split(df, seed):
    split = pd.read_csv(f"split_seed_{seed}.csv", dtype={'cmpair': str})
    test_ids = set(split.loc[split['split'] == 'test', 'cmpair'])
    is_test = df['cmpair'].astype(str).isin(test_ids)
    return df[~is_test], df[is_test]


def inner_folds(train_df):
    return list(GroupKFold(n_splits=N_INNER_FOLDS).split(train_df, groups=train_df['cmpair'].astype(str)))


'''
Fit one comparator on the training split. Returns the fitted model, the selected hyperparameters and the
inner-CV mean squared error of the selected configuration (NaN for linear regression, which has nothing to tune).
'''
def fit_comparator(name, X_train, y_train, folds, seed):
    if name == "LinearRegression":
        model = LinearRegression().fit(X_train, y_train)
        return model, {}, np.nan

    if name == "ElasticNet":
        # Inputs are standardised so that one penalty applies equally to all items
        model = make_pipeline(StandardScaler(),
                              ElasticNetCV(l1_ratio=ELASTIC_NET_L1_RATIOS, cv=folds,   # default path of 100 penalties
                                           max_iter=20000, n_jobs=-1))
        model.fit(X_train, y_train)
        enet = model[-1]
        l1_index = ELASTIC_NET_L1_RATIOS.index(enet.l1_ratio_)
        alpha_index = int(np.argmin(np.abs(enet.alphas_[l1_index] - enet.alpha_)))
        cv_mse = enet.mse_path_[l1_index, alpha_index].mean()
        return model, {'alpha': enet.alpha_, 'l1_ratio': enet.l1_ratio_}, cv_mse

    if name == "GradientBoosting":
        # Early stopping is off: its internal validation split would not be grouped by twin pair
        search = RandomizedSearchCV(HistGradientBoostingRegressor(early_stopping=False, random_state=seed),
                                    BOOSTING_SPACE, n_iter=N_BOOSTING_CONFIGS, cv=folds,
                                    scoring='neg_mean_squared_error', random_state=seed, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_, -search.best_score_

    raise ValueError(f"Unknown comparator {name}")


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna().reset_index(drop=True)      # index = 0-based row_id, as in the other prediction scripts
    print(f"Data shape after dropping na: {df.shape}")
    sex_col = [c for c in df.columns if c.startswith("SEX")][0]

    results, tuning_rows = [], []
    for predictor_set in PREDICTOR_SETS:
        input_cols = get_input_cols(df, predictor_set)
        print(f"\n{predictor_set} predictors: {len(input_cols)} input columns")
        for seed in SEEDS:
            train_df, test_df = read_split(df, seed)
            print(f"\n{'='*60}\n  {predictor_set} predictors, seed {seed}: "
                  f"train {len(train_df)} rows, test {len(test_df)} rows\n{'='*60}")
            folds = inner_folds(train_df)
            X_train, X_test = train_df[input_cols].values, test_df[input_cols].values

            for model_tag, outcome in OUTCOMES:
                y_train, y_test = train_df[outcome].values, test_df[outcome].values
                for name in ["LinearRegression", "ElasticNet", "GradientBoosting"]:
                    model, params, cv_mse = fit_comparator(name, X_train, y_train, folds, seed)
                    y_pred = model.predict(X_test)
                    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
                    print(f"  {model_tag} {name:<17} test R²: {r2:.4f}  {json.dumps(params, default=float)}")

                    results.append(pd.DataFrame({
                        'model': name, 'predictor_set': predictor_set, 'outcome': model_tag, 'seed': seed,
                        'row_id': test_df.index.values, 'y_true': y_test, 'y_pred': y_pred,
                        'sex': test_df[sex_col].values,
                    }))
                    tuning_rows.append({'model': name, 'predictor_set': predictor_set, 'outcome': model_tag,
                                        'seed': seed, 'inner_cv_mse': cv_mse, 'test_r2': r2,
                                        'selected_hyperparameters': json.dumps(params, default=float)})
                    pd.concat(results, ignore_index=True).to_csv(OUT_FILE, index=False)
                    pd.DataFrame(tuning_rows).to_csv(TUNING_FILE, index=False)

    print(f"\nSaved: {OUT_FILE} (keep on the secure server) and {TUNING_FILE}")
