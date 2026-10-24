'''
Test-set predictions of DCMFNet for the prediction-level evaluation (prediction_evaluation.py):
calibration, sex-stratified and top-decile metrics, paired comparisons, and the age-15-only sensitivity analysis.
Date: 23/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Predictor sets:
    full  : all inputs, as in the main analysis (retrained with the same seeds, splits and hyperparameters).
    age15 : without the information in the age-18 predictors (ACE18, SUD18), which are measured at the same
            age as the outcome. The architecture and tuned hyperparameters are kept, and the ACE18 and SUD18
            inputs are set to their training-set mean in both training and test data, so the model cannot use them.
            The GLMM/GAMM age-15 models drop these columns instead (glmm_gamm_test_predictions.R).

Usage: run from the folder containing catss_final_data.csv and hyperparameters.json
(the same set-up as importance_analysis_avg.py).
    python dcmfnet_test_predictions.py

Output: SERVER_ONLY_predictions_DCMFNet.csv, one row per test participant (observed and predicted outcome, sex).
This is participant-level data: keep it on the secure server and export only the outputs of prediction_evaluation.py.
The file is rewritten after every model, so an interrupted run keeps the finished models.
'''

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The legacy training code is imported from the repository folder
METHOD_DIR = Path(__file__).resolve().parent.parent / "Method"
if METHOD_DIR.is_dir() and str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from train import random_split, prepare_data, calculate_modality_sizes, create_dataloader, evaluate, train

SEEDS = [42, 43, 44, 45, 46]
PREDICTOR_SETS = ["full", "age15"]
AGE18_PATTERN = r"^(ACE18|SUD18)"
OUT_FILE = "SERVER_ONLY_predictions_DCMFNet.csv"


'''
Remove the information in the age-18 predictors: replace them by their training-set mean.
'''
def neutralise_age18(train_df, test_df):
    age18_cols = [c for c in train_df.columns if re.match(AGE18_PATTERN, c)]
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in age18_cols:
        mean = train_df[col].mean()
        train_df[col] = mean
        test_df[col] = mean
    return train_df, test_df, age18_cols


'''
Train the Pos and Neg models of one seed in the same order as the final run (legacy/train.py __main__):
the seed is set once, the Pos model is trained and evaluated on a shuffled test loader, then the Neg model is trained.
The Neg model therefore starts from the random state left by the Pos model. Reseeding before the Neg model gives a
different initialisation (and test R² 0.190 instead of 0.198 for the final configuration; D-23).
'''
def train_seed_models(train_df, test_df, seed, modality_sizes, hyperparameters_json):
    torch.manual_seed(seed)
    models = {}
    for model_tag in ["Pos", "Neg"]:
        hyperparams = hyperparameters_json[model_tag]
        (model, *_) = train(train_df, seed, modality_sizes, model_tag, hyperparams)
        # Same random-number use as evaluate_final_test() in the final run (shuffled test loader), without its plot
        X_test, Y_test = prepare_data(test_df, model_tag)
        evaluate(model, create_dataloader(X_test, Y_test, hyperparams["batch_size"]))
        models[model_tag] = model
    return models


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model([torch.tensor(x, dtype=torch.float32) for x in X]).numpy().flatten()


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna().reset_index(drop=True)      # index = 0-based row_id, as in glmm_gamm_test_predictions.R
    print(f"Data shape after dropping na: {df.shape}")
    modality_sizes = calculate_modality_sizes(df)
    sex_col = [c for c in df.columns if c.startswith("SEX")][0]

    with open("hyperparameters.json", "r") as f:
        hyperparameters_json = json.load(f)

    results = []
    for predictor_set in PREDICTOR_SETS:
        for seed in SEEDS:
            print(f"\n{'='*60}\n  DCMFNet Pos and Neg, {predictor_set} predictors, seed {seed}\n{'='*60}")
            train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
            if predictor_set == "age15":
                train_df, test_df, age18_cols = neutralise_age18(train_df, test_df)
                print(f"  Age-18 inputs set to the training mean: {age18_cols}")

            models = train_seed_models(train_df, test_df, seed, modality_sizes, hyperparameters_json)

            for model_tag, outcome in [("Pos", "SCZ18_Pos_Norm"), ("Neg", "SCZ18_Neg_Norm")]:
                X_test, Y_test = prepare_data(test_df, model_tag)
                y_pred = predict(models[model_tag], X_test)
                r2 = 1 - np.sum((Y_test - y_pred) ** 2) / np.sum((Y_test - Y_test.mean()) ** 2)
                print(f"  {model_tag} test R²: {r2:.4f}")

                results.append(pd.DataFrame({
                    'model': 'DCMFNet', 'predictor_set': predictor_set, 'outcome': model_tag, 'seed': seed,
                    'row_id': test_df.index.values, 'y_true': test_df[outcome].values, 'y_pred': y_pred,
                    'sex': test_df[sex_col].values,
                }))
                pd.concat(results, ignore_index=True).to_csv(OUT_FILE, index=False)

    print(f"\nSaved: {OUT_FILE} (keep on the secure server)")
