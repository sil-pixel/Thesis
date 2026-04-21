'''
Lazy Predict to find the best ML model for the given dataset and hyperparameters.
Author: Silpa Soni Nallacheruvu
Date: 20/04/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

'''

import lazypredict
from lazypredict.Supervised import LazyRegressor
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
import time

'''
Split the data into test and train sets using twin_id as the identifier and return the train and test dataframes with input and target modalities.
Input:
    df: pandas dataframe containing the data
    test_size: size of the test set
    random_state: random seed for reproducibility
Output:
    train_df: pandas dataframe containing the training data
    test_df: pandas dataframe containing the test data
'''
def random_split(df, test_size=0.25, random_state=42):
    unique_twin_ids = df['cmpair'].unique()
    train_ids, test_ids = train_test_split(unique_twin_ids, test_size=test_size, random_state=random_state)
    train_df = df[df['cmpair'].isin(train_ids)]
    test_df = df[df['cmpair'].isin(test_ids)]
    print(f"Train: {len(train_df)} rows ({len(train_ids)} IDs),"
            f"Test: {len(test_df)} rows ({len(test_ids)} IDs) \n")
    return train_df, test_df

'''
Get the input and output columns for the given model tag (Pos or Neg) based on the column names in the dataframe.
'''
def get_input_output_cols(df, model_tag):
    input_modalities = ["SUD15", "PRS", "SCZ15", "ADHD9", "ASD9", "ACE15", "ACE18", "SUD18", "SES", "SEX"]
    regex_patterns = [r"^batch_.*_x_PC"]
    target_col = f"SCZ18_{model_tag}_Norm"
    input_cols = [col for col in df.columns if any(col.startswith(mod) for mod in input_modalities)]
    input_cols += [col for pattern in regex_patterns for col in df.columns if re.match(pattern, col)]
    print(f"Input columns: {input_cols}\nTarget column: {target_col}\n")
    print(f"count of input columns :{len(input_cols)}")
    return input_cols, target_col


'''
Train a LazyRegressor model on the training data and evaluate it on the test data, printing the model performance metrics and saving the results to CSV files.
'''
def train_lazy_predict(train_df, test_df, input_cols, target_col):
    X_train = train_df[input_cols]
    y_train = train_df[target_col]
    X_test = test_df[input_cols]
    y_test = test_df[target_col]
    
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    
    print(models)
    # save the predictions and model performance metrics to a CSV file
    models['Model'] = models.index
    results_df = pd.DataFrame(models)
    results_df.to_csv(f"lazy_predict_results_{model_tag}.csv", index=False)
    print(f"Saved Lazy Predict results to lazy_predict_results_{model_tag}.csv\n")
    print(f"Predictions for {model_tag} model:\n{predictions}\n")



if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape after dropping na: {df.shape}")
    seeds = [42] #[42, 43, 44, 45, 46]  # Example seeds for multiple runs
    for seed in seeds:
        torch.manual_seed(seed)
        train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
        # Positive and negative symptom model training
        for model_tag in ["Pos", "Neg"]:
            print(f"\n{'='*60}")
            print(f"  Training {model_tag} symptom model")
            print(f"{'='*60}")
            start_time = time.time()
            input_cols, target_col = get_input_output_cols(train_df, model_tag=model_tag)
            train_lazy_predict(train_df, test_df, input_cols, target_col)
            end_time = time.time()
            print(f"Time taken for {model_tag} model: {end_time - start_time:.2f} seconds\n")

