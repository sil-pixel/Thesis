'''
Lazy Predict to find the best ML model for the given dataset and hyperparameters.
Author: Silpa Soni Nallacheruvu
Date: 20/04/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.
'''

import lazypredict
from lazypredict.Supervised import REGRESSORS
import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
import time
import numpy as np

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
Compute all regression metrics for a fitted model.
'''
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    if np.std(preds) < 1e-6:
        spearman_rho, spearman_p = float('nan'), float('nan')
        pearson_r, pearson_p = float('nan'), float('nan')
    else:
        spearman_rho, spearman_p = spearmanr(y_test, preds)
        pearson_r, pearson_p = pearsonr(y_test, preds)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
    }

'''
Train all regressors across multiple seeds and return averaged results.
'''
def train_all_models(df, seeds, model_tag):
    input_cols, target_col = get_input_output_cols(df, model_tag)
    
    # {model_name: [metrics_dict_per_seed]}
    all_seed_results = {}
    
    for seed in seeds:
        print(f"\n  Seed {seed}...")
        train_df, test_df = random_split(df, test_size=0.25, random_state=seed)
        
        X_train = train_df[input_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[input_cols].values
        y_test = test_df[target_col].values
        
        for name, model_class in REGRESSORS:
            try:
                model = model_class()
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_test, y_test)
                
                if name not in all_seed_results:
                    all_seed_results[name] = []
                all_seed_results[name].append(metrics)
                
            except Exception as e:
                print(f" Skipping {name} (seed {seed}): {e}")
                continue
    
    return all_seed_results


'''
Compute mean ± std across seeds for each model.
Only include models that ran successfully on all seeds.
'''
def compute_summary(all_seed_results, n_seeds):
    metric_names = ['rmse', 'r2', 'mae', 'spearman_rho', 'spearman_p', 'pearson_r', 'pearson_p']
    summary_rows = []
    per_seed_rows = []
    
    for model_name, seed_metrics_list in all_seed_results.items():
        # Only include models that completed all seeds
        if len(seed_metrics_list) < n_seeds:
            print(f"  Skipping {model_name}: only {len(seed_metrics_list)}/{n_seeds} seeds completed")
            continue
        
        row = {'Model': model_name}
        for metric in metric_names:
            values = [m[metric] for m in seed_metrics_list]
            # Filter NaN values for correlation metrics
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                row[f'{metric}_mean'] = np.mean(valid_values)
                row[f'{metric}_std'] = np.std(valid_values)
            else:
                row[f'{metric}_mean'] = float('nan')
                row[f'{metric}_std'] = float('nan')
        
        summary_rows.append(row)
        
        # Per-seed detail rows
        for i, metrics in enumerate(seed_metrics_list):
            seed_row = {'Model': model_name, 'seed_index': i}
            seed_row.update(metrics)
            per_seed_rows.append(seed_row)
    
    summary_df = pd.DataFrame(summary_rows)
    per_seed_df = pd.DataFrame(per_seed_rows)
    
    return summary_df, per_seed_df


'''
Print a clean summary table sorted by RMSE.
'''
def print_summary_table(summary_df, model_tag):

    summary_df = summary_df.sort_values('rmse_mean', ascending=True)
    
    print(f"\n{'='*100}")
    print(f"  {model_tag} Symptom Model - Benchmark Results (mean ± std across seeds)")
    print(f"{'='*100}")
    print(f"  {'Model':<30} {'RMSE':>14} {'R2':>14} {'Spearman rho':>14} {'Pearson r':>14}")
    print(f"  {'-'*86}")
    
    for _, row in summary_df.iterrows():
        name = row['Model']
        rmse_str = f"{row['rmse_mean']:.4f}±{row['rmse_std']:.4f}"
        r2_str = f"{row['r2_mean']:.4f}±{row['r2_std']:.4f}"
        sp_str = f"{row['spearman_rho_mean']:.4f}±{row['spearman_rho_std']:.4f}"
        pr_str = f"{row['pearson_r_mean']:.4f}±{row['pearson_r_std']:.4f}"
        print(f"  {name:<30} {rmse_str:>14} {r2_str:>14} {sp_str:>14} {pr_str:>14}")
    
    print(f"  {'-'*86}")


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape after dropping na: {df.shape}")
    
    seeds = [42, 43, 44, 45, 46]
    
    for model_tag in ["Pos", "Neg"]:
        print(f"\n{'='*60}")
        print(f"  Benchmarking {model_tag} symptom model ({len(seeds)} seeds)")
        print(f"{'='*60}")
        
        start_time = time.time()
        all_seed_results = train_all_models(df, seeds, model_tag)
        elapsed = time.time() - start_time
        print(f"\nTotal time for {model_tag}: {elapsed:.1f} seconds")
        
        # Compute and display summary
        summary_df, per_seed_df = compute_summary(all_seed_results, n_seeds=len(seeds))
        print_summary_table(summary_df, model_tag)
        
        # Save to CSV
        summary_df.sort_values('rmse_mean', ascending=True).to_csv(
            f'{model_tag}_benchmark_summary.csv', index=False
        )
        per_seed_df.to_csv(
            f'{model_tag}_benchmark_per_seed.csv', index=False
        )
        print(f"Saved: '{model_tag}_benchmark_summary.csv' and '{model_tag}_benchmark_per_seed.csv'")