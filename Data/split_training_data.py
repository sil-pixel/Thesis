# Project: Modeling Psychotic Risk from Substance Use in Adolescents
# File title: Functions to extract data and split into training and testing set
# Author: Silpa Soni Nallacheruvu
# Date: 12/2/2026

# Load the modules 
import runpy, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# module load Python3/
# python3 -m pip install pandas

# splitting for various ratios 
ratios = [
    (0.60, 0.40, "60:40"),
    (0.70, 0.30, "70:30"),
    (0.80, 0.20, "80:20")
]
strategy = 'random'
dataset = "catss_merged.csv"

# Utility function for random split for a specific test ratio
def random_split(test_ratio, dataset, seed):
        # Load the merged dataset from local 
        # run python3 merge_catss_data.py if not present
        catss = pd.read_csv(dataset)
        # getting the unique twin pairs
        unique_twin_pair_ids = catss['cmpair'].unique()
        # split them randomly
        train_ids, test_ids = train_test_split(
            unique_twin_pair_ids,
            test_size=test_ratio,
            random_state=seed
        )
        # Filter the merged dataset by the twin pair ids
        train_random = catss[catss['cmpair'].isin(train_ids)]
        test_random = catss[catss['cmpair'].isin(test_ids)]
        print(f"Train: {len(train_random)} rows ({len(train_ids)} IDs),"
            f"Test: {len(test_random)} rows ({len(test_ids)} IDs) \n")
        return train_random, test_random
        

# 1. List Random split counts for a list of ratios
def random_splits(catss, ratios):
    print("Random split grouped by cmpair:")
    for _, test_size, label in ratios:
        print(f"{label} -")
        train_random, test_random = random_split(test_ratio=test_size, dataset=dataset)
        print("\n")

# 2. Generate bar plot comparing distributions for a set of variables
def generate_plot_compare_distributions(train_df, test_df, variables, split_label, strategy, file_name):
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4*n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(variables):
        # Since CATSS is discrete variables we will use bar plots
        
        train_counts = train_df[var].value_counts(normalize=True)
        test_counts = test_df[var].value_counts(normalize=True)

        all_categories = sorted(set(train_counts.index) | set(test_counts.index))
        train_vals = [train_counts.get(cat, 0) for cat in all_categories]
        test_vals = [test_counts.get(cat, 0) for cat in all_categories]

        x = np.arange(len(all_categories))
        width = 0.35

        axes[idx, 0].bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
        axes[idx, 0].bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
        axes[idx, 0].set_xlabel(var)
        axes[idx, 0].set_ylabel('Proportion')
        axes[idx, 0].set_title(f'{var} Distribution')
        axes[idx, 0].set_xticks(x)
        axes[idx, 0].set_xticklabels(all_categories, rotation=45)
        axes[idx, 0].legend()

        axes[idx, 1].axis('off')
    plt.suptitle(f'{strategy} - {split_label} Split', fontsize=16, y=1.001)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()


# Compare distributions plots for each ratio in random splits for a list of variables
def compare_distributions_list(variables_list, variable):
    for _, test_size, label in ratios:
        file_name = f'{variable}_{strategy}_{label}_barplot_comparison.png'
        train_random, test_random = random_split(test_size, dataset=dataset)
        generate_plot_compare_distributions(train_random, test_random, variables_list, label, strategy, file_name)


def compare_exposure_distributions():
     exposure_variables = ["SC_A_D_5", "SC_A_D_6", "SC_A_D_7", "SC_A_D_8", "SC_A_D_9", "SC_A_D_10", "SC_A_D_11", "SC_A_D_12", "SC_A_D_13", "SC_A_D_14", "SC_A_D_15", "SC_A_D_16", "SC_A_D_17"]
     compare_distributions_list(exposure_variables, 'drug_use')


def compare_outcome_distributions():
     outcome_variables = ["PLE_TB_02", "PLE_TB_03",  "PLE_TB_01", "PLE_TB_04", "PLE_TB_05", "PLE_TB_06", "PLE_TB_07"]
     compare_distributions_list(outcome_variables, 'psychotic')
