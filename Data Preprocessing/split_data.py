'''
Export the twin-pair train/test splits used by DCMFNet and the ML benchmark, so that the R models
(GLMM_updated_split.R, GAMM_updated_split.R) are evaluated on exactly the same partitions.
Date: 22/10/2026
Project: Deep Cross Modal Fusion Model for predicting schizophrenia from Substance use in adolescents.

Usage: run from the folder containing catss_final_data.csv. Outputs are saved in the same folder.
    python split_data.py

Output:
    split_seed_<seed>.csv : one row per twin pair (cmpair, split = train/test)
    split_summary.csv     : rows and twin pairs per split and seed

The split files contain pseudonymised twin-pair identifiers derived from CATSS. Keep them on the
secure server together with the data and do not commit them to the repository.
'''

import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS = [42, 43, 44, 45, 46]
TEST_SIZE = 0.25

# Test-set rows per seed logged by the final DCMFNet run
# (Models/DCMFNet/Results/Final Best Model/best_model_training.log). Used only as a check.
EXPECTED_TEST_ROWS = {42: 1028, 43: 1010, 44: 998, 45: 1015, 46: 1021}


'''
Split the twin-pair IDs into train and test sets.
Identical to random_split() in DCMFNet/Method/legacy/train.py and ML/Method/lazy_predict.py:
same data loading (read_csv + dropna), same order of unique IDs, same call to train_test_split.
'''
def random_split_ids(df, test_size=0.25, random_state=42):
    unique_twin_ids = df['cmpair'].unique()
    train_ids, test_ids = train_test_split(unique_twin_ids, test_size=test_size, random_state=random_state)
    return train_ids, test_ids


if __name__ == "__main__":
    df = pd.read_csv("catss_final_data.csv")
    df = df.dropna()
    print(f"Data shape after dropping na: {df.shape}")

    summary_rows = []
    for seed in SEEDS:
        train_ids, test_ids = random_split_ids(df, test_size=TEST_SIZE, random_state=seed)

        split_df = pd.concat([
            pd.DataFrame({'cmpair': train_ids, 'split': 'train'}),
            pd.DataFrame({'cmpair': test_ids, 'split': 'test'}),
        ], ignore_index=True)
        split_df.to_csv(f"split_seed_{seed}.csv", index=False)

        n_train_rows = int(df['cmpair'].isin(train_ids).sum())
        n_test_rows = int(df['cmpair'].isin(test_ids).sum())
        matches_log = n_test_rows == EXPECTED_TEST_ROWS[seed]
        summary_rows.append({
            'seed': seed,
            'train_ids': len(train_ids), 'test_ids': len(test_ids),
            'train_rows': n_train_rows, 'test_rows': n_test_rows,
            'expected_test_rows': EXPECTED_TEST_ROWS[seed],
            'matches_dcmfnet_log': matches_log,
        })
        print(f"Seed {seed}: Train: {n_train_rows} rows ({len(train_ids)} IDs), "
              f"Test: {n_test_rows} rows ({len(test_ids)} IDs) | matches DCMFNet log: {matches_log}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("split_summary.csv", index=False)
    print("\nSaved: split_seed_<seed>.csv and split_summary.csv")

    if not summary_df['matches_dcmfnet_log'].all():
        print("\nWARNING: test-set sizes differ from the DCMFNet log. Check that catss_final_data.csv is the "
              "same file used for the final DCMFNet and ML runs before using these splits.")
