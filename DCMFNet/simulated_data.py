import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_rows = 9000
n_cols = 150
categories = [np.nan, 0, 1, 2, 3]
probabilities = [0.30, 0.45, 0.15, 0.05, 0.05]

def generate_simulated_data(n_rows, n_cols, categories, probabilities):
    # For each variable, create the main column and a missing indicator
    # Create ID column
    df = pd.DataFrame({'id': range(1, n_rows + 1)})
    df['twin_id'] = (df['id'] - 1) // 2 + 1

    # Create column names
    column_names = [f'var_{i+1}' for i in range(n_cols)]
    for col_name in column_names:
        # Sample from categories with specified probabilities
        # Use object dtype to allow NaN values
        df[col_name] = np.random.choice(categories, size=n_rows, p=probabilities)
        # Convert columns to nullable integer type (Int64 allows NaN)
        df[col_name] = df[col_name].astype('Int64')  # Note: capital 'I' for nullable integer

    # Display info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes.value_counts())

    print(f"\nMissing values per column (first 5 columns):")
    print(df.iloc[:, :5].isnull().sum())

    print(f"\nValue distribution for first column:")
    print(df['var_1'].value_counts(dropna=False, normalize=True).sort_index())

    
    # Verify the distribution across all columns
    print(f"\nOverall distribution (as percentage):")
    total_values = df.shape[0] * df.shape[1]
    missing_count = df.isnull().sum().sum()
    print(f"Missing: {missing_count / total_values * 100:.2f}%")
    for cat in [0, 1, 2, 3]:
        cat_count = (df == cat).sum().sum()
        print(f"{int(cat)}: {cat_count / total_values * 100:.2f}%")

    # Save to CSV (optional)
    df.to_csv('simulated_data_without_modalities.csv', index=False)
    print(f"\nData saved to 'simulated_data_without_modalities.csv'")


def assign_modalities():
    df = pd.read_csv('simulated_data_without_modalities.csv')
    # Define modality groups
    modality_groups = {
        'SUD15': 5,
        'PRS': 10,
        'SCZ15': 25,
        'ADHD9': 20,
        'ASD9': 15,
        'ACE': 35,
        'SES': 4,
        'Sex': 1,
        'PCA': 10,
        'SCZ18': 25
    }
    
    # Rename columns based on modality groups
    current_col = 0
    for modality, count in modality_groups.items():
        for i in range(count):
            df.rename(columns={f'var_{current_col + 1}': f'{modality}_var_{i+1}'}, inplace=True)
            current_col += 1
        
    df = df[['id', 'twin_id'] + [col for col in df.columns if col.startswith(tuple(modality_groups.keys()))]]
    print(f"Columns after modality assignment: {df.columns.tolist()}")
    df.to_csv('simulated_data.csv', index=False)

assign_modalities()