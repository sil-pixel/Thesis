import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_rows = 10000
n_cols = 250
categories = [np.nan, 0, 1, 2, 3]
probabilities = [0.30, 0.45, 0.15, 0.05, 0.05]

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
df.to_csv('simulated_data.csv', index=False)
print(f"\nData saved to 'simulated_data.csv'")