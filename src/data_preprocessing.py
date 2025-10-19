import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def load_all_datasets(folder_path="../data"):
    """Load and merge all CSV datasets in the folder"""
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(path)
                print(f"[INFO] Loaded: {file} ({df.shape[0]} rows)")
                all_dfs.append(df)
            except Exception as e:
                print(f"[WARN] Skipping {file} due to read error: {e}")

    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Merged dataset shape: {merged.shape}")
    return merged

def preprocess_data(df: pd.DataFrame):
    """Clean and encode dataset"""

    missing_percentage = df.isnull().sum() / len(df)
    cols_to_drop = missing_percentage[missing_percentage > 0.5].index

    # Ensure at least one column remains
    if len(df.columns) > 1 and len(cols_to_drop) < len(df.columns):
        df = df.drop(columns=cols_to_drop)
        print(f"[INFO] Dropped {len(cols_to_drop)} columns with >50% missing values.")

        if df.empty:
            print("[WARN] DataFrame is empty after dropping columns with >50% missing values. Adjusting threshold or data loading may be necessary.")
            print("[WARN] Returning empty DataFrame.")
            return pd.DataFrame({'default': [None]})  # Return a DataFrame with a default column
        #raise ValueError("DataFrame is empty after dropping columns with missing values.")


    # Impute missing numerical values with the mean
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Impute missing categorical values with the mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    cat_cols = df.select_dtypes(include=['object']).columns

    if len(cat_cols) > 0:
        encoder = LabelEncoder()
        for col in cat_cols:
            try:
                df[col] = encoder.fit_transform(df[col].astype(str))
            except Exception as e:
                print(f"[WARN] Could not label encode column '{col}': {e}")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns # re-select num_cols after potential column drops and encoding

    # Replace infinite values with a large finite number
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col][~np.isnan(df[col])].max())
    if len(num_cols) > 0: # check that there are numerical columns to scale
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[INFO] Preprocessing complete. Columns: {len(df.columns)}")

    return df
