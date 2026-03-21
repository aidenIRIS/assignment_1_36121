# Data Preparation Script
# This script will handle loading and preprocessing the job advertisement datasets.
import pandas as pd
import os

def load_dataset(path):
    """Load a dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """Basic preprocessing: drop duplicates, handle missing values, normalize column names."""
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

# Example usage (to be replaced with actual dataset paths)
# df1 = load_dataset('../data/dataset1.csv')
# df1 = preprocess_data(df1)
# df2 = load_dataset('../data/dataset2.csv')
# df2 = preprocess_data(df2)
