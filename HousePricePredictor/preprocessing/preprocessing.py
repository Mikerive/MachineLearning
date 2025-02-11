import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import numpy as np
from .components.feature_engineering import create_time_series_features
from .components.feature_processing import process_features
from .components.column_classifier import classify_columns


def load_and_preprocess_data(csv_path: str, target_column: str, test_size: float = 0.2) -> tuple:
    """
    Load and preprocess housing price data.
    
    Args:
        csv_path: Path to the raw CSV data file
        target_column: Name of the target variable column
        test_size: Proportion of data to use for testing (split by time)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n=== Data Loading and Preprocessing ===")
    print(f"Loading data from: {csv_path}")
    
    # Read CSV with proper formatting
    df = pd.read_csv(
        csv_path,
        quotechar='"',
        quoting=1,  # QUOTE_ALL
        na_values=['NA', 'na', ''],
        thousands=','
    )
    
    print(f"\nInitial data shape: {df.shape}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    missing_pcts = df.isnull().mean() * 100
    
    print("\nMissing values per column:")
    for col in df.columns:
        if missing_counts[col] > 0:
            print(f"{col}: {missing_counts[col]} ({missing_pcts[col]:.1f}%)")
    
    print(f"\nUsing {target_column} as target")
    
    # Create time series features
    df_ts = create_time_series_features(df, target_column)
    
    # Determine feature types
    categorical_features = ['hpi_type', 'hpi_flavor', 'frequency', 'level', 'place_id', 'climate_zone']
    numerical_features = ['yr', 'period']
    time_series_features = [col for col in df_ts.columns if col not in categorical_features + numerical_features + [target_column]]
    
    # Process features
    X = process_features(
        df_ts,
        categorical_features,
        numerical_features,
        time_series_features,
        target_column
    )
    
    # Get target variable
    y = df[target_column]
    
    # Split data temporally
    split_idx = int(len(df) * (1 - test_size))
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def extract_state(place_name: str) -> str:
    """Extract state abbreviation from place names"""
    if ', ' in place_name:
        return place_name.split(', ')[-1].upper()
    return place_name


def save_processed_data(df: pd.DataFrame, output_dir: str = 'data/processed') -> str:
    """Save processed data to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/processed_data_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    return output_path
