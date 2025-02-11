import pandas as pd

def classify_columns(df):
    """
    Automatically classify columns as categorical or numerical.
    """
    categorical_columns = []
    numerical_columns = []

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            categorical_columns.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            numerical_columns.append(col)
        else:
            print(f"Column {col} has an unrecognized data type: {df[col].dtype}")

    return categorical_columns, numerical_columns
