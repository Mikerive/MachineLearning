import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats

def create_time_series_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Create time series features from the target column.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column to create features from
        
    Returns:
        DataFrame with added time series features
    """
    print("\nCreating time-series features:")

    # Create a copy to avoid modifying the original
    df_ts = df.copy()
    
    # Handle missing values in target column using forward/backward fill
    df_ts[target_column] = df_ts[target_column].ffill().bfill()
    
    # Store original index for later
    original_index = df_ts.index.copy()
    df_ts = df_ts.reset_index(drop=True)
    
    # 1. Rolling Statistics with multiple windows
    windows = [3, 6, 12]
    for window in windows:
        print(f"- Calculating {window}-period rolling statistics")
        df_ts[f'rolling_mean_{window}'] = df_ts[target_column].rolling(window=window, min_periods=1).mean()
        df_ts[f'rolling_std_{window}'] = df_ts[target_column].rolling(window=window, min_periods=1).std()
        
    # 2. Lagged Values (more comprehensive)
    lags = [1, 2, 3, 6, 12]
    print("- Calculating lagged prices")
    for lag in lags:
        df_ts[f'lag_{lag}_price'] = df_ts[target_column].shift(lag)
    
    # 3. Price Momentum
    print("- Calculating price momentum")
    for period in [3, 6, 12]:
        # Momentum = (current_price - price_n_periods_ago) / price_n_periods_ago
        df_ts[f'momentum_{period}'] = df_ts[target_column].diff(period)
    
    # 4. Seasonal Components
    print("- Extracting seasonal components")
    try:
        # Decompose the series into trend, seasonal, and residual components
        decomposition = seasonal_decompose(df_ts[target_column], period=12, extrapolate_trend='freq')
        df_ts['seasonal'] = decomposition.seasonal
        df_ts['trend'] = decomposition.trend
        df_ts['residual'] = decomposition.resid
    except:
        print("Warning: Could not perform seasonal decomposition")
    
    # 5. Trend Indicators
    print("- Calculating trend indicators")
    # Simple Moving Average Crossovers
    df_ts['sma_fast'] = df_ts[target_column].rolling(window=3, min_periods=1).mean()
    df_ts['sma_slow'] = df_ts[target_column].rolling(window=6, min_periods=1).mean()
    df_ts['trend_signal'] = (df_ts['sma_fast'] > df_ts['sma_slow']).astype(int)
    
    # 6. Rate of Change
    print("- Calculating rate of change features")
    for period in [1, 3, 6, 12]:
        df_ts[f'roc_{period}'] = df_ts[target_column].pct_change(period)
    
    # 7. Statistical Features
    print("- Calculating statistical features")
    # Z-score of prices
    df_ts['price_zscore'] = stats.zscore(df_ts[target_column], nan_policy='omit')
    
    # Volatility (standard deviation of returns)
    returns = df_ts[target_column].pct_change()
    df_ts['volatility'] = returns.rolling(window=12).std()
    
    # Price acceleration (change in rate of change)
    df_ts['price_acceleration'] = returns.diff()
    
    # 8. Technical Indicators
    print("- Calculating technical indicators")
    # Relative Strength Index (RSI)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df_ts['rsi'] = calculate_rsi(df_ts[target_column])
    
    # Handle any remaining NaN values
    print("- Handling missing values")
    for column in df_ts.columns:
        if df_ts[column].dtype in [np.float64, np.float32]:
            df_ts[column] = df_ts[column].ffill().bfill()
    
    # Restore original index
    df_ts.index = original_index
    
    # Print feature creation summary
    print("\nTime-series features created:")
    feature_counts = {col: df_ts[col].count() for col in df_ts.columns if col != target_column}
    for feature, count in feature_counts.items():
        print(f"{feature}: {count} values")
    
    return df_ts
