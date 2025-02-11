# Feature Engineering Documentation

## Overview
This document describes the comprehensive time series feature engineering process implemented in the House Price Predictor system.

## Feature Categories

### 1. Rolling Statistics (Multiple Windows)
- Windows: 3, 6, and 12 periods
- Features:
  - Rolling mean: Captures local price trends
  - Rolling standard deviation: Measures local price volatility
- Purpose: Provides different time horizons for price movement analysis

### 2. Lagged Values
- Lags: 1, 2, 3, 6, and 12 periods
- Purpose: 
  - Captures price history at different intervals
  - Essential for time series prediction
  - Helps model learn from historical patterns

### 3. Price Momentum
- Periods: 3, 6, and 12
- Calculation: (current_price - price_n_periods_ago) / price_n_periods_ago
- Purpose:
  - Measures rate and strength of price movements
  - Indicates market sentiment and trend strength

### 4. Seasonal Components
- Features extracted using seasonal_decompose:
  - Seasonal: Captures recurring patterns
  - Trend: Long-term price direction
  - Residual: Unexplained price movements
- Purpose: Separates different components of price movement

### 5. Trend Indicators
- Simple Moving Average (SMA) Crossover:
  - Fast SMA (3 periods)
  - Slow SMA (6 periods)
  - Trend Signal: Binary indicator (1 when fast > slow)
- Purpose: Technical analysis for trend direction

### 6. Rate of Change (ROC)
- Periods: 1, 3, 6, and 12
- Calculation: Percentage change over specified period
- Purpose: 
  - Measures price momentum
  - Indicates acceleration/deceleration of price changes

### 7. Statistical Features
- Z-score: Identifies unusual price levels
- Volatility: Rolling 12-period standard deviation of returns
- Price Acceleration: Change in rate of change
- Purpose: Captures statistical properties of price movements

### 8. Technical Indicators
- Relative Strength Index (RSI):
  - 14-period calculation
  - Measures momentum and overbought/oversold conditions
- Purpose: Technical analysis indicator for price strength

## Missing Value Handling
- Forward and backward filling for numerical features
- Ensures continuity in time series data
- Maintains data quality for model training

## Usage Notes
1. Features are calculated in sequence to maintain data integrity
2. Original index is preserved throughout the process
3. All features are properly scaled and normalized
4. NaN values are handled appropriately for time series continuity

## Dependencies
- pandas: Data manipulation
- numpy: Numerical operations
- statsmodels: Seasonal decomposition
- scipy: Statistical calculations

## Performance Considerations
- Feature calculation is optimized for memory usage
- Processes data in a single pass where possible
- Handles large datasets efficiently
