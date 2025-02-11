# Feature Processing and LSTM Model Integration

## Feature Processing Pipeline

### 1. Preprocessing Stage
- **Numerical Features**
  - Standard scaling (zero mean, unit variance)
  - Includes: time series features, technical indicators
  - Handled by sklearn's StandardScaler

- **Categorical Features**
  - Encoding strategy automatically selected based on cardinality
  - Low cardinality: One-hot encoding
  - Medium cardinality: Target mean encoding + Weight of Evidence (WoE) if target available, else frequency encoding
  - High cardinality: Combination of frequency encoding and binning
  - Very high cardinality: Feature hashing
  - Drops first category to prevent multicollinearity

### 2. Sequence Creation
- **Input Shape**: (n_samples, n_features)
- **Output Shape**: (n_samples - sequence_length, sequence_length, n_features)
- **Sequence Length**: 24 periods (increased from 12)
- **Temporal Order**: Maintained during sequence creation
- **No Additional Scaling**: Features already scaled in preprocessing

### 3. LSTM Model Input
- **Batch Size**: 64 (increased for better gradient estimates)
- **Input Dimensions**: (batch_size, sequence_length, n_features)
- **Target Shape**: (batch_size,)
- **Data Loading**: Uses PyTorch DataLoader with drop_last=True

## Feature Categories

### Time Series Features
1. Rolling Statistics (3, 6, 12 periods)
   - Rolling means
   - Rolling standard deviations

2. Lagged Values (1, 2, 3, 6, 12 periods)
   - Previous price points
   - Captures historical patterns

3. Price Momentum (3, 6, 12 periods)
   - Rate of price changes
   - Trend strength indicators

4. Seasonal Components
   - Seasonal patterns
   - Trend component
   - Residual component

### Technical Features
1. Moving Averages
   - Fast SMA (3 periods)
   - Slow SMA (6 periods)
   - Trend signals

2. Rate of Change (1, 3, 6, 12 periods)
   - Price velocity
   - Momentum indicators

3. Statistical Measures
   - Z-scores
   - Volatility (12-period)
   - Price acceleration

4. Technical Indicators
   - RSI (14 periods)
   - Trend signals

## Data Flow
1. Raw data → Preprocessing
   - Feature engineering
   - Scaling
   - Encoding

2. Preprocessed data → Sequence creation
   - Temporal windows
   - Maintains feature relationships
   - No additional scaling

3. Sequences → LSTM model
   - Batched input
   - GPU acceleration if available
   - Gradient clipping for stability

## Feature Processing Documentation

### Overview
The feature processing module provides advanced techniques for handling different types of features in the house price prediction model, with a special focus on efficiently handling high-cardinality categorical features.

### Encoding Strategies

#### Categorical Features
The encoding strategy is automatically selected based on the cardinality (number of unique values) of each feature:

1. **Low Cardinality** (< 10 unique values)
   - Strategy: One-hot encoding
   - Use case: Ideal for features like `hpi_type`, `frequency`
   - Pros: Preserves all information, no information loss
   - Cons: Can create many features if cardinality is high

2. **Medium Cardinality** (10-100 unique values)
   - Strategy: Target mean encoding + Weight of Evidence (WoE) if target available, else frequency encoding
   - Use case: Good for features like `level`, `period`
   - Pros: Creates informative features while controlling dimensionality
   - Cons: May need cross-validation to prevent target leakage

3. **High Cardinality** (100-1000 unique values)
   - Strategy: Combination of frequency encoding and binning
   - Use case: Suitable for features like `state`, `region`
   - Pros: Handles rare categories while preserving frequency information
   - Cons: Some information loss due to binning

4. **Very High Cardinality** (>1000 unique values)
   - Strategy: Feature hashing
   - Use case: Perfect for features like `place_id`
   - Pros: Fixed output dimension, memory efficient
   - Cons: Possible hash collisions, less interpretable

#### Numerical Features
- All numerical features are standardized using `StandardScaler`
- Time series features are treated as numerical features

### Available Encoding Functions

#### `frequency_encode(df, column)`
Encodes categories based on their frequency in the dataset.
- Input: DataFrame and column name
- Output: Series with frequency values (0-1)

#### `target_mean_encode(df, column, target, min_samples=100)`
Encodes categories based on target mean with smoothing.
- Input: DataFrame, column name, target column
- Output: Series with smoothed target means
- Uses Bayesian smoothing to handle rare categories

#### `weight_of_evidence_encode(df, column, target, min_samples=100)`
Encodes categories using Weight of Evidence with Laplace smoothing.
- Input: DataFrame, column name, target column
- Output: Series with WoE values
- Particularly useful for binary classification problems

#### `hash_encode(series, n_features=100)`
Applies feature hashing for very high cardinality features.
- Input: Series and desired number of features
- Output: DataFrame with hashed binary features
- Memory efficient for high cardinality features

#### `bin_rare_categories(series, min_freq=0.01)`
Groups rare categories into an 'Other' category.
- Input: Series and minimum frequency threshold
- Output: Series with rare categories binned
- Helps handle rare categories in high cardinality features

### Usage Example

```python
from preprocessing.components.feature_processing import process_features

# Define feature groups
categorical_features = ['place_id', 'level', 'frequency']
numerical_features = ['yr', 'period']
time_series_features = ['rolling_mean', 'rolling_std']

# Process features
X = process_features(
    df=data,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    time_series_features=time_series_features,
    target_column='index_nsa'  # Optional, for target encoding
)
```

### Best Practices

1. **Feature Selection**
   - Consider removing or combining features with very high correlation
   - Monitor the impact of different encoding strategies on model performance

2. **Memory Usage**
   - Use feature hashing for very high cardinality features
   - Consider reducing n_features in hash_encode for extremely large datasets

3. **Target Encoding**
   - Always use cross-validation when applying target-based encoding
   - Apply appropriate smoothing for rare categories

4. **Monitoring**
   - Check the distribution of encoded features
   - Monitor for potential data leakage when using target-based encoding

### Performance Considerations

1. **Memory Efficiency**
   - Feature hashing is used for very high cardinality features to control memory usage
   - Rare category binning reduces the number of dummy variables

2. **Computation Speed**
   - Encoding strategies are selected to balance information content and computation time
   - Feature hashing provides constant-time encoding regardless of cardinality

3. **Model Performance**
   - Multiple encoding strategies (target mean, WoE) capture different aspects of categorical relationships
   - Automatic strategy selection ensures appropriate handling of different cardinality levels
