# House Price Predictor

A machine learning project for predicting housing prices using various models and time series features.

## Methodology

### Data Preprocessing

1. **Missing Value Handling**
   - Time Series Features: Forward fill followed by backward fill
   - Numerical Features (yr, period): Filled with -1
   - Categorical Features: Filled with 'unknown' before encoding

2. **Feature Engineering**
   - 12-month rolling mean price
   - Lagged prices (1, 3, and 12 months)
   - Smart Encoding:
     - Mean Encoding for high-cardinality features (>20 unique values)
     - One-Hot Encoding for low-cardinality features

3. **Temporal Split**
   - Training: Historical data up to 2021
   - Testing: Recent data (2022-2024)

## Geographical Feature Processing

The model uses sophisticated geographical feature processing to capture location-based price patterns:

### Metropolitan Area Analysis
- Tracks 30 major metropolitan areas across the US
- Uses real coordinates from OpenStreetMap's Nominatim API
- Includes employment data and economic weight factors
- Covers diverse economic regions:
  - Major financial centers (NYC, Chicago)
  - Tech hubs (SF, Seattle, Austin)
  - Regional economic centers (Atlanta, Dallas)
  - Emerging markets (Nashville, Raleigh)

### Economic Proximity Score
Calculates a weighted score based on:
- Distance to all metropolitan areas
- Employment levels in each metro (in millions)
- Economic importance weights (0.55-1.0)
- Regional influence factors

### Additional Geographical Features
- Precise latitude/longitude coordinates
- Coastal proximity indicators
- Climate zone classification
- Distance to nearest metropolitan area

### Data Management
- Coordinates cached locally to avoid API rate limits
- Automatic coordinate updates via `build_metro_cache.py`
- Graceful handling of missing or invalid coordinates
- Standardized numerical features for model input

### Integration
All geographical features are processed early in the pipeline and combined with:
- Time series features
- Market indicators
- Property characteristics

This comprehensive geographical analysis helps the model understand:
- Regional economic patterns
- Proximity to job markets
- Coastal/climate influences
- Metropolitan area relationships

## Model Architecture

## Physics-Informed LSTM

Implements domain knowledge through:

- **Price Positivity Constraint**: Penalizes negative price predictions
- **Temporal Smoothness**: Encourages gradual price changes
- **Combined Loss Function**: Balances accuracy and physical realism
- **Input Size**: Matches number of processed features
- **Hidden Layers**: 2 LSTM layers with 64 units
- **Dropout**: 0.2 regularization
- **Training**: 50 epochs with batch size 32
- **Optimizer**: Adam (learning rate 0.001)

## Training Monitoring

The LSTM model includes real-time loss tracking:
- **Live Plotting**: Shows MSE and Physics loss curves
- **Interactive Window**: Updates every epoch
- **Key Metrics**:
  - MSE Loss: Standard prediction error
  - Physics Loss: Domain knowledge constraints
  - Learning Rate: Current optimization step size

### Monitoring Usage

During training:
1. A separate window will appear showing loss curves
2. The plot updates automatically each epoch
3. Use the window controls to:
   - Zoom in/out
   - Pan across epochs
   - Save the plot

## Training Analysis

View historical training runs:
```bash
python run.py --analyze training_logs/lstm_20240206_193456/training_log.json
```

Key features:
- Stores training logs in timestamped directories
- Saves loss curves as PNG images
- JSON format for programmatic analysis
- Interactive visualization of historical runs

## Model Performance

## Usage

```bash
# Train specific models
python run.py --models lstm forest

# Train all models
python run.py --all

# Get help
python run.py -h
```

## Available Models

1. Linear Regression (`linear`)
2. Decision Tree (`tree`)
3. Random Forest (`forest`)
4. LSTM

## Results

Latest model performance:
| Model | MAE | R² Score |
|-------|-----|----------|
| Random Forest | $31.04 | 0.879 |
| LSTM | Pending | Pending |

## Future Improvements

1. Hyperparameter tuning for existing models
2. Feature importance analysis
3. Cross-validation with time series considerations
