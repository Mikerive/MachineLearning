# House Price Predictor

An advanced machine learning project for predicting housing prices using multiple models and sophisticated geographical features. The system combines traditional ML approaches with deep learning to capture both temporal and spatial patterns in housing markets.

## 🏗 Project Structure

```
HousePricePredictor/
├── data/                    # Data files (not in git)
│   ├── raw/                # Original data files
│   ├── processed/          # Cleaned and processed datasets
│   └── mappings/          # Geographical coordinate mappings
├── models/                 # Model implementations
│   ├── linear_regression.py
│   ├── decision_tree.py
│   └── lstm_model/        # Deep learning implementation
├── preprocessing/          # Data processing pipeline
├── trained_models/        # Saved model files (not in git)
├── docs/                  # Documentation
└── tests/                # Unit tests
```

## 🤖 Models

### 1. LSTM Neural Network
- Specialized for temporal patterns in housing markets
- Features:
  - Multi-layer architecture with dropout
  - Sequence-based prediction
  - Handles both numerical and categorical features
  - Optimized for time series forecasting

### 2. Random Forest
- Ensemble learning approach
- Features:
  - Handles non-linear relationships
  - Feature importance ranking
  - Robust to outliers
  - Cross-validated predictions

### 3. Linear Regression
- Baseline model for comparison
- Features:
  - L1/L2 regularization
  - Interaction term handling
  - Robust scaling

## 🔄 Data Processing Pipeline

### Feature Engineering
1. **Temporal Features**
   - Rolling statistics (mean, std, min, max)
   - Lagged values (1, 3, 12 months)
   - Seasonal decomposition

2. **Geographical Features**
   - Coordinates (latitude/longitude)
   - Distance to economic centers
   - Climate zone indicators
   - Metropolitan area proximity scores

3. **Market Indicators**
   - Price trends
   - Market volatility
   - Seasonal patterns

## 🛠 Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Mikerive/MachineLearning.git
cd HousePricePredictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required data files:
   - Request access to the data files from project maintainers
   - Place raw data files in `data/raw/`
   - Run preprocessing scripts:
   ```bash
   python preprocessing/scripts/build_coordinate_cache.py
   python preprocessing/scripts/process_raw_data.py
   ```

## 🗄 Large Files and Data

Due to size limitations, the following files are not included in the repository:
- Trained models (`trained_models/*.pkl`)
- Processed datasets (`data/processed/*.csv`)
- Raw data files (`data/raw/*.csv`)
- Training logs (`training_logs/`)

These files can be:
1. Generated using the provided scripts
2. Downloaded from our data storage (contact maintainers)
3. Created through the model training process

## 📊 Performance

Latest model performance metrics (as of Feb 11, 2025):

### Physics-Informed LSTM
- RMSE: $8.15
- MAE: $5.67
- R² Score: 0.994
- MAPE: 3.4%
- Direction Accuracy: 80.5%
- 95% Prediction Interval: ±$13.56

### Training Details
- Early stopping triggered at epoch 41
- Training time: 45 minutes
- Input features: 551 dimensions
- Sequence length: 12 periods
- Architecture: 64 hidden units

### Dataset Statistics
- Total samples: 127,329
- Training samples: 101,851
- Features engineered: 37 time-series features including:
  - Rolling statistics (3, 6, 12 periods)
  - Price momentum indicators
  - Seasonal decomposition
  - Technical indicators (RSI, SMA)
  - Volatility measures

### Key Improvements
- Achieved 99.4% variance explanation (R² Score)
- Sub-$8 RMSE on house price predictions
- Less than 3.5% mean absolute percentage error
- Strong directional accuracy at 80.5%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For access to large data files or any questions, contact the maintainers.

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
| LSTM | $5.67 | 0.994 |

## Future Improvements

1. Hyperparameter tuning for existing models
2. Feature importance analysis
3. Cross-validation with time series considerations
