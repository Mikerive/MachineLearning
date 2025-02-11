import argparse
from datetime import datetime
from preprocessing.preprocessing import load_and_preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from models.lstm_model.lstm_model import PhysicsInformedLSTMModel
from models.lstm_model.components.performance_analyzer import ModelPerformanceAnalyzer
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    print(f"\n=== Training {model_name} ===")
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%H:%M:%S')}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Adjust y_test to match the length of y_pred if using LSTM
    if hasattr(model, 'sequence_length'):
        y_test = y_test[model.sequence_length:]
    
    # Remove any NaN values from both test and predictions
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_test)
    y_test_valid = y_test[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Initialize performance analyzer
    analyzer = ModelPerformanceAnalyzer(model_name)
    metrics = analyzer.calculate_metrics(y_test_valid, y_pred_valid)
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Completed {model_name} in {duration}")
    
    # Print detailed performance report
    analyzer.print_metrics_report(metrics)
    
    # Generate performance plots
    analyzer.plot_prediction_vs_actual(y_test_valid, y_pred_valid)
    analyzer.plot_residuals(y_test_valid, y_pred_valid)
    plt.show()
    
    return metrics

def get_available_models(target_column):
    """Return dictionary of available models"""
    return {
        'linear': (LinearRegression(), "Linear Regression"),
        'tree': (DecisionTreeRegressor(random_state=42), "Decision Tree"),
        'forest': (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest"),
        'lstm': (PhysicsInformedLSTMModel(
            sequence_length=12,
            hidden_size=64,
            model_name="LSTM_Physics"
        ), "Physics-Informed LSTM"),
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate housing price prediction models.')
    parser.add_argument('--models', nargs='+', choices=get_available_models('target_column').keys(),
                       help='Models to train (e.g., --models linear tree forest)')
    parser.add_argument('--all', action='store_true',
                       help='Train all available models')
    parser.add_argument('--target_column', type=str, default='index_nsa',
                       help='Target column to predict')
    parser.add_argument('--log_path', type=str, default='training_log.csv',
                       help='Path to save training logs')
    args = parser.parse_args()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(
        'data/raw/hpi_master.csv',
        args.target_column
    )
    
    # Get models to train
    available_models = get_available_models('target_column')
    models_to_train = []
    
    if args.all:
        models_to_train = list(available_models.items())
    elif args.models:
        models_to_train = [(name, available_models[name]) for name in args.models]
    else:
        parser.print_help()
        return
    
    # Train and evaluate selected models
    results = {}
    for model_key, (model, model_name) in models_to_train:
        metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name)
        results[model_key] = {'name': model_name, 'metrics': metrics}
        
        # Keep plot window open for LSTM
        if hasattr(model, 'monitor'):
            model.monitor.show()

    # Print results comparison
    print("\nModel Performance Comparison:")
    for model_key, metrics in results.items():
        print(f"{metrics['name']}:")
        print(f"  RMSE: ${metrics['metrics']['rmse']:.2f}")
        print(f"  MAE: ${metrics['metrics']['mae']:.2f}")
        print(f"  R² Score: {metrics['metrics']['r2']:.3f}\n")

def analyze_training(log_path):
    from HousePricePredictorFunctions.models.lstm_model import TrainingMonitor
    monitor = TrainingMonitor.load(log_path)
    monitor.show()  # Keep window open

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Models to train')
    parser.add_argument('--analyze', help='Path to training log JSON')
    parser.add_argument('--target_column', type=str, default='index_nsa',
                       help='Target column to predict')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_training(args.analyze)
    else:
        main()
