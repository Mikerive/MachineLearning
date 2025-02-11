import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List

class ModelPerformanceAnalyzer:
    def __init__(self, model_name: str):
        """Initialize the performance analyzer.
        
        Args:
            model_name: Name of the model being analyzed
        """
        self.model_name = model_name
        self.metrics_history = []
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of performance metrics.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            
        Returns:
            Dictionary containing various performance metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Direction accuracy (for time series)
        direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
        direction_accuracy = direction_correct / (len(y_true) - 1) * 100
        
        # Calculate prediction intervals
        residuals = y_true - y_pred
        std_residuals = np.std(residuals)
        prediction_interval_95 = 1.96 * std_residuals
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'prediction_interval_95': prediction_interval_95
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                title: str = None) -> None:
        """Plot predicted vs actual values with error bands.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            title: Optional title for the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot actual vs predicted
        plt.scatter(y_true, y_pred, alpha=0.5, label='Predictions')
        
        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title or f'{self.model_name} - Predicted vs Actual Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot residual analysis charts.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True)
        
        # Residual Distribution
        ax2.hist(residuals, bins=50, density=True, alpha=0.7)
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Residual Distribution')
        ax2.grid(True)
        
        plt.tight_layout()
    
    def print_metrics_report(self, metrics: Dict[str, float]) -> None:
        """Print a comprehensive metrics report.
        
        Args:
            metrics: Dictionary of metrics to report
        """
        print(f"\nDetailed Performance Report for {self.model_name}")
        print("=" * 50)
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"MAPE: {metrics['mape']:.1f}%")
        print(f"Direction Accuracy: {metrics['direction_accuracy']:.1f}%")
        print(f"95% Prediction Interval: ±${metrics['prediction_interval_95']:.2f}")
        
        if metrics['r2'] < 0:
            print("\nWarning: Negative R² Score indicates poor model performance:")
            print("- Model predictions are worse than using the mean value")
            print("- Consider model retraining or hyperparameter tuning")
        
        if metrics['mape'] > 20:
            print("\nWarning: High MAPE indicates large percentage errors:")
            print("- Average percentage error is above 20%")
            print("- Consider feature engineering or data normalization")
