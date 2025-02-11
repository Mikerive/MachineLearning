"""
Model evaluation components for LSTM model.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from scipy import stats

class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def calculate_regression_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        metrics = {}
        
        # Mean Squared Error
        metrics['mse'] = np.mean((y_true - y_pred) ** 2)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Error
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot)
        
        return metrics
    
    def calculate_prediction_intervals(self, y_pred: np.ndarray,
                                     confidence_level: float = 0.95) -> List[Tuple[float, float]]:
        """Calculate prediction intervals for forecasts."""
        # Estimate prediction error distribution
        errors = np.random.normal(0, np.std(y_pred), size=(1000, len(y_pred)))
        simulated_predictions = y_pred + errors
        
        # Calculate intervals
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        intervals = []
        for i in range(len(y_pred)):
            predictions = simulated_predictions[:, i]
            lower = np.percentile(predictions, lower_percentile * 100)
            upper = np.percentile(predictions, upper_percentile * 100)
            intervals.append((lower, upper))
        
        return intervals
    
    def analyze_errors(self, y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze prediction errors."""
        errors = y_true - y_pred
        
        analysis = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'skewness': stats.skew(errors),
            'kurtosis': stats.kurtosis(errors)
        }
        
        return analysis

class PerformanceMetrics:
    """Custom performance metrics for house price prediction."""
    
    def mean_absolute_percentage_error(self, y_true: np.ndarray,
                                     y_pred: np.ndarray) -> float:
        """Calculate MAPE with handling for edge cases."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def price_range_accuracy(self, y_true: np.ndarray,
                           y_pred: np.ndarray,
                           tolerance: float = 0.1) -> float:
        """Calculate percentage of predictions within tolerance range."""
        within_range = np.abs((y_true - y_pred) / y_true) <= tolerance
        return np.mean(within_range)
    
    def weighted_mean_absolute_error(self, y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   weights: np.ndarray) -> float:
        """Calculate weighted MAE."""
        return np.average(np.abs(y_true - y_pred), weights=weights)

class VisualizationTools:
    """Tools for visualizing model performance."""
    
    def plot_predictions(self, dates: pd.DatetimeIndex,
                        y_true: np.ndarray,
                        y_pred: np.ndarray) -> plt.Figure:
        """Plot actual vs predicted prices over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, y_true, label='Actual', color='blue', alpha=0.7)
        ax.plot(dates, y_pred, label='Predicted', color='red', alpha=0.7)
        
        ax.set_title('House Price Predictions Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray,
                              y_pred: np.ndarray) -> plt.Figure:
        """Plot distribution of prediction errors."""
        errors = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(errors, bins=50, density=True, alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Add normal distribution fit
        mu, std = np.mean(errors), np.std(errors)
        x = np.linspace(mu - 3*std, mu + 3*std, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r-', alpha=0.7, label='Normal Distribution')
        
        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: List[str],
                              importance_scores: List[float]) -> plt.Figure:
        """Plot feature importance scores."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importance_scores)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()
        
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance Score')
        
        plt.tight_layout()
        return fig
