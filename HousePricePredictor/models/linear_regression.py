"""
Linear Regression model for time series prediction.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from .base_model import TimeSeriesModel

class LinearRegressionModel(TimeSeriesModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.model = None
    
    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Linear regression takes standard (n_samples, n_features) input.
        No reshaping needed, but validate input format.
        """
        return X
    
    def _inverse_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Linear regression outputs in standard format.
        No reshaping needed.
        """
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        """
        Fit the linear regression model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        # Validate input format
        self._validate_input(X)
        
        # Initialize and fit model
        self.model = LinearRegression(**self.kwargs)
        self.model.fit(X, y)
        
        # Print model info
        print("\nLinear Regression Configuration:")
        print(f"Feature Count: {X.shape[1]}")
        print(f"Sample Count: {X.shape[0]}")
        print(f"R² Score: {self.model.score(X, y):.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the linear regression model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        # Validate input format
        self._validate_input(X)
        
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
