"""
Decision Tree model for time series prediction.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from .base_model import TimeSeriesModel

class DecisionTreeModel(TimeSeriesModel):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__()
        self.max_depth = max_depth
        self.kwargs = kwargs
        self.model = None
    
    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Decision tree takes standard (n_samples, n_features) input.
        No reshaping needed, but validate input format.
        """
        return X
    
    def _inverse_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Decision tree outputs in standard format.
        No reshaping needed.
        """
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeModel':
        """
        Fit the decision tree model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        # Validate input format
        self._validate_input(X)
        
        # Initialize and fit model
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, **self.kwargs)
        self.model.fit(X, y)
        
        # Print model info
        print("\nDecision Tree Configuration:")
        print(f"Max Depth: {self.max_depth}")
        print(f"Feature Count: {X.shape[1]}")
        print(f"Sample Count: {X.shape[0]}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the decision tree model.
        
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
