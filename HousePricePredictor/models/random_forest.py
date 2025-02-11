"""
Random Forest model for time series prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base_model import TimeSeriesModel

class RandomForestModel(TimeSeriesModel):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        **kwargs
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.kwargs = kwargs
        self.model = None
    
    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Random forest takes standard (n_samples, n_features) input.
        No reshaping needed, but validate input format.
        """
        return X
    
    def _inverse_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Random forest outputs in standard format.
        No reshaping needed.
        """
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestModel':
        """
        Fit the random forest model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target array of shape (n_samples,)
        """
        # Validate input format
        self._validate_input(X)
        
        # Initialize and fit model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            **self.kwargs
        )
        self.model.fit(X, y)
        
        # Print model info
        print("\nRandom Forest Configuration:")
        print(f"Number of Trees: {self.n_estimators}")
        print(f"Max Depth: {self.max_depth}")
        print(f"Min Samples Split: {self.min_samples_split}")
        print(f"Min Samples Leaf: {self.min_samples_leaf}")
        print(f"Feature Count: {X.shape[1]}")
        print(f"Sample Count: {X.shape[0]}")
        
        # Print feature importances
        importances = sorted(zip(self.model.feature_importances_, X.columns), reverse=True)
        print("\nTop 10 Feature Importances:")
        for importance, feature in importances[:10]:
            print(f"{feature}: {importance:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the random forest model.
        
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
