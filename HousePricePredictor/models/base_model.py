"""
Base class for all time series models in the project.
Handles common functionality and enforces consistent interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, RegressorMixin

class TimeSeriesModel(BaseEstimator, RegressorMixin, ABC):
    @abstractmethod
    def _reshape_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape the standard (n_samples, n_features) input into model-specific format.
        Must be implemented by each model class.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Reshaped array specific to the model's requirements
        """
        pass
    
    @abstractmethod
    def _inverse_reshape(self, X: np.ndarray) -> np.ndarray:
        """
        Convert model-specific shape back to standard (n_samples, n_features) format.
        Must be implemented by each model class.
        
        Args:
            X: Model-specific shaped input
            
        Returns:
            Array of shape (n_samples, n_features)
        """
        pass
    
    def _validate_input(self, X: np.ndarray) -> None:
        """
        Validate that input is in standard format before reshaping.
        
        Args:
            X: Input features
            
        Raises:
            ValueError: If input is not in correct format
        """
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Input must be numpy array. Got {type(X)}")
        
        if len(X.shape) != 2:
            raise ValueError(f"Input must be 2D array of shape (n_samples, n_features). Got shape {X.shape}")
            
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(f"Input must be numeric. Got dtype {X.dtype}")
