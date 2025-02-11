"""
Data loader module for handling various data import operations.
Provides a unified interface for loading and validating different data sources.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from .hud_api import HUDDataFetcher

class DataLoader:
    """Handles loading and basic validation of different data sources."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize DataLoader with base path for data files.
        
        Args:
            base_path: Base directory for data files. Defaults to preprocessing/data.
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent / 'data'
        self.base_path = base_path
        self.hud_fetcher = HUDDataFetcher()
    
    def load_metro_data(self) -> Dict[str, Any]:
        """
        Load metropolitan area data including coordinates and economic indicators.
        
        Returns:
            Dictionary containing metropolitan area data
        """
        metro_path = self.base_path / 'metro_coordinates.json'
        try:
            with open(metro_path, 'r') as f:
                data = json.load(f)
            self._validate_metro_data(data)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Metro coordinates file not found at {metro_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in metro coordinates file: {metro_path}")
    
    def load_house_data(self, filename: str = 'house_data.csv') -> pd.DataFrame:
        """
        Load house price dataset.
        
        Args:
            filename: Name of the house data CSV file
            
        Returns:
            DataFrame containing house data
        """
        file_path = self.base_path / filename
        try:
            df = pd.read_csv(file_path)
            self._validate_house_data(df)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"House data file not found at {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty CSV file: {file_path}")
    
    def load_feature_config(self) -> Dict[str, Any]:
        """
        Load feature processing configuration.
        
        Returns:
            Dictionary containing feature processing settings
        """
        config_path = self.base_path / 'feature_config.json'
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self._validate_feature_config(config)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature config file not found at {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in feature config file: {config_path}")
    
    def get_market_trends(self, state: str, county: Optional[str] = None,
                         year: Optional[int] = None) -> pd.DataFrame:
        """
        Get housing market trends from HUD API.
        
        Args:
            state: State abbreviation (e.g., 'CA')
            county: County name (optional)
            year: Data year (optional)
            
        Returns:
            DataFrame containing market trends
        """
        try:
            return self.hud_fetcher.get_market_data(state, county, year)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch market trends: {str(e)}")
    
    def get_income_data(self, state: str, county: Optional[str] = None,
                       year: Optional[int] = None) -> pd.DataFrame:
        """
        Get income limits data from HUD API.
        
        Args:
            state: State abbreviation
            county: County name (optional)
            year: Data year (optional)
            
        Returns:
            DataFrame containing income limits data
        """
        try:
            return self.hud_fetcher.get_income_limits(state, county, year)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch income data: {str(e)}")
    
    def get_affordability_metrics(self, state: str, county: Optional[str] = None,
                                year: Optional[int] = None) -> pd.DataFrame:
        """
        Get housing affordability metrics from HUD API.
        
        Args:
            state: State abbreviation
            county: County name (optional)
            year: Data year (optional)
            
        Returns:
            DataFrame containing affordability metrics
        """
        try:
            return self.hud_fetcher.get_affordability_data(state, county, year)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch affordability metrics: {str(e)}")
    
    def _validate_metro_data(self, data: Dict[str, Any]) -> None:
        """
        Validate metropolitan area data structure.
        
        Args:
            data: Metropolitan area data to validate
            
        Raises:
            ValueError: If data structure is invalid
        """
        required_fields = {'coordinates', 'weight', 'employment'}
        for metro, info in data.items():
            missing_fields = required_fields - set(info.keys())
            if missing_fields:
                raise ValueError(f"Missing required fields {missing_fields} for metro area: {metro}")
            
            if not isinstance(info['coordinates'], list) or len(info['coordinates']) != 2:
                raise ValueError(f"Invalid coordinates format for metro area: {metro}")
            
            if not isinstance(info['weight'], (int, float)) or not 0 <= info['weight'] <= 1:
                raise ValueError(f"Invalid weight value for metro area: {metro}")
            
            if not isinstance(info['employment'], (int, float)) or info['employment'] < 0:
                raise ValueError(f"Invalid employment value for metro area: {metro}")
    
    def _validate_house_data(self, df: pd.DataFrame) -> None:
        """
        Validate house data structure.
        
        Args:
            df: House data DataFrame to validate
            
        Raises:
            ValueError: If data structure is invalid
        """
        required_columns = {'price', 'location', 'sqft', 'bedrooms', 'bathrooms'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in house data: {missing_columns}")
        
        if df.empty:
            raise ValueError("House data DataFrame is empty")
    
    def _validate_feature_config(self, config: Dict[str, Any]) -> None:
        """
        Validate feature processing configuration.
        
        Args:
            config: Feature configuration to validate
            
        Raises:
            ValueError: If configuration structure is invalid
        """
        required_sections = {'numerical_features', 'categorical_features', 'geo_features'}
        missing_sections = required_sections - set(config.keys())
        if missing_sections:
            raise ValueError(f"Missing required sections in feature config: {missing_sections}")
        
        for feature_type in required_sections:
            if not isinstance(config[feature_type], list):
                raise ValueError(f"Invalid format for {feature_type} in feature config")
