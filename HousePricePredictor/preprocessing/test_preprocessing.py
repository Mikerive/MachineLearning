"""
Test script for preprocessing functionality.
Tests the feature processing pipeline, geographical features, and data consistency.
"""

import unittest
import pandas as pd
import numpy as np
import os
from preprocessing.preprocessing import load_and_preprocess_data
from preprocessing.components.feature_processing import (
    add_geographical_features, process_features,
    load_metro_coordinates
)
from preprocessing.components.feature_engineering import create_time_series_features

class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample data with known properties
        self.sample_data = pd.DataFrame({
            'hpi_type': ['traditional', 'distress-free'] * 50,
            'hpi_flavor': ['purchase-only'] * 100,
            'frequency': ['quarterly'] * 100,
            'level': ['State'] * 100,
            'place_name': ['California', 'New York'] * 50,
            'place_id': ['CA', 'NY'] * 50,
            'yr': list(range(2000, 2025)) * 4,
            'period': [1, 2, 3, 4] * 25,
            'index_nsa': np.random.normal(200, 20, 100),
            'index_sa': np.random.normal(200, 20, 100)
        })
        
        # Add some missing values
        self.sample_data.loc[0:10, 'index_sa'] = np.nan
        
        # Save sample data
        self.sample_data.to_csv('data/test_data.csv', index=False)
        
    def test_load_and_preprocess(self):
        """Test the main preprocessing pipeline."""
        X_train, X_test, y_train, y_test = load_and_preprocess_data(
            'data/test_data.csv',
            'index_nsa'
        )
        
        # Check shapes
        self.assertEqual(len(y_train) + len(y_test), len(self.sample_data))
        self.assertTrue(len(X_train.columns) > len(self.sample_data.columns))
        
        # Check feature types
        self.assertTrue(all(X_train.dtypes == 'float64'))
        self.assertTrue(all(X_test.dtypes == 'float64'))
        
        # Check no missing values
        self.assertTrue(X_train.isnull().sum().sum() == 0)
        self.assertTrue(X_test.isnull().sum().sum() == 0)
        
    def test_geographical_features(self):
        """Test geographical feature generation."""
        # Create test data
        test_data = pd.DataFrame({
            'place_name': ['New York', 'San Francisco', 'Chicago'],
            'place_id': ['NY1', 'SF1', 'CH1']
        })
        
        # Generate geographical features
        df_geo = add_geographical_features(test_data)
        
        # Check if all required features are present
        required_features = [
            'latitude', 'longitude', 'economic_proximity',
            'closest_metro', 'coastal_proximity', 'climate_zone'
        ]
        for feature in required_features:
            self.assertIn(feature, df_geo.columns)
        
        # Check if distance features are present for all metros
        metro_data = load_metro_coordinates()
        for metro in metro_data.keys():
            dist_col = f"distance_to_{metro.lower().replace(' ', '_')}"
            self.assertIn(dist_col, df_geo.columns)
        
        # Check if values are within expected ranges
        self.assertTrue(all(df_geo['latitude'].between(-90, 90)))
        self.assertTrue(all(df_geo['longitude'].between(-180, 180)))
        self.assertTrue(all(df_geo['economic_proximity'].between(0, 1)))
        self.assertTrue(all(df_geo['coastal_proximity'].between(0, 1)))
        
        # Test with sample data
        df_with_geo = add_geographical_features(self.sample_data)
        
        # Check required geographical features
        required_features = [
            'latitude', 'longitude', 'economic_proximity',
            'closest_metro', 'coastal_proximity', 'climate_zone'
        ]
        for feature in required_features:
            self.assertIn(feature, df_with_geo.columns)
            
        # Check if distance features are present
        for metro in metro_data.keys():
            dist_col = f"distance_to_{metro.lower().replace(' ', '_')}"
            self.assertIn(dist_col, df_with_geo.columns)
        
        # Check value ranges
        self.assertTrue(df_with_geo['latitude'].between(-90, 90).all())
        self.assertTrue(df_with_geo['longitude'].between(-180, 180).all())
        self.assertTrue(df_with_geo['coastal_proximity'].between(0, 1).all())
        
        # Check climate zones are valid
        self.assertTrue(df_with_geo['climate_zone'].isin(['temperate', 'warm']).all())
        
        # Check no missing values
        self.assertTrue(df_with_geo[required_features].notna().all().all())

    def test_feature_processing(self):
        """Test feature processing with different feature types."""
        # Define feature groups
        categorical = ['hpi_type', 'hpi_flavor', 'frequency', 'level', 'place_id']
        numerical = ['yr', 'period']
        time_series = ['index_sa']
        
        processed = process_features(
            self.sample_data,
            categorical,
            numerical,
            time_series,
            'index_nsa'
        )
        
        # Check output properties
        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(len(processed), len(self.sample_data))
        
        # Check one-hot encoding
        self.assertTrue(any(col.startswith('hpi_type_') for col in processed.columns))
        
        # Check numerical features
        for num_feature in numerical:
            self.assertIn(num_feature, processed.columns)
            
        # Check no missing values
        self.assertTrue(processed.isnull().sum().sum() == 0)
        
    def test_time_series_features(self):
        """Test time series feature generation."""
        df_with_ts = create_time_series_features(self.sample_data, target_column='index_nsa')
        
        # Check time series features
        ts_features = [
            'rolling_mean_3', 'rolling_std_3',
            'lag_1_price', 'momentum_3',
            'seasonal', 'trend', 'residual'
        ]
        for feature in ts_features:
            self.assertIn(feature, df_with_ts.columns)
        
        # Check rolling statistics
        self.assertTrue(df_with_ts['rolling_mean_3'].std() <= self.sample_data['index_nsa'].std())
        self.assertTrue(len(df_with_ts) == len(self.sample_data))
        
        # Check no missing values
        self.assertTrue(df_with_ts[ts_features].notna().all().all())
        
    def tearDown(self):
        """Clean up test data."""
        if os.path.exists('data/test_data.csv'):
            os.remove('data/test_data.csv')

if __name__ == '__main__':
    unittest.main()
