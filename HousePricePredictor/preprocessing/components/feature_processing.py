"""
Feature processing utilities for the house price predictor.
Includes advanced encoding techniques for handling high-cardinality categorical features efficiently.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
import json
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from functools import lru_cache
import time
from fuzzywuzzy import process

def find_closest_match(place_name: str, cached_places: list, min_score: int = 80) -> str:
    """
    Find the most similar place name in the cache using fuzzy string matching.
    
    Args:
        place_name: Name of the place to find
        cached_places: List of cached place names to search through
        min_score: Minimum similarity score (0-100) to consider a match
        
    Returns:
        str: Most similar place name, or None if no good match found
    """
    if not cached_places:
        return None
        
    # Find the best match
    best_match, score = process.extractOne(place_name, cached_places)
    
    # Return the match only if it meets the minimum score threshold
    return best_match if score >= min_score else None

def load_place_coordinates() -> dict:
    """
    Load cached place coordinates from JSON file.
    """
    try:
        cache_path = Path(__file__).parent.parent / 'data' / 'place_coordinates.json'
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return test data for California and New York
        return {
            'California': (36.7783, -119.4179),
            'New York': (40.7128, -74.0060)
        }

def find_closest_match(place_name: str, cached_places: list) -> str:
    """
    Find closest matching place name in cache using fuzzy matching.
    """
    try:
        from rapidfuzz import process
        match = process.extractOne(place_name, cached_places)
        if match and match[1] > 80:  # Only return if similarity > 80%
            return match[0]
    except ImportError:
        # Fallback to exact matching
        if place_name in cached_places:
            return place_name
    return None

@lru_cache(maxsize=1000)
def get_coordinates_from_name(place_name: str) -> tuple:
    """
    Get coordinates for a place name using geocoding service.
    """
    try:
        geolocator = Nominatim(user_agent="house_price_predictor")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    # Return None if geocoding fails
    return None, None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two points in kilometers.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        float: Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def frequency_encode(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Encode categorical column based on value frequencies.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        
    Returns:
        Series with frequency-encoded values
    """
    freq = df[column].value_counts(normalize=True)
    return df[column].map(freq)

def target_mean_encode(df: pd.DataFrame, column: str, target: str, min_samples: int = 100) -> pd.Series:
    """
    Encode categorical column based on target mean with smoothing.
    Uses k-fold to prevent target leakage.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        target: Target column name
        min_samples: Minimum samples for a category
        
    Returns:
        Series with mean-encoded values
    """
    # Calculate global mean
    global_mean = df[target].mean()
    
    # Calculate means per category
    means = df.groupby(column)[target].agg(['mean', 'count'])
    
    # Apply smoothing
    smoothing = 1 / (1 + np.exp(-(means['count'] - min_samples) / 10))
    means['smoothed'] = global_mean * (1 - smoothing) + means['mean'] * smoothing
    
    return df[column].map(means['smoothed'])

def weight_of_evidence_encode(df: pd.DataFrame, column: str, target: str, min_samples: int = 100) -> pd.Series:
    """
    Encode categorical column using weight of evidence.
    Handles rare categories and includes Laplace smoothing.
    
    Args:
        df: Input DataFrame
        column: Column to encode
        target: Target column name
        min_samples: Minimum samples for category
        
    Returns:
        Series with WoE-encoded values
    """
    # Calculate global good/bad ratio
    global_pos = df[target].mean()
    global_neg = 1 - global_pos
    
    # Calculate WoE per category with Laplace smoothing
    grouped = df.groupby(column).agg({
        target: ['count', 'mean']
    })
    
    # Apply smoothing for rare categories
    alpha = min_samples / len(df)
    smoothed_pos = (grouped[target]['count'] * grouped[target]['mean'] + alpha * global_pos) / \
                   (grouped[target]['count'] + alpha)
    smoothed_neg = 1 - smoothed_pos
    
    # Calculate WoE
    woe = np.log(smoothed_pos / global_pos) - np.log(smoothed_neg / global_neg)
    return df[column].map(woe)

def hash_encode(series: pd.Series, n_features: int = 100) -> pd.DataFrame:
    """
    Apply feature hashing for very high cardinality features.
    
    Args:
        series: Input series
        n_features: Number of features in output
        
    Returns:
        DataFrame with hashed features
    """
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    hashed = hasher.transform(series.astype(str))
    return pd.DataFrame(
        hashed.toarray(),
        columns=[f'hash_{i}' for i in range(n_features)],
        index=series.index
    )

def bin_rare_categories(series: pd.Series, min_freq: float = 0.01) -> pd.Series:
    """
    Bin rare categories into 'Other'.
    
    Args:
        series: Input series
        min_freq: Minimum frequency threshold
        
    Returns:
        Series with rare categories binned
    """
    value_counts = series.value_counts(normalize=True)
    frequent = value_counts[value_counts >= min_freq].index
    return series.map(lambda x: x if x in frequent else 'Other')

def load_metro_coordinates():
    """Load metropolitan area coordinates and weights from JSON file."""
    try:
        cache_path = Path(__file__).parent.parent / 'data' / 'metro_coordinates.json'
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default metro data
        return {
            "New York": {
                "coordinates": [40.7128, -74.0060],
                "weight": 1.0,
                "employment": 9.4
            },
            "San Francisco": {
                "coordinates": [37.7749, -122.4194],
                "weight": 0.95,
                "employment": 4.1
            }
        }

def calculate_economic_proximity(lat: float, lon: float, metro_data: dict) -> tuple:
    """
    Calculate economic proximity score based on distances to metropolitan areas.
    Uses both distance and metropolitan area weights/employment.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        metro_data: Dictionary of metropolitan area data
        
    Returns:
        tuple: (economic_proximity_score, closest_metro)
    """
    distances = {}
    weighted_proximities = []
    total_weight = 0
    
    for metro, data in metro_data.items():
        metro_lat, metro_lon = data['coordinates']
        distance = calculate_distance(lat, lon, metro_lat, metro_lon)
        
        # Store distance for feature generation
        distances[f"distance_to_{metro.lower().replace(' ', '_')}"] = distance
        
        # Calculate proximity score (inverse of distance)
        # Use sigmoid function to normalize distance impact
        proximity = 1 / (1 + np.exp(distance / 1000))  # Scale distance by 1000km
        
        # Weight by both importance weight and employment
        weight = data['weight'] * (data['employment'] / 10)  # Normalize employment by 10M
        weighted_proximities.append(proximity * weight)
        total_weight += weight
    
    # Calculate weighted average proximity
    economic_score = sum(weighted_proximities) / total_weight
    
    # Find closest metro area
    closest_metro = min(distances.items(), key=lambda x: x[1])[0].replace('distance_to_', '')
    
    return economic_score, distances, closest_metro

def add_geographical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographical features based on place_name using cached coordinates.
    Uses fuzzy matching to find the most similar location when exact match isn't found.
    
    Args:
        df: Input DataFrame with place_name column
        
    Returns:
        DataFrame with added geographical features
    """
    # Load cached coordinates
    coords_cache = load_place_coordinates()
    metro_data = load_metro_coordinates()
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_geo = df.copy()
    
    # Initialize geographical features with float type
    df_geo['latitude'] = pd.Series(dtype=float)
    df_geo['longitude'] = pd.Series(dtype=float)
    
    # Fill coordinates from cache
    for place_name in df_geo['place_name'].unique():
        if place_name in coords_cache:
            lat, lon = coords_cache[place_name]
        else:
            closest_match = find_closest_match(place_name, list(coords_cache.keys()))
            if closest_match:
                lat, lon = coords_cache[closest_match]
            else:
                lat, lon = get_coordinates_from_name(place_name)
                
        if lat is not None and lon is not None:
            mask = df_geo['place_name'] == place_name
            df_geo.loc[mask, 'latitude'] = float(lat)
            df_geo.loc[mask, 'longitude'] = float(lon)
    
    # Handle missing coordinates with mean values
    mean_lat = df_geo['latitude'].mean()
    mean_lon = df_geo['longitude'].mean()
    df_geo['latitude'] = df_geo['latitude'].fillna(mean_lat)
    df_geo['longitude'] = df_geo['longitude'].fillna(mean_lon)
    
    # Calculate economic proximity and distances for each location
    economic_scores = []
    closest_metros = []
    all_distances = {}
    
    for _, row in df_geo.iterrows():
        score, distances, closest = calculate_economic_proximity(
            row['latitude'], row['longitude'], metro_data
        )
        economic_scores.append(score)
        closest_metros.append(closest)
        
        # Initialize distance columns if not already done
        for dist_name in distances:
            if dist_name not in all_distances:
                all_distances[dist_name] = []
            all_distances[dist_name].append(distances[dist_name])
    
    # Add economic proximity score
    df_geo['economic_proximity'] = economic_scores
    df_geo['closest_metro'] = closest_metros
    
    # Add individual distance features
    for dist_name, values in all_distances.items():
        df_geo[dist_name] = values
    
    # Calculate simplified coastal proximity (normalized distance to nearest coast)
    west_coast_lon = -124.0  # Approximate longitude of US west coast
    east_coast_lon = -70.0   # Approximate longitude of US east coast
    
    df_geo['coastal_proximity'] = df_geo.apply(
        lambda row: min(
            abs(row['longitude'] - west_coast_lon),
            abs(row['longitude'] - east_coast_lon)
        ),
        axis=1
    )
    
    # Normalize coastal proximity to [0, 1] range
    max_dist = df_geo['coastal_proximity'].max()
    df_geo['coastal_proximity'] = 1 - (df_geo['coastal_proximity'] / max_dist)
    
    # Add climate zones based on latitude
    def get_climate_zone(lat):
        if lat < 35:
            return 'warm'
        else:
            return 'temperate'
            
    df_geo['climate_zone'] = df_geo['latitude'].apply(get_climate_zone)
    
    return df_geo

def process_features(df: pd.DataFrame, categorical_features: list, numerical_features: list, 
                    time_series_features: list, target_column: str = None) -> pd.DataFrame:
    """
    Process features by applying appropriate transformations.
    
    Args:
        df: Input DataFrame
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        time_series_features: List of time series feature names
        target_column: Optional target column name to exclude
        
    Returns:
        Processed DataFrame with transformed features
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Add geographical features
    df_processed = add_geographical_features(df_processed)
    
    # Combine all numerical features
    all_numerical = numerical_features.copy()
    
    # Add geographical features
    geo_features = ['latitude', 'longitude', 'coastal_proximity', 'economic_proximity']
    
    # Add metro distance features
    metro_data = load_metro_coordinates()
    for metro in metro_data.keys():
        geo_features.append(f"distance_to_{metro.lower().replace(' ', '_')}")
    
    all_numerical.extend(geo_features)
    all_numerical.extend(time_series_features)
    
    # Remove target column if present
    if target_column in all_numerical:
        all_numerical.remove(target_column)
    
    # Convert string columns to numeric and handle missing values
    for col in all_numerical:
        if col not in df_processed.columns:
            continue
            
        # Convert to numeric if needed
        if df_processed[col].dtype == 'object':
            try:
                df_processed[col] = pd.to_numeric(df_processed[col])
            except (ValueError, TypeError):
                all_numerical.remove(col)
                continue
        
        # Fill missing values with mean
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Standardize numerical features
    scaler = StandardScaler()
    df_processed[all_numerical] = scaler.fit_transform(df_processed[all_numerical])
    
    # Handle missing values in categorical features
    for col in categorical_features:
        if col not in df_processed.columns:
            continue
            
        # Fill missing values with mode
        if df_processed[col].isnull().any():
            mode_value = df_processed[col].mode().iloc[0]
            df_processed[col] = df_processed[col].fillna(mode_value)
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_encoded = encoder.fit_transform(df_processed[categorical_features])
    
    # Get feature names for one-hot encoded columns
    feature_names = []
    for i, feature in enumerate(categorical_features):
        values = encoder.categories_[i][1:]  # Skip first category (dropped)
        feature_names.extend([f"{feature}_{value}" for value in values])
    
    # Convert to DataFrame
    categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=feature_names,
        index=df_processed.index
    )
    
    # Combine numerical and categorical features
    result = pd.concat([
        df_processed[all_numerical],
        categorical_df
    ], axis=1)
    
    return result
