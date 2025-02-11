#!/usr/bin/env python3
"""
Script to build a cache of metropolitan area coordinates using OpenStreetMap's Nominatim API.
Includes employment data and economic weights for major US metropolitan areas.
"""

import json
import time
from pathlib import Path
import requests
from typing import Dict, Any, Optional, Tuple

def load_metro_data() -> Dict[str, Any]:
    """Load metropolitan area data from JSON file."""
    data_path = Path(__file__).parent.parent / 'data' / 'metro_coordinates_v2.json'
    with open(data_path, 'r') as f:
        return json.load(f)

def get_coordinates(query: str) -> Optional[Tuple[float, float]]:
    """
    Get coordinates for a location using Nominatim API.
    Respects usage policy with proper user agent and rate limiting.
    
    Args:
        query: Location query string
        
    Returns:
        Tuple of (latitude, longitude) if found, None otherwise
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    
    headers = {
        'User-Agent': 'HousePricePredictorApp/1.0 (https://github.com/Mikerive/MachineLearning)'
    }
    
    params = {
        'q': query,
        'format': 'json',
        'limit': 1
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        
        results = response.json()
        if results:
            lat = float(results[0]['lat'])
            lon = float(results[0]['lon'])
            return lat, lon
        
    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        print(f"Error fetching coordinates for {query}: {str(e)}")
    
    return None

def build_metro_cache() -> Dict[str, Any]:
    """
    Build cache of metropolitan area data including coordinates.
    
    Returns:
        Dictionary of metropolitan areas with coordinates and metadata
    """
    metro_data = load_metro_data()
    cache = {}
    
    for metro, data in metro_data.items():
        print(f"Fetching coordinates for {metro}...")
        
        coords = get_coordinates(data['query'])
        if coords:
            lat, lon = coords
            cache[metro] = {
                "coordinates": [lat, lon],
                "weight": data['weight'],
                "employment": data['employment']
            }
        else:
            print(f"Warning: Could not fetch coordinates for {metro}")
        
        # Rate limiting: 1 request per second as per Nominatim usage policy
        time.sleep(1)
    
    return cache

def save_cache(cache: Dict[str, Any], filename: str = "metro_coordinates.json") -> None:
    """
    Save metropolitan area cache to JSON file.
    
    Args:
        cache: Dictionary of metropolitan area data
        filename: Output filename
    """
    cache_path = Path(__file__).parent.parent / 'data' / filename
    
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=4)
    
    print(f"Cache saved to {cache_path}")

def main():
    """Main function to build and save metropolitan area cache."""
    print("Building metropolitan area coordinate cache...")
    cache = build_metro_cache()
    
    if cache:
        save_cache(cache)
        print(f"Successfully cached coordinates for {len(cache)} metropolitan areas")
    else:
        print("Error: No coordinates were cached")

if __name__ == "__main__":
    main()
