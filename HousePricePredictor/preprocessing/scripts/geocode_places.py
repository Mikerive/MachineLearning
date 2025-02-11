"""
Script to batch geocode all unique places in the dataset and save to a JSON file.
This only needs to be run once to create the coordinates cache.
"""

import pandas as pd
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
import time
from pathlib import Path

def batch_geocode_places(csv_path: str, output_path: str):
    """
    Batch geocode all unique places in the dataset and save to JSON.
    
    Args:
        csv_path: Path to the raw CSV data file
        output_path: Path to save the coordinates JSON file
    """
    # Read the dataset
    df = pd.read_csv(csv_path)
    unique_places = df['place_name'].unique()
    print(f"Found {len(unique_places)} unique places to geocode")
    
    # Initialize geocoder with rate limiting
    geolocator = Nominatim(user_agent="house_price_predictor_batch")
    # 1 second between calls to respect API limits
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Create coordinates dictionary
    coordinates = {}
    failed_places = []
    
    for i, place in enumerate(unique_places, 1):
        try:
            print(f"Geocoding {i}/{len(unique_places)}: {place}")
            # Add ", USA" to improve accuracy
            location = geocode(f"{place}, USA", timeout=10)
            if location:
                coordinates[place] = {
                    "latitude": location.latitude,
                    "longitude": location.longitude
                }
            else:
                failed_places.append(place)
                print(f"Warning: No coordinates found for {place}")
        except (GeocoderTimedOut, Exception) as e:
            failed_places.append(place)
            print(f"Error geocoding {place}: {str(e)}")
        
        # Save progress every 10 places in case of interruption
        if i % 10 == 0:
            with open(output_path, 'w') as f:
                json.dump({
                    "coordinates": coordinates,
                    "failed_places": failed_places,
                    "total_processed": i,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump({
            "coordinates": coordinates,
            "failed_places": failed_places,
            "total_processed": len(unique_places),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\nGeocoding completed!")
    print(f"Successfully geocoded: {len(coordinates)} places")
    print(f"Failed to geocode: {len(failed_places)} places")
    if failed_places:
        print("\nFailed places:")
        for place in failed_places:
            print(f"- {place}")

if __name__ == "__main__":
    # Create data/processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    batch_geocode_places(
        csv_path="data/raw/hpi_master.csv",
        output_path="data/processed/place_coordinates.json"
    )
