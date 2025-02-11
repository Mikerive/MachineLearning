"""
Build and maintain a cache of place coordinates using OpenStreetMap's Nominatim service.
This script processes unique place names and stores their coordinates in a JSON file.
"""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import time
import sys
from ..components.feature_processing import get_coordinates_from_name

def build_coordinate_cache(data_path: str, output_path: str):
    """
    Build a cache of place coordinates from the dataset.
    
    Args:
        data_path: Path to the raw data CSV file
        output_path: Path to save the coordinates JSON file
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input data exists
    if not data_path.exists():
        print(f"Error: Input data file not found at {data_path}")
        print("Please ensure your raw data CSV file is in the correct location.")
        print("Expected structure:")
        print("  data/")
        print("    raw/")
        print("      house_data.csv")
        print("    processed/")
        print("      place_coordinates.json")
        sys.exit(1)
    
    # Load existing cache if available
    existing_cache = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_cache = json.load(f)
            print(f"Loaded existing cache with {len(existing_cache)} entries")
    
    # Load and get unique places
    df = pd.read_csv(data_path)
    if 'place_name' not in df.columns:
        print("Error: Input data must contain a 'place_name' column")
        sys.exit(1)
        
    unique_places = df['place_name'].unique()
    print(f"Found {len(unique_places)} unique places in dataset")
    
    # Process places not in cache
    new_coordinates = {}
    for place in tqdm(unique_places, desc="Geocoding places"):
        if place in existing_cache:
            continue
            
        lat, lon = get_coordinates_from_name(place)
        if lat is not None and lon is not None:
            new_coordinates[place] = {'latitude': lat, 'longitude': lon}
        
        # Rate limiting to respect Nominatim's usage policy
        time.sleep(1)
    
    # Merge and save updated cache
    updated_cache = {**existing_cache, **new_coordinates}
    with open(output_path, 'w') as f:
        json.dump(updated_cache, f, indent=2)
    
    print(f"Updated coordinate cache with {len(new_coordinates)} new entries")
    print(f"Total places in cache: {len(updated_cache)}")

if __name__ == "__main__":
    build_coordinate_cache(
        data_path=Path(__file__).parent.parent.parent / "data/raw/hpi_master.csv",
        output_path=Path(__file__).parent.parent.parent / "data/processed/place_coordinates.json"
    )
