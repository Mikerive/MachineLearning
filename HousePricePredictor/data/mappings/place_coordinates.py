"""
Mapping of place IDs and names to their geographical coordinates.
Coordinates are approximate centers for Census Divisions and MSAs.
"""

PLACE_COORDINATES = {
    # Census Divisions
    "DV_ENC": {"lat": 43.0000, "lon": -85.7500},  # East North Central
    "DV_ESC": {"lat": 33.0000, "lon": -86.0000},  # East South Central
    "DV_MA": {"lat": 41.0000, "lon": -75.0000},   # Middle Atlantic
    "DV_MTN": {"lat": 39.0000, "lon": -110.0000}, # Mountain
    "DV_NE": {"lat": 43.0000, "lon": -71.0000},   # New England
    "DV_PAC": {"lat": 44.0000, "lon": -120.0000}, # Pacific
    "DV_SA": {"lat": 33.0000, "lon": -81.0000},   # South Atlantic
    "DV_WNC": {"lat": 43.0000, "lon": -96.0000},  # West North Central
    "DV_WSC": {"lat": 32.0000, "lon": -97.0000},  # West South Central
    
    # Example MSAs (add more as needed)
    "30980": {"lat": 32.5007, "lon": -94.7405},   # Longview, TX
    # Add more MSA mappings here
}

def get_coordinates(place_id: str) -> tuple:
    """
    Get the latitude and longitude for a given place ID.
    
    Args:
        place_id: The place ID from the dataset
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    if place_id in PLACE_COORDINATES:
        coords = PLACE_COORDINATES[place_id]
        return coords["lat"], coords["lon"]
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
