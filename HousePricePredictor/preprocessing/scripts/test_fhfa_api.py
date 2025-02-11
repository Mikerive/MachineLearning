"""
Test script for FHFA HPI data integration.
Tests data fetching and processing capabilities.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from preprocessing.data_handlers.fhfa_api import FHFADataFetcher

def test_metro_data():
    """Test fetching metropolitan area HPI data."""
    print("\nTesting Metro Area Data...")
    fetcher = FHFADataFetcher()
    
    # Test major tech hubs
    tech_hubs = [
        "San Francisco",
        "Seattle",
        "Austin",
        "Boston"
    ]
    
    print("\nTech Hub Housing Price Trends:")
    for metro in tech_hubs:
        try:
            data = fetcher.get_metro_hpi(metro)
            print(f"\n{metro}:")
            if not data.empty:
                latest = data.iloc[-1]
                print(f"- Latest HPI: {latest['Index Value']:.2f}")
                print(f"- Year: {latest['year']}")
                print(f"- Annual Change: {latest['Annual Change (%)']:.1f}%")
        except Exception as e:
            print(f"[X] Error fetching data for {metro}: {str(e)}")

def test_state_data():
    """Test fetching state-level HPI data."""
    print("\nTesting State-Level Data...")
    fetcher = FHFADataFetcher()
    
    # Test major states
    states = ["CA", "NY", "TX", "FL"]
    
    print("\nState Housing Price Trends:")
    for state in states:
        try:
            data = fetcher.get_state_hpi(state)
            print(f"\n{state}:")
            if not data.empty:
                latest = data.iloc[-1]
                print(f"- Latest HPI: {latest['Index Value']:.2f}")
                print(f"- Year: {latest['year']}")
                print(f"- Annual Change: {latest['Annual Change (%)']:.1f}%")
        except Exception as e:
            print(f"[X] Error fetching data for {state}: {str(e)}")

def test_county_data():
    """Test fetching county-level HPI data."""
    print("\nTesting County-Level Data...")
    fetcher = FHFADataFetcher()
    
    # Test major counties
    counties = [
        {"state": "CA", "county": "Los Angeles"},
        {"state": "NY", "county": "New York"},
        {"state": "IL", "county": "Cook"},
        {"state": "TX", "county": "Travis"}
    ]
    
    print("\nCounty Housing Price Trends:")
    for loc in counties:
        try:
            data = fetcher.get_county_hpi(loc["state"], loc["county"])
            print(f"\n{loc['county']}, {loc['state']}:")
            if not data.empty:
                latest = data.iloc[-1]
                print(f"- Latest HPI: {latest['Index Value']:.2f}")
                print(f"- Year: {latest['year']}")
                print(f"- Annual Change: {latest['Annual Change (%)']:.1f}%")
        except Exception as e:
            print(f"[X] Error fetching data for {loc['county']}: {str(e)}")

def test_price_trends():
    """Test price trend analysis functionality."""
    print("\nTesting Price Trend Analysis...")
    fetcher = FHFADataFetcher()
    
    # Test different location types
    locations = [
        {"state": "CA"},
        {"metro": "San Francisco"},
        {"state": "NY", "county": "New York"},
    ]
    
    print("\nPrice Trends (Last 5 Years):")
    for loc in locations:
        try:
            loc_str = ", ".join(f"{k}: {v}" for k, v in loc.items())
            data = fetcher.get_price_trends(loc, start_year=2020)
            print(f"\n{loc_str}:")
            if not data.empty:
                latest = data.iloc[-1]
                earliest = data.iloc[0]
                total_change = ((latest['Index Value'] / earliest['Index Value']) - 1) * 100
                print(f"- Total Change (2020-Present): {total_change:.1f}%")
                print(f"- Latest HPI: {latest['Index Value']:.2f}")
                print(f"- Annual Change: {latest['Annual Change (%)']:.1f}%")
        except Exception as e:
            print(f"[X] Error analyzing trends for {loc_str}: {str(e)}")

def main():
    """Run all tests and demonstrations."""
    print("=== FHFA House Price Index Tests ===")
    
    test_metro_data()
    test_state_data()
    test_county_data()
    test_price_trends()
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    main()
