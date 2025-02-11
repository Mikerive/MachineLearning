"""
Test script for HUD API integration.
Tests API connection, token handling, and data fetching capabilities.
"""

import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from preprocessing.data_handlers.data_loader import DataLoader
from preprocessing.data_handlers.hud_api import HUDDataFetcher

def test_api_connection():
    """Test basic API connectivity and token validation."""
    print("\nTesting API Connection...")
    fetcher = HUDDataFetcher()
    
    if not fetcher.api_token:
        print("[X] API token not found!")
        return False
    
    try:
        # Test with a simple query
        data = fetcher.get_market_data("CA", "Los Angeles")
        print("[+] Successfully connected to HUD API")
        print(f"[+] Retrieved {len(data)} records")
        return True
    except Exception as e:
        print(f"[X] API connection failed: {str(e)}")
        return False

def fetch_metro_data():
    """Fetch data for major metropolitan areas."""
    print("\nFetching Metropolitan Area Data...")
    loader = DataLoader()
    
    # Major tech hubs
    tech_hubs = [
        ("CA", "San Francisco"),
        ("WA", "King"),  # Seattle
        ("TX", "Travis"), # Austin
        ("MA", "Suffolk"), # Boston
    ]
    
    print("\nTech Hub Market Trends (2024):")
    for state, county in tech_hubs:
        try:
            data = loader.get_market_trends(state, county)
            if not data.empty:
                print(f"\n{county}, {state}:")
                if 'rent_50_pct' in data.columns:
                    print(f"- Median Rent: ${data['rent_50_pct'].iloc[0]:,.2f}")
                if 'rent_40_pct' in data.columns:
                    print(f"- Lower Market: ${data['rent_40_pct'].iloc[0]:,.2f}")
                if 'rent_60_pct' in data.columns:
                    print(f"- Upper Market: ${data['rent_60_pct'].iloc[0]:,.2f}")
        except Exception as e:
            print(f"[X] Error fetching data for {county}: {str(e)}")

def analyze_affordability():
    """Analyze housing affordability across regions."""
    print("\nAnalyzing Housing Affordability...")
    loader = DataLoader()
    
    # Major economic centers
    cities = [
        ("NY", "New York"),
        ("CA", "Los Angeles"),
        ("IL", "Cook"),      # Chicago
        ("TX", "Harris"),    # Houston
    ]
    
    print("\nAffordability Metrics (2024):")
    for state, county in cities:
        try:
            metrics = loader.get_affordability_metrics(state, county)
            income_data = loader.get_income_data(state, county)
            
            if not metrics.empty and not income_data.empty:
                print(f"\n{county}, {state}:")
                if 'median_income' in income_data.columns:
                    print(f"- Median Income: ${income_data['median_income'].iloc[0]:,.2f}")
                if 'vlil_limit' in income_data.columns:
                    print(f"- Very Low Income Limit: ${income_data['vlil_limit'].iloc[0]:,.2f}")
                if 'cost_burden' in metrics.columns:
                    print(f"- Cost Burdened Households: {metrics['cost_burden'].iloc[0]:.1f}%")
        except Exception as e:
            print(f"[X] Error analyzing {county}: {str(e)}")

def test_token_refresh():
    """Test token refresh handling."""
    print("\nTesting Token Refresh...")
    fetcher = HUDDataFetcher()
    
    if not fetcher.api_token:
        print("[X] No token to test refresh")
        return
    
    try:
        # Check token expiration
        import jwt
        token_data = jwt.decode(fetcher.api_token, options={"verify_signature": False})
        exp_time = datetime.fromtimestamp(token_data['exp'])
        days_until_expiry = (exp_time - datetime.now()).days
        
        print(f"[+] Token valid for {days_until_expiry} more days")
        if days_until_expiry < 30:
            print("[!] Token will expire soon. Consider refreshing.")
    except Exception as e:
        print(f"[X] Error checking token: {str(e)}")

def main():
    """Run all tests and demonstrations."""
    print("=== HUD API Integration Tests ===")
    
    if not test_api_connection():
        print("\n[X] Basic API connection failed. Stopping tests.")
        return
    
    fetch_metro_data()
    analyze_affordability()
    test_token_refresh()
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    main()
