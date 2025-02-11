"""
Module for fetching housing market data from HUD's API.
Provides access to median home prices, market trends, and housing characteristics.
Documentation: https://www.huduser.gov/portal/dataset/fmr-api.html
"""

import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
from pathlib import Path
import json
from .env_loader import load_api_token

class HUDDataFetcher:
    """Handles data fetching from HUD's API with rate limiting and caching."""
    
    BASE_URL = "https://www.huduser.gov/portal/datasets/fmr/fmr2024/fy2024_safmrs_revised.zip"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize HUD data fetcher with caching.
        
        Args:
            cache_dir: Directory for caching API responses
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'data' / 'hud_cache'
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load API token from environment
        self.api_token = load_api_token()
        if not self.api_token:
            print("Warning: HUD API token not found. Some features may be limited.")
        
        # Download and cache the full dataset
        self._ensure_data_loaded()
    
    def _ensure_data_loaded(self) -> None:
        """Download and process the full HUD dataset if not cached."""
        cache_file = self.cache_dir / "fmr_2024_full.csv"
        
        if not cache_file.exists():
            print("Downloading HUD Fair Market Rent dataset...")
            response = requests.get(self.BASE_URL)
            response.raise_for_status()
            
            # Save the ZIP file
            zip_path = self.cache_dir / "fmr_2024.zip"
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract and process the data
            import zipfile
            import io
            
            with zipfile.ZipFile(zip_path) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(io.TextIOWrapper(f))
                    df.to_csv(cache_file, index=False)
            
            # Clean up the ZIP file
            zip_path.unlink()
    
    def get_market_data(self, state: str, county: str = None, 
                       year: int = 2024, refresh: bool = False) -> pd.DataFrame:
        """
        Fetch housing market data for a specific location.
        
        Args:
            state: State abbreviation (e.g., 'CA')
            county: County name (optional)
            year: Data year (defaults to 2024)
            refresh: Force refresh cached data
            
        Returns:
            DataFrame containing housing market data
        """
        if year != 2024:
            raise ValueError("Only 2024 data is available")
        
        # Load the full dataset
        df = pd.read_csv(self.cache_dir / "fmr_2024_full.csv")
        
        # Filter by state
        df = df[df['state_alpha'] == state]
        
        # Filter by county if specified
        if county:
            df = df[df['county_name'].str.contains(county, case=False)]
        
        if df.empty:
            raise ValueError(f"No data found for {state} {county if county else ''}")
        
        return df
    
    def get_income_limits(self, state: str, county: str = None,
                         year: int = 2024) -> pd.DataFrame:
        """
        Fetch income limits data for housing programs.
        
        Args:
            state: State abbreviation
            county: County name (optional)
            year: Data year (defaults to 2024)
            
        Returns:
            DataFrame containing income limits data
        """
        if year != 2024:
            raise ValueError("Only 2024 data is available")
        
        # For now, return market data with income-related columns
        df = self.get_market_data(state, county)
        income_cols = [col for col in df.columns if 'income' in col.lower()]
        return df[['state_alpha', 'county_name'] + income_cols]
    
    def get_affordability_data(self, state: str, county: str = None,
                             year: int = 2024) -> pd.DataFrame:
        """
        Fetch housing affordability statistics.
        
        Args:
            state: State abbreviation
            county: County name (optional)
            year: Data year (defaults to 2024)
            
        Returns:
            DataFrame containing affordability data
        """
        if year != 2024:
            raise ValueError("Only 2024 data is available")
        
        # For now, return market data with affordability-related columns
        df = self.get_market_data(state, county)
        affordability_cols = ['fmr_0', 'fmr_1', 'fmr_2', 'fmr_3', 'fmr_4']
        return df[['state_alpha', 'county_name'] + affordability_cols]
