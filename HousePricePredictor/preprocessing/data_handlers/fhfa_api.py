"""
Module for fetching and processing FHFA House Price Index (HPI) data.
Data source: https://www.fhfa.gov/data/hpi
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import zipfile
import io
import time

class FHFADataFetcher:
    """Handles fetching and processing of FHFA House Price Index data."""
    
    # FHFA HPI dataset URLs (updated for 2024)
    DATASETS = {
        'metro': 'https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_metro.xlsx',
        'state': 'https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_state.xlsx',
        'county': 'https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_county.xlsx',
        'zip': 'https://www.fhfa.gov/DataTools/Downloads/Documents/HPI/HPI_PO_ZIP5.xlsx'
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize FHFA data fetcher with caching.
        
        Args:
            cache_dir: Directory for caching downloaded data
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'data' / 'fhfa_cache'
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _download_dataset(self, dataset_type: str, refresh: bool = False) -> pd.DataFrame:
        """
        Download and cache FHFA dataset.
        
        Args:
            dataset_type: Type of dataset ('metro', 'state', 'county', or 'zip')
            refresh: Force refresh cached data
            
        Returns:
            DataFrame containing the requested dataset
        """
        if dataset_type not in self.DATASETS:
            raise ValueError(f"Invalid dataset type. Must be one of: {list(self.DATASETS.keys())}")
        
        cache_file = self.cache_dir / f"hpi_{dataset_type}.csv"
        
        # Return cached data if available and fresh
        if not refresh and cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 24 * 60 * 60:  # 24 hours
                return pd.read_csv(cache_file)
        
        # Download fresh data
        print(f"Downloading FHFA {dataset_type} HPI dataset...")
        response = requests.get(self.DATASETS[dataset_type])
        response.raise_for_status()
        
        # Parse Excel file directly from response content
        df = pd.read_excel(io.BytesIO(response.content))
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Cache the data
        df.to_csv(cache_file, index=False)
        
        return df
    
    def get_metro_hpi(self, metro_area: str = None, refresh: bool = False) -> pd.DataFrame:
        """
        Get House Price Index data for metropolitan areas.
        
        Args:
            metro_area: Metropolitan area name (optional)
            refresh: Force refresh cached data
            
        Returns:
            DataFrame with HPI data
        """
        df = self._download_dataset('metro', refresh)
        
        if metro_area:
            df = df[df['MSA Name'].str.contains(metro_area, case=False, na=False)]
            
        if df.empty:
            raise ValueError(f"No data found for metro area: {metro_area}")
            
        return df
    
    def get_state_hpi(self, state: str = None, refresh: bool = False) -> pd.DataFrame:
        """
        Get House Price Index data by state.
        
        Args:
            state: State name or abbreviation (optional)
            refresh: Force refresh cached data
            
        Returns:
            DataFrame with HPI data
        """
        df = self._download_dataset('state', refresh)
        
        if state:
            state_filter = (df['State'].str.contains(state, case=False, na=False) | 
                          df['State Code'].str.contains(state, case=False, na=False))
            df = df[state_filter]
            
        if df.empty:
            raise ValueError(f"No data found for state: {state}")
            
        return df
    
    def get_county_hpi(self, state: str = None, county: str = None,
                      refresh: bool = False) -> pd.DataFrame:
        """
        Get House Price Index data by county.
        
        Args:
            state: State name or abbreviation (optional)
            county: County name (optional)
            refresh: Force refresh cached data
            
        Returns:
            DataFrame with HPI data
        """
        df = self._download_dataset('county', refresh)
        
        if state:
            state_filter = (df['State'].str.contains(state, case=False, na=False) | 
                          df['State Code'].str.contains(state, case=False, na=False))
            df = df[state_filter]
            
        if county:
            df = df[df['County Name'].str.contains(county, case=False, na=False)]
            
        if df.empty:
            raise ValueError(f"No data found for state: {state}, county: {county}")
            
        return df
    
    def get_zip_hpi(self, zip_code: str = None, refresh: bool = False) -> pd.DataFrame:
        """
        Get House Price Index data by ZIP code.
        
        Args:
            zip_code: 5-digit ZIP code (optional)
            refresh: Force refresh cached data
            
        Returns:
            DataFrame with HPI data
        """
        df = self._download_dataset('zip', refresh)
        
        if zip_code:
            df = df[df['ZIP Code'] == zip_code]
            
        if df.empty:
            raise ValueError(f"No data found for ZIP code: {zip_code}")
            
        return df
    
    def get_price_trends(self, location: Dict[str, str],
                        start_year: Optional[int] = None,
                        end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Get price trends for a specific location.
        
        Args:
            location: Dictionary with location info (e.g., {'state': 'CA', 'county': 'Los Angeles'})
            start_year: Start year for trend analysis (optional)
            end_year: End year for trend analysis (optional)
            
        Returns:
            DataFrame with price trends
        """
        if 'zip' in location:
            df = self.get_zip_hpi(location['zip'])
        elif 'county' in location:
            df = self.get_county_hpi(location.get('state'), location['county'])
        elif 'metro' in location:
            df = self.get_metro_hpi(location['metro'])
        elif 'state' in location:
            df = self.get_state_hpi(location['state'])
        else:
            raise ValueError("Location must specify zip, county, metro, or state")
        
        # Filter by year if specified
        if 'Year' in df.columns:
            if start_year:
                df = df[df['Year'] >= start_year]
            if end_year:
                df = df[df['Year'] <= end_year]
        
        return df.sort_values('Year') if 'Year' in df.columns else df
