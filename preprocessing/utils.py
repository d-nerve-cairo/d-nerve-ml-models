"""
Utility functions shared across ML pipeline

Author: Group 2 - ML Team
Environment: dnervenv
"""

import yaml
import pandas as pd
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

def load_config(config_path='config/config.yaml'):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        dict: Configuration dictionary
    """
    # Handle both Windows and Unix paths
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate great-circle distance between two GPS points
    
    Args:
        lon1, lat1: First point (decimal degrees)
        lon2, lat2: Second point (decimal degrees)
    
    Returns:
        float: Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    km = 6371 * c  # Earth radius in km
    return km

def calculate_speed(lon1, lat1, lon2, lat2, time_diff_seconds):
    """
    Calculate speed between two GPS points
    
    Args:
        lon1, lat1, lon2, lat2: Coordinates (decimal degrees)
        time_diff_seconds: Time difference in seconds
    
    Returns:
        float: Speed in km/h
    """
    if time_diff_seconds == 0:
        return 0.0
    
    distance_km = haversine_distance(lon1, lat1, lon2, lat2)
    time_hours = time_diff_seconds / 3600
    
    return distance_km / time_hours if time_hours > 0 else 0.0

def create_output_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_dataframe(df, filepath, format='parquet'):
    """
    Save DataFrame in specified format
    
    Args:
        df: pandas DataFrame
        filepath: Output file path
        format: 'parquet', 'csv', or 'pickle'
    """
    filepath = Path(filepath)
    create_output_dir(filepath.parent)
    
    if format == 'parquet':
        df.to_parquet(filepath, compression='snappy', index=False)
    elif format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"💾 Saved to {filepath} ({file_size_mb:.2f} MB)")

def load_dataframe(filepath):
    """
    Load DataFrame (auto-detect format from extension)
    
    Args:
        filepath: Path to data file
    
    Returns:
        pandas DataFrame
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix in ['.pkl', '.pickle']:
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unknown format: {filepath.suffix}")

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test config loading
    try:
        config = load_config()
        print(f" Config loaded: {config['project']['name']}")
    except FileNotFoundError:
        print("  config/config.yaml not found (create it first)")
    
    # Test haversine
    # Cairo Tahrir Square to Nasr City (approx)
    dist = haversine_distance(31.2357, 30.0444, 31.3387, 30.0626)
    print(f" Haversine distance: {dist:.2f} km")
    
    # Test speed calculation
    speed = calculate_speed(31.2357, 30.0444, 31.3387, 30.0626, 600)  # 10 minutes
    print(f" Calculated speed: {speed:.2f} km/h")
    
    print("\n All utilities working!")