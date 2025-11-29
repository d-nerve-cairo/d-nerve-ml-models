"""
Load T-Drive dataset into pandas DataFrame

Usage:
    python data_loading/load_tdrive.py

Author: Group 2 - ML Team
Environment: dnervenv
Platform: Windows
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.utils import load_config, save_dataframe, create_output_dir


def load_single_taxi(filepath):
    """
    Load one taxi's GPS data from text file
    
    Args:
        filepath: Path to taxi log file (e.g., "1.txt")
    
    Returns:
        pandas DataFrame or None if error
    """
    try:
        df = pd.read_csv(
            filepath, 
            header=None,
            names=['taxi_id', 'timestamp', 'longitude', 'latitude'],
            parse_dates=['timestamp']
        )
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_multiple_taxis(data_dir, num_taxis=100, start_id=1):
    """
    Load first N taxis for development
    
    Args:
        data_dir: Path to taxi_log_2008_by_id folder
        num_taxis: How many taxis to load (default 100)
        start_id: Starting taxi ID (default 1)
    
    Returns:
        Combined DataFrame with all GPS points
    """
    data_frames = []
    
    print(f"Loading {num_taxis} taxis from {data_dir}...")
    
    for taxi_id in tqdm(range(start_id, start_id + num_taxis), desc="Loading taxis"):
        filepath = os.path.join(data_dir, f"{taxi_id}.txt")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping")
            continue
            
        df = load_single_taxi(filepath)
        if df is not None and len(df) > 0:
            data_frames.append(df)
    
    # Combine all taxis
    if not data_frames:
        raise ValueError("No data loaded! Check data directory path.")
    
    combined = pd.concat(data_frames, ignore_index=True)
    
    print(f"\n Loaded {len(data_frames)} taxis")
    print(f" Total GPS points: {len(combined):,}")
    print(f" Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print(f" Unique taxis: {combined['taxi_id'].nunique()}")
    
    return combined

def get_data_summary(df):
    """Print comprehensive dataset statistics"""
    print("\n" + "="*60)
    print("T-DRIVE DATASET SUMMARY")
    print("="*60)
    
    # Basic stats
    print(f"\n Basic Statistics:")
    print(f"  Total GPS Points: {len(df):,}")
    print(f"  Unique Vehicles: {df['taxi_id'].nunique()}")
    print(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Spatial bounds
    print(f"\n Spatial Coverage (Beijing):")
    print(f"  Latitude:  {df['latitude'].min():.4f}° to {df['latitude'].max():.4f}°")
    print(f"  Longitude: {df['longitude'].min():.4f}° to {df['longitude'].max():.4f}°")
    
    # Sampling rate
    df_sorted = df.sort_values(['taxi_id', 'timestamp']).copy()
    df_sorted['time_diff'] = df_sorted.groupby('taxi_id')['timestamp'].diff()
    median_interval = df_sorted['time_diff'].dt.total_seconds().median()
    
    print(f"\n  Sampling Characteristics:")
    print(f"  Median interval: {median_interval:.0f} seconds ({median_interval/60:.1f} minutes)")
    print(f"  Avg points per vehicle: {len(df) / df['taxi_id'].nunique():.0f}")
    
    # Missing data
    print(f"\n Data Quality:")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate timestamps: {df.duplicated(subset=['taxi_id', 'timestamp']).sum()}")
    
    # Points per taxi distribution
    points_per_taxi = df.groupby('taxi_id').size()
    print(f"\n Points per Taxi:")
    print(f"  Min: {points_per_taxi.min()}")
    print(f"  Median: {points_per_taxi.median():.0f}")
    print(f"  Max: {points_per_taxi.max()}")
    
    return df_sorted

def main():
    """Main execution function"""
    print("="*60)
    print("T-DRIVE DATA LOADING")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Get paths (convert to Windows Path)
    tdrive_path = Path(config['data']['tdrive_path'])
    num_taxis = config['data']['num_taxis_sample']
    output_dir = Path(config['data']['processed_dir'])
    
    # Create output directory
    create_output_dir(output_dir)
    
    # Check if T-Drive path exists
    if not tdrive_path.exists():
        print(f"\n ERROR: T-Drive path not found!")
        print(f"   Expected: {tdrive_path}")
        print(f"\n   Please download T-Drive dataset:")
        print(f"   1. Visit: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/")
        print(f"   2. Download and extract to: {tdrive_path.parent}")
        print(f"   3. Update path in config/config.yaml if needed")
        print(f"   4. Re-run this script")
        return
    
    # Load data
    df = load_multiple_taxis(str(tdrive_path), num_taxis=num_taxis)
    
    # Get statistics
    df_with_stats = get_data_summary(df)
    
    # Save processed version
    output_file = output_dir / f"tdrive_{num_taxis}taxis.parquet"
    save_dataframe(df_with_stats, output_file, format='parquet')
    
    print("\n" + "="*60)
    print(" DATA LOADING COMPLETE")
    print("="*60)
    print(f"Output: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Run: python preprocessing\\01_remove_outliers.py")
    print(f"  2. Run: python preprocessing\\02_segment_trips.py")
    print(f"  3. Run: python clustering\\dbscan_routes.py")

if __name__ == "__main__":
    main()