"""
Remove GPS outliers based on:
1. Geographic bounds (outside Beijing or at 0,0)
2. Impossible speeds (>120 km/h)
3. Duplicate timestamps

Author: Group 2 - ML Team
Environment: dnervenv
Platform: Windows
"""

import pandas as pd
import numpy as np
import os
import sys
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

# Fix imports on Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config, save_dataframe

# Beijing bounding box (conservative)
BEIJING_BOUNDS = {
    'min_lat': 39.4,
    'max_lat': 40.5,
    'min_lon': 116.0,
    'max_lon': 117.0
}

MAX_SPEED_KPH = 120  # Maximum realistic speed

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate distance between two GPS points (km)
    Uses Haversine formula
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    km = 6371 * c  # Earth radius in km
    return km

def remove_invalid_coordinates(df):
    """Remove points at (0,0) or with obviously invalid coordinates"""
    original_count = len(df)
    
    # Remove points at exactly (0,0) - GPS errors
    df_clean = df[~((df['latitude'] == 0) & (df['longitude'] == 0))].copy()
    
    removed = original_count - len(df_clean)
    print(f"Invalid (0,0) points removed: {removed:,} ({removed/original_count*100:.2f}%)")
    
    return df_clean

def remove_geographic_outliers(df, bounds):
    """Remove points outside city bounds"""
    original_count = len(df)
    
    df_clean = df[
        (df['latitude'] >= bounds['min_lat']) &
        (df['latitude'] <= bounds['max_lat']) &
        (df['longitude'] >= bounds['min_lon']) &
        (df['longitude'] <= bounds['max_lon'])
    ].copy()
    
    removed = original_count - len(df_clean)
    print(f"Geographic outliers removed: {removed:,} ({removed/original_count*100:.2f}%)")
    
    return df_clean

def remove_speed_outliers(df, max_speed):
    """Remove points with impossible calculated speeds"""
    print("\nCalculating speeds between consecutive points...")
    print("(This may take 1-2 minutes)")
    
    df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
    
    valid_indices = []
    
    unique_taxis = df['taxi_id'].unique()
    total_taxis = len(unique_taxis)
    
    for idx, taxi_id in enumerate(unique_taxis, 1):
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total_taxis} taxis")
        
        taxi_df = df[df['taxi_id'] == taxi_id].copy()
        indices = taxi_df.index.tolist()
        
        if len(indices) < 2:
            valid_indices.extend(indices)
            continue
        
        # First point is always valid
        valid_indices.append(indices[0])
        
        for i in range(1, len(indices)):
            prev_idx = indices[i-1]
            curr_idx = indices[i]
            
            # Calculate distance
            dist_km = haversine_distance(
                df.loc[prev_idx, 'longitude'],
                df.loc[prev_idx, 'latitude'],
                df.loc[curr_idx, 'longitude'],
                df.loc[curr_idx, 'latitude']
            )
            
            # Calculate time difference
            time_diff_hours = (
                df.loc[curr_idx, 'timestamp'] - df.loc[prev_idx, 'timestamp']
            ).total_seconds() / 3600
            
            # Calculate speed
            if time_diff_hours > 0:
                speed_kph = dist_km / time_diff_hours
                
                if speed_kph <= max_speed:
                    valid_indices.append(curr_idx)
    
    df_clean = df.loc[valid_indices].reset_index(drop=True)
    
    removed = len(df) - len(df_clean)
    print(f"Speed outliers removed: {removed:,} ({removed/len(df)*100:.2f}%)")
    
    return df_clean

def remove_duplicates(df):
    """Remove duplicate GPS points (same taxi, same timestamp)"""
    original_count = len(df)
    
    df_clean = df.drop_duplicates(subset=['taxi_id', 'timestamp'], keep='first')
    
    removed = original_count - len(df_clean)
    print(f"Duplicate timestamps removed: {removed:,} ({removed/original_count*100:.2f}%)")
    
    return df_clean

def clean_dataset(input_file, output_file):
    """Complete cleaning pipeline"""
    print("="*60)
    print("PREPROCESSING: OUTLIER REMOVAL")
    print("="*60)
    
    # Load data
    print(f"\n Loading {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Original points: {len(df):,}")
    
    # Step 1: Remove invalid (0,0) coordinates
    print("\n Step 1: Removing invalid coordinates...")
    df = remove_invalid_coordinates(df)
    
    # Step 2: Geographic filtering
    print("\n  Step 2: Geographic filtering (Beijing bounds)...")
    df = remove_geographic_outliers(df, BEIJING_BOUNDS)
    
    # Step 3: Remove duplicates
    print("\n Step 3: Removing duplicate timestamps...")
    df = remove_duplicates(df)
    
    # Step 4: Speed filtering
    print(f"\n Step 4: Speed filtering (max {MAX_SPEED_KPH} km/h)...")
    df = remove_speed_outliers(df, MAX_SPEED_KPH)
    
    # Save cleaned data
    print(f"\n Saving cleaned data to {output_file}...")
    save_dataframe(df, output_file, format='parquet')
    
    # Final summary
    print("\n" + "="*60)
    print(" CLEANING COMPLETE")
    print("="*60)
    print(f"Final points: {len(df):,}")
    print(f"Vehicles remaining: {df['taxi_id'].nunique()}")
    print(f"\nNext step:")
    print(f"  python preprocessing\\02_segment_trips.py")
    
    return df

if __name__ == "__main__":
    config = load_config()
    
    INPUT = Path(config['data']['processed_dir']) / "tdrive_100taxis.parquet"
    OUTPUT = Path(config['data']['processed_dir']) / "tdrive_100taxis_clean.parquet"
    
    df_clean = clean_dataset(INPUT, OUTPUT)