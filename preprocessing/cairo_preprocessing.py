"""
Cairo GPS Data Preprocessing

Adapted from Beijing preprocessing for Cairo microbus data.
- Geographic bounds for Cairo
- Speed filtering
- Data format conversion

Author: D-Nerve Team
Environment: dnervenv
"""

import pandas as pd
import numpy as np
import os
import sys
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

# Cairo bounding box (Greater Cairo Region)
CAIRO_BOUNDS = {
    'min_lat': 29.75,   # South (Helwan area)
    'max_lat': 30.20,   # North (Shubra El Kheima)
    'min_lon': 30.85,   # West (6th October)
    'max_lon': 31.55    # East (New Cairo)
}

MAX_SPEED_KPH = 100  # Maximum realistic speed for Cairo traffic

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS points (km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    km = 6371 * c
    return km

def load_cairo_data(input_file):
    """Load Cairo trajectory data from CSV"""
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Total GPS points: {len(df):,}")
    print(f"  Unique trips: {df['trip_id'].nunique()}")
    print(f"  Unique routes: {df['route_id'].nunique()}")
    
    return df

def remove_invalid_coordinates(df):
    """Remove points at (0,0) or with invalid coordinates"""
    original_count = len(df)
    
    df_clean = df[~((df['latitude'] == 0) & (df['longitude'] == 0))].copy()
    
    removed = original_count - len(df_clean)
    if removed > 0:
        print(f"  Invalid (0,0) points removed: {removed:,}")
    
    return df_clean

def remove_geographic_outliers(df, bounds):
    """Remove points outside Cairo bounds"""
    original_count = len(df)
    
    df_clean = df[
        (df['latitude'] >= bounds['min_lat']) &
        (df['latitude'] <= bounds['max_lat']) &
        (df['longitude'] >= bounds['min_lon']) &
        (df['longitude'] <= bounds['max_lon'])
    ].copy()
    
    removed = original_count - len(df_clean)
    if removed > 0:
        print(f"  Geographic outliers removed: {removed:,}")
    
    return df_clean

def remove_speed_outliers(df, max_speed):
    """Remove points with impossible speeds"""
    print("\n  Calculating speeds between consecutive points...")
    
    df = df.sort_values(['trip_id', 'timestamp']).reset_index(drop=True)
    
    valid_indices = []
    unique_trips = df['trip_id'].unique()
    
    for trip_id in unique_trips:
        trip_df = df[df['trip_id'] == trip_id].copy()
        indices = trip_df.index.tolist()
        
        if len(indices) < 2:
            valid_indices.extend(indices)
            continue
        
        # First point always valid
        valid_indices.append(indices[0])
        
        for i in range(1, len(indices)):
            prev_idx = indices[i-1]
            curr_idx = indices[i]
            
            dist_km = haversine_distance(
                df.loc[prev_idx, 'longitude'],
                df.loc[prev_idx, 'latitude'],
                df.loc[curr_idx, 'longitude'],
                df.loc[curr_idx, 'latitude']
            )
            
            time_diff_hours = (
                df.loc[curr_idx, 'timestamp'] - df.loc[prev_idx, 'timestamp']
            ).total_seconds() / 3600
            
            if time_diff_hours > 0:
                speed_kph = dist_km / time_diff_hours
                if speed_kph <= max_speed:
                    valid_indices.append(curr_idx)
            else:
                valid_indices.append(curr_idx)
    
    df_clean = df.loc[valid_indices].reset_index(drop=True)
    
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"  Speed outliers removed: {removed:,}")
    
    return df_clean

def add_derived_features(df):
    """Add useful features for ML"""
    print("\n  Adding derived features...")
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_peak'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Calculate trip statistics per trip
    trip_stats = df.groupby('trip_id').agg({
        'latitude': ['first', 'last', 'count'],
        'longitude': ['first', 'last'],
        'time_offset_sec': 'max'
    }).reset_index()
    
    trip_stats.columns = ['trip_id', 'start_lat', 'end_lat', 'num_points', 
                          'start_lon', 'end_lon', 'duration_sec']
    
    # Calculate trip distance (approximate)
    trip_stats['trip_distance_km'] = trip_stats.apply(
        lambda row: haversine_distance(
            row['start_lon'], row['start_lat'],
            row['end_lon'], row['end_lat']
        ), axis=1
    )
    
    print(f"    Trip statistics computed for {len(trip_stats)} trips")
    
    return df, trip_stats

def preprocess_cairo_data(input_file, output_dir):
    """Complete Cairo preprocessing pipeline"""
    print("="*60)
    print("CAIRO DATA PREPROCESSING")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_cairo_data(input_file)
    original_count = len(df)
    
    # Step 1: Remove invalid coordinates
    print("\nStep 1: Removing invalid coordinates...")
    df = remove_invalid_coordinates(df)
    
    # Step 2: Geographic filtering
    print("\nStep 2: Geographic filtering (Cairo bounds)...")
    df = remove_geographic_outliers(df, CAIRO_BOUNDS)
    
    # Step 3: Speed filtering
    print(f"\nStep 3: Speed filtering (max {MAX_SPEED_KPH} km/h)...")
    df = remove_speed_outliers(df, MAX_SPEED_KPH)
    
    # Step 4: Add derived features
    print("\nStep 4: Adding derived features...")
    df, trip_stats = add_derived_features(df)
    
    # Save cleaned data
    clean_file = output_dir / "cairo_trajectories_clean.csv"
    df.to_csv(clean_file, index=False)
    print(f"\n✓ Saved cleaned trajectories: {clean_file}")
    
    stats_file = output_dir / "cairo_trip_stats.csv"
    trip_stats.to_csv(stats_file, index=False)
    print(f"✓ Saved trip statistics: {stats_file}")
    
    # Summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Original points: {original_count:,}")
    print(f"Clean points: {len(df):,} ({len(df)/original_count*100:.1f}% retained)")
    print(f"Trips: {df['trip_id'].nunique()}")
    print(f"Routes (ground truth): {df['route_id'].nunique()}")
    
    print(f"\nNext step:")
    print(f"  python clustering/cairo_dbscan_routes.py")
    
    return df, trip_stats

if __name__ == "__main__":
    INPUT_FILE = "data/cairo/raw/cairo_trajectories_full.csv"
    OUTPUT_DIR = "data/cairo/processed"
    
    df, stats = preprocess_cairo_data(INPUT_FILE, OUTPUT_DIR)
