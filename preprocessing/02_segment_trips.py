"""
Segment continuous GPS streams into individual trips

Trip boundaries defined by:
- Time gap > 10 minutes (likely taxi turned off GPS)
- Minimum 10 points per trip

Author: Group 2 - ML Team
Environment: dnervenv
Platform: Windows
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Fix imports on Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config, save_dataframe

# Segmentation parameters
MAX_TIME_GAP_SECONDS = 600  # 10 minutes
MIN_POINTS_PER_TRIP = 10     # Minimum points to be valid trip

def segment_into_trips(df):
    """
    Segment GPS stream into trips
    
    Returns DataFrame with additional 'trip_id' column
    """
    print("="*60)
    print("PREPROCESSING: TRIP SEGMENTATION")
    print("="*60)
    
    df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
    df['trip_id'] = None
    
    trip_counter = 0
    total_taxis = df['taxi_id'].nunique()
    
    print(f"\nProcessing {total_taxis} taxis...")
    print("(This will take 1-2 minutes)\n")
    
    for taxi_num, taxi_id in enumerate(df['taxi_id'].unique(), 1):
        if taxi_num % 10 == 0:
            print(f"  Progress: {taxi_num}/{total_taxis} taxis")
        
        taxi_df = df[df['taxi_id'] == taxi_id].copy()
        indices = taxi_df.index.tolist()
        
        current_trip_indices = [indices[0]]
        
        for i in range(1, len(indices)):
            prev_idx = indices[i-1]
            curr_idx = indices[i]
            
            # Calculate time gap
            time_gap = (
                df.loc[curr_idx, 'timestamp'] - df.loc[prev_idx, 'timestamp']
            ).total_seconds()
            
            # Check if trip should end (large time gap)
            if time_gap > MAX_TIME_GAP_SECONDS:
                # Save current trip if long enough
                if len(current_trip_indices) >= MIN_POINTS_PER_TRIP:
                    df.loc[current_trip_indices, 'trip_id'] = f"trip_{trip_counter:06d}"
                    trip_counter += 1
                
                # Start new trip
                current_trip_indices = [curr_idx]
            else:
                # Continue current trip
                current_trip_indices.append(curr_idx)
        
        # Save last trip
        if len(current_trip_indices) >= MIN_POINTS_PER_TRIP:
            df.loc[current_trip_indices, 'trip_id'] = f"trip_{trip_counter:06d}"
            trip_counter += 1
    
    # Remove points not in any trip
    df_trips = df[df['trip_id'].notna()].copy()
    
    # Calculate trip statistics
    print("\n" + "="*60)
    print("TRIP SEGMENTATION RESULTS")
    print("="*60)
    print(f"Original GPS points: {len(df):,}")
    print(f"Points in valid trips: {len(df_trips):,}")
    print(f"Points discarded: {len(df) - len(df_trips):,} ({(len(df)-len(df_trips))/len(df)*100:.1f}%)")
    print(f"\nTotal trips created: {df_trips['trip_id'].nunique():,}")
    print(f"Avg points per trip: {len(df_trips) / df_trips['trip_id'].nunique():.1f}")
    print(f"Avg trips per vehicle: {df_trips['trip_id'].nunique() / df_trips['taxi_id'].nunique():.1f}")
    
    # Calculate trip durations
    trip_stats = df_trips.groupby('trip_id').agg({
        'timestamp': ['min', 'max'],
        'taxi_id': 'first'
    }).reset_index()
    
    trip_stats['duration_minutes'] = (
        (trip_stats[('timestamp', 'max')] - trip_stats[('timestamp', 'min')])
        .dt.total_seconds() / 60
    )
    
    print(f"\nTrip Duration Statistics:")
    print(f"  Min: {trip_stats['duration_minutes'].min():.1f} minutes")
    print(f"  Median: {trip_stats['duration_minutes'].median():.1f} minutes")
    print(f"  Mean: {trip_stats['duration_minutes'].mean():.1f} minutes")
    print(f"  Max: {trip_stats['duration_minutes'].max():.1f} minutes")
    
    return df_trips

if __name__ == "__main__":
    config = load_config()
    
    INPUT = Path(config['data']['processed_dir']) / "tdrive_100taxis_clean.parquet"
    OUTPUT = Path(config['data']['processed_dir']) / "tdrive_100taxis_trips.parquet"
    
    # Load cleaned data
    print(f" Loading {INPUT}...")
    df = pd.read_parquet(INPUT)
    
    # Segment into trips
    df_trips = segment_into_trips(df)
    
    # Save
    print(f"\n Saving to {OUTPUT}...")
    save_dataframe(df_trips, OUTPUT, format='parquet')
    
    print("\n" + "="*60)
    print(" TRIP SEGMENTATION COMPLETE")
    print("="*60)
    print(f"\nNext step:")
    print(f"  python clustering\\dbscan_routes.py")