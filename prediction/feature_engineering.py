"""
Feature engineering for ETA prediction

Extracts features from GPS trajectories for LightGBM model

Author: Group 2 - ML Team
Environment: dnervenv
Target: MAE ≤ 3.0 minutes
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import pickle

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config, haversine_distance

def extract_trip_features(df):
    """
    Extract features from each trip
    
    Features:
    - Distance (km)
    - Number of GPS points
    - Time of day (hour)
    - Day of week
    - Start/end coordinates
    - Average speed
    - Actual duration (target variable)
    """
    print("\n Extracting features from trips...")
    
    features_list = []
    
    trip_ids = df['trip_id'].unique()
    
    for trip_id in trip_ids:
        trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
        
        if len(trip_data) < 2:
            continue
        
        # Basic info
        start_time = trip_data.iloc[0]['timestamp']
        end_time = trip_data.iloc[-1]['timestamp']
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Skip very short or very long trips
        if duration_minutes < 1 or duration_minutes > 300:  # 1 min to 5 hours
            continue
        
        # Coordinates
        start_lon = trip_data.iloc[0]['longitude']
        start_lat = trip_data.iloc[0]['latitude']
        end_lon = trip_data.iloc[-1]['longitude']
        end_lat = trip_data.iloc[-1]['latitude']
        
        # Calculate straight-line distance
        distance_km = haversine_distance(start_lon, start_lat, end_lon, end_lat)
        
        # Time features
        hour = start_time.hour
        day_of_week = start_time.dayofweek  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        
        # Trip characteristics
        num_points = len(trip_data)
        
        # Average speed (if duration > 0)
        avg_speed_kph = (distance_km / duration_minutes * 60) if duration_minutes > 0 else 0
        
        features = {
            'trip_id': trip_id,
            'distance_km': distance_km,
            'num_points': num_points,
            'start_lon': start_lon,
            'start_lat': start_lat,
            'end_lon': end_lon,
            'end_lat': end_lat,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_rush_hour': is_rush_hour,
            'avg_speed_kph': avg_speed_kph,
            'duration_minutes': duration_minutes  # TARGET variable
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    print(f" Extracted features from {len(features_df)} trips")
    print(f"\nFeature columns: {list(features_df.columns)}")
    
    return features_df

def add_route_features(features_df, routes_results_path):
    """
    Add route-based features from DBSCAN results
    
    Features:
    - Is trip part of discovered route?
    - Route ID
    - Route popularity (number of trips in route)
    """
    print("\n  Adding route features...")
    
    # Load DBSCAN results
    with open(routes_results_path, 'rb') as f:
        results = pickle.load(f)
    
    routes = results['routes']
    trip_ids = results['trip_ids']
    labels = results['labels']
    
    # Create trip_id to route mapping
    trip_to_route = {}
    for i, trip_id in enumerate(trip_ids):
        label = labels[i]
        if label != -1:  # Not noise
            trip_to_route[trip_id] = f"route_{label:03d}"
    
    # Add route features
    features_df['route_id'] = features_df['trip_id'].map(trip_to_route).fillna('no_route')
    features_df['is_on_route'] = (features_df['route_id'] != 'no_route').astype(int)
    
    # Add route popularity
    route_popularity = {}
    for route_id, route_data in routes.items():
        route_popularity[route_id] = route_data['num_trips']
    
    features_df['route_popularity'] = features_df['route_id'].map(route_popularity).fillna(0).astype(int)
    
    print(f" Added route features")
    print(f"   Trips on routes: {features_df['is_on_route'].sum()} ({features_df['is_on_route'].sum()/len(features_df)*100:.1f}%)")
    
    return features_df

def split_train_test(features_df, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = features_df.drop(['trip_id', 'duration_minutes', 'route_id'], axis=1)
    y = features_df['duration_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\n Data Split:")
    print(f"   Training set: {len(X_train)} trips")
    print(f"   Test set: {len(X_test)} trips")
    print(f"   Features: {X.columns.tolist()}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main feature engineering pipeline"""
    print("="*60)
    print("FEATURE ENGINEERING FOR ETA PREDICTION")
    print("="*60)
    
    config = load_config()
    
    # Load trip data
    trips_file = Path(config['data']['processed_dir']) / "tdrive_100taxis_trips.parquet"
    print(f"\n Loading trips from {trips_file}...")
    df = pd.read_parquet(trips_file)
    print(f" Loaded {df['trip_id'].nunique()} trips")
    
    # Extract features
    features_df = extract_trip_features(df)
    
    # Add route features
    routes_results = Path(config['outputs']['route_discovery_dir']) / "route_discovery_results.pkl"
    if routes_results.exists():
        features_df = add_route_features(features_df, routes_results)
    else:
        print("  DBSCAN results not found, skipping route features")
    
    # Save features
    output_file = Path(config['data']['final_dir']) / "trip_features.parquet"
    features_df.to_parquet(output_file, compression='snappy', index=False)
    print(f"\n Features saved to {output_file}")
    
    # Display statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(features_df[['distance_km', 'num_points', 'avg_speed_kph', 'duration_minutes']].describe())
    
    print("\n FEATURE ENGINEERING COMPLETE\n")
    
    return features_df

if __name__ == "__main__":
    main()