"""
Beijing T-Drive Dataset: Download and Preprocessing Pipeline
Real-world taxi GPS trajectories for route discovery validation

Dataset: Microsoft Research T-Drive
- 10,357 taxis in Beijing
- 1 week of data (Feb 2-8, 2008)
- ~15 million GPS points
- Format: taxi_id, timestamp, longitude, latitude

Author: D-Nerve Team
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directories
DATA_DIR = Path("data/beijing_tdrive")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Beijing geographic bounds
BEIJING_BOUNDS = {
    'lat_min': 39.4,
    'lat_max': 41.1,
    'lon_min': 115.4,
    'lon_max': 117.5
}

# Trip segmentation parameters
MAX_TIME_GAP_MINUTES = 20  # Max gap before starting new trip
MIN_TRIP_POINTS = 10       # Minimum GPS points per trip
MIN_TRIP_DURATION_MIN = 3  # Minimum trip duration
MAX_TRIP_DURATION_MIN = 120  # Maximum trip duration (filter outliers)
MAX_SPEED_KPH = 120        # Maximum realistic speed

# Sampling for manageable computation
MAX_TAXIS = 500            # Limit taxis for faster processing
MAX_TRIPS = 1000           # Limit trips for clustering

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_sample_data():
    """
    Download a sample of T-Drive data
    
    Note: The full dataset requires manual download from Microsoft Research
    https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
    
    This function provides instructions and downloads what's available.
    """
    print("="*70)
    print("T-DRIVE DATA DOWNLOAD")
    print("="*70)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    existing_files = list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("*.csv"))
    if len(existing_files) > 0:
        print(f"✓ Found {len(existing_files)} existing data files in {RAW_DIR}")
        return True
    
    # Check for zip files
    zip_files = list(RAW_DIR.glob("*.zip"))
    if zip_files:
        print(f"Found {len(zip_files)} zip files. Extracting...")
        for zf in zip_files:
            extract_zip(zf)
        return True
    
    print("\n" + "!"*70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("!"*70)
    print("""
The T-Drive dataset must be downloaded manually from Microsoft Research.

OPTION 1: Microsoft Research (Original Source)
   1. Go to: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
   2. Click "Download Dataset"
   3. Download all zip files (1.zip through 14.zip)
   4. Place them in: {raw_dir}
   5. Re-run this script

OPTION 2: Kaggle (Alternative)
   1. Go to: https://www.kaggle.com/datasets/arashnic/tdriver
   2. Download the dataset
   3. Extract to: {raw_dir}
   4. Re-run this script

After downloading, the folder should contain files like:
   - 1.txt, 2.txt, ... (individual taxi trajectories)
   - Or folders with numbered files

Press Enter after downloading to continue, or Ctrl+C to exit...
""".format(raw_dir=RAW_DIR.absolute()))
    
    try:
        input()
        # Re-check for files
        existing_files = list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("**/*.txt"))
        if existing_files:
            print(f"✓ Found {len(existing_files)} files!")
            return True
        else:
            print("No data files found. Please download the dataset first.")
            return False
    except KeyboardInterrupt:
        print("\nExiting...")
        return False


def extract_zip(zip_path):
    """Extract a zip file"""
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DIR)


def create_synthetic_sample():
    """
    Create a small synthetic sample for testing the pipeline
    (Use this if you can't download the real data)
    """
    print("\nCreating synthetic sample for pipeline testing...")
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Beijing center coordinates
    center_lat, center_lon = 39.9042, 116.4074
    
    # Generate 50 synthetic taxi trajectories
    for taxi_id in range(1, 51):
        trajectory = []
        
        # Generate 3-5 trips per taxi
        num_trips = np.random.randint(3, 6)
        base_time = datetime(2008, 2, 2, 6, 0, 0)
        
        for trip in range(num_trips):
            # Random start position
            lat = center_lat + np.random.uniform(-0.1, 0.1)
            lon = center_lon + np.random.uniform(-0.1, 0.1)
            
            # Trip duration: 10-60 minutes
            trip_duration = np.random.randint(10, 60)
            num_points = trip_duration * 2  # ~30 sec sampling
            
            for i in range(num_points):
                # Add movement
                lat += np.random.uniform(-0.002, 0.002)
                lon += np.random.uniform(-0.002, 0.002)
                
                timestamp = base_time + timedelta(seconds=i*30)
                trajectory.append(f"{taxi_id},{timestamp.strftime('%Y-%m-%d %H:%M:%S')},{lon:.5f},{lat:.5f}")
            
            # Gap between trips
            base_time += timedelta(minutes=trip_duration + np.random.randint(30, 120))
        
        # Save taxi file
        with open(RAW_DIR / f"{taxi_id}.txt", 'w') as f:
            f.write('\n'.join(trajectory))
    
    print(f"✓ Created 50 synthetic taxi files in {RAW_DIR}")
    return True

# ============================================================================
# DATA LOADING
# ============================================================================

def load_raw_data(max_taxis=MAX_TAXIS):
    """
    Load raw T-Drive data files
    
    Format: taxi_id, timestamp, longitude, latitude
    """
    print("\n" + "="*70)
    print("LOADING RAW T-DRIVE DATA")
    print("="*70)
    
    # Find all trajectory files
    txt_files = list(RAW_DIR.glob("*.txt"))
    
    # Also check subdirectories
    for subdir in RAW_DIR.iterdir():
        if subdir.is_dir():
            txt_files.extend(subdir.glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found!")
        return None
    
    print(f"Found {len(txt_files)} taxi trajectory files")
    
    # Limit for faster processing
    if len(txt_files) > max_taxis:
        print(f"Sampling {max_taxis} taxis for faster processing...")
        txt_files = np.random.choice(txt_files, max_taxis, replace=False)
    
    all_data = []
    
    for txt_file in tqdm(txt_files, desc="Loading taxis"):
        try:
            # Try to extract taxi ID from filename
            taxi_id = txt_file.stem
            
            # Read file
            df = pd.read_csv(
                txt_file, 
                header=None,
                names=['taxi_id', 'timestamp', 'longitude', 'latitude'],
                parse_dates=['timestamp']
            )
            
            # Ensure taxi_id is set
            if df['taxi_id'].isna().all():
                df['taxi_id'] = taxi_id
            
            all_data.append(df)
            
        except Exception as e:
            # Try alternative format (some files may have different structure)
            try:
                df = pd.read_csv(txt_file, header=None)
                if len(df.columns) == 4:
                    df.columns = ['taxi_id', 'timestamp', 'longitude', 'latitude']
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    all_data.append(df)
            except:
                continue
    
    if not all_data:
        print("Failed to load any data!")
        return None
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✓ Loaded {len(df):,} GPS points from {len(all_data)} taxis")
    
    return df

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Clean and preprocess GPS data
    """
    print("\n" + "="*70)
    print("PREPROCESSING GPS DATA")
    print("="*70)
    
    initial_count = len(df)
    print(f"Initial records: {initial_count:,}")
    
    # 1. Remove invalid coordinates
    df = df[
        (df['latitude'] >= BEIJING_BOUNDS['lat_min']) &
        (df['latitude'] <= BEIJING_BOUNDS['lat_max']) &
        (df['longitude'] >= BEIJING_BOUNDS['lon_min']) &
        (df['longitude'] <= BEIJING_BOUNDS['lon_max'])
    ]
    print(f"After geographic filter: {len(df):,} ({len(df)/initial_count*100:.1f}%)")
    
    # 2. Remove duplicates
    df = df.drop_duplicates(subset=['taxi_id', 'timestamp'])
    print(f"After duplicate removal: {len(df):,}")
    
    # 3. Sort by taxi and time
    df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
    
    # 4. Calculate speed between consecutive points
    df['prev_lat'] = df.groupby('taxi_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('taxi_id')['longitude'].shift(1)
    df['prev_time'] = df.groupby('taxi_id')['timestamp'].shift(1)
    
    # Distance in km (Haversine approximation)
    df['distance_km'] = haversine_vectorized(
        df['prev_lon'], df['prev_lat'],
        df['longitude'], df['latitude']
    )
    
    # Time difference in hours
    df['time_diff_h'] = (df['timestamp'] - df['prev_time']).dt.total_seconds() / 3600
    
    # Speed in km/h
    df['speed_kph'] = df['distance_km'] / df['time_diff_h'].replace(0, np.nan)
    
    # 5. Filter unrealistic speeds
    speed_mask = (df['speed_kph'].isna()) | (df['speed_kph'] <= MAX_SPEED_KPH)
    df = df[speed_mask]
    print(f"After speed filter: {len(df):,}")
    
    # Clean up temporary columns
    df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_time', 'distance_km', 'time_diff_h', 'speed_kph'])
    
    print(f"\n✓ Preprocessing complete: {len(df):,} points retained")
    
    return df


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance calculation"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# ============================================================================
# TRIP SEGMENTATION
# ============================================================================

def segment_trips(df, max_gap_minutes=MAX_TIME_GAP_MINUTES):
    """
    Segment continuous GPS streams into individual trips
    
    A new trip starts when:
    - Time gap > max_gap_minutes (taxi was stopped/waiting)
    """
    print("\n" + "="*70)
    print("SEGMENTING INTO TRIPS")
    print("="*70)
    
    df = df.sort_values(['taxi_id', 'timestamp']).reset_index(drop=True)
    
    # Calculate time gaps
    df['time_gap'] = df.groupby('taxi_id')['timestamp'].diff()
    df['time_gap_min'] = df['time_gap'].dt.total_seconds() / 60
    
    # Mark trip boundaries (new trip when gap > threshold)
    df['new_trip'] = (df['time_gap_min'] > max_gap_minutes) | (df['time_gap_min'].isna())
    
    # Assign trip IDs
    df['trip_id'] = df.groupby('taxi_id')['new_trip'].cumsum()
    df['trip_id'] = df['taxi_id'].astype(str) + '_' + df['trip_id'].astype(str)
    
    # Clean up
    df = df.drop(columns=['time_gap', 'time_gap_min', 'new_trip'])
    
    # Count trips
    trip_counts = df.groupby('trip_id').size()
    print(f"Total trips identified: {len(trip_counts):,}")
    
    # Filter trips by point count
    valid_trips = trip_counts[trip_counts >= MIN_TRIP_POINTS].index
    df = df[df['trip_id'].isin(valid_trips)]
    print(f"Trips with ≥{MIN_TRIP_POINTS} points: {len(valid_trips):,}")
    
    # Calculate trip statistics
    trip_stats = df.groupby('trip_id').agg({
        'timestamp': ['min', 'max'],
        'latitude': 'count'
    })
    trip_stats.columns = ['start_time', 'end_time', 'num_points']
    trip_stats['duration_min'] = (trip_stats['end_time'] - trip_stats['start_time']).dt.total_seconds() / 60
    
    # Filter by duration
    valid_duration = trip_stats[
        (trip_stats['duration_min'] >= MIN_TRIP_DURATION_MIN) &
        (trip_stats['duration_min'] <= MAX_TRIP_DURATION_MIN)
    ].index
    
    df = df[df['trip_id'].isin(valid_duration)]
    print(f"Trips with valid duration: {len(valid_duration):,}")
    
    print(f"\n✓ Final trips: {df['trip_id'].nunique():,}")
    
    return df

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================

def save_processed_data(df, max_trips=MAX_TRIPS):
    """Save processed data for clustering"""
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample trips if needed
    unique_trips = df['trip_id'].unique()
    if len(unique_trips) > max_trips:
        print(f"\nSampling {max_trips} trips for clustering...")
        selected_trips = np.random.choice(unique_trips, max_trips, replace=False)
        df = df[df['trip_id'].isin(selected_trips)]
    
    # Save
    output_file = PROCESSED_DIR / "beijing_trips.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to {output_file}")
    print(f"  Trips: {df['trip_id'].nunique():,}")
    print(f"  GPS points: {len(df):,}")
    
    # Save trip summary
    trip_summary = df.groupby('trip_id').agg({
        'taxi_id': 'first',
        'timestamp': ['min', 'max', 'count'],
        'latitude': ['first', 'last', 'mean'],
        'longitude': ['first', 'last', 'mean']
    })
    trip_summary.columns = ['taxi_id', 'start_time', 'end_time', 'num_points',
                           'start_lat', 'end_lat', 'mean_lat',
                           'start_lon', 'end_lon', 'mean_lon']
    trip_summary['duration_min'] = (trip_summary['end_time'] - trip_summary['start_time']).dt.total_seconds() / 60
    
    summary_file = PROCESSED_DIR / "trip_summary.csv"
    trip_summary.to_csv(summary_file)
    print(f"✓ Saved trip summary to {summary_file}")
    
    return df, trip_summary

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(use_synthetic=False):
    """
    Run complete preprocessing pipeline
    """
    print("\n" + "="*70)
    print("BEIJING T-DRIVE PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Get data
    if use_synthetic:
        create_synthetic_sample()
    else:
        if not download_sample_data():
            print("\nNo data available. Creating synthetic sample for testing...")
            create_synthetic_sample()
    
    # Step 2: Load raw data
    df = load_raw_data()
    if df is None:
        print("Failed to load data!")
        return None, None
    
    # Step 3: Preprocess
    df = preprocess_data(df)
    
    # Step 4: Segment trips
    df = segment_trips(df)
    
    # Step 5: Save
    df, summary = save_processed_data(df)
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nReady for clustering:")
    print(f"  Data file: data/beijing_tdrive/processed/beijing_trips.csv")
    print(f"  Summary: data/beijing_tdrive/processed/trip_summary.csv")
    
    return df, summary


if __name__ == "__main__":
    # Check for --synthetic flag
    use_synthetic = '--synthetic' in sys.argv
    
    df, summary = run_preprocessing_pipeline(use_synthetic=use_synthetic)
    
    if df is not None:
        print("\nNext step:")
        print("  python clustering/beijing_clustering.py")
