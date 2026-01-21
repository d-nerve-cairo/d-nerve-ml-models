"""
Cairo Microbus Route Generator - HARD MODE
Generates more challenging data for robust evaluation:
1. Higher GPS noise (30-50m)
2. Overlapping routes (shared road segments)
3. Variable trip patterns

Author: D-Nerve Team
Date: January 2026
"""

import openrouteservice
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import random

# ============================================================================
# CONFIGURATION - HARD MODE
# ============================================================================

API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjMxMGI1M2E2YjliOTRiYWJhMGM3NGIwZWNmMTAyMWMzIiwiaCI6Im11cm11cjY0In0="  # <-- REPLACE WITH YOUR KEY

OUTPUT_DIR = "data/cairo_hard/raw"

# Challenge settings
NUM_ROUTES = 40  # More routes
TRIPS_PER_ROUTE = 15  # More trips per route
GPS_NOISE_LEVELS = [30, 40, 50]  # Variable noise (meters)
SAMPLING_RATE_SECONDS = 5

# ============================================================================
# CAIRO HUBS (Same as before)
# ============================================================================

CAIRO_HUBS = {
    "ramses": {"name": "Ramses Square", "coords": [31.2466, 30.0619], "type": "major"},
    "tahrir": {"name": "Tahrir Square", "coords": [31.2357, 30.0444], "type": "major"},
    "giza": {"name": "Giza Square", "coords": [31.2089, 30.0131], "type": "major"},
    "abdel_moneim": {"name": "Abdel Moneim Riad", "coords": [31.2340, 30.0453], "type": "major"},
    "ataba": {"name": "Ataba Square", "coords": [31.2469, 30.0531], "type": "major"},
    "maadi": {"name": "Maadi", "coords": [31.2569, 29.9602], "type": "district"},
    "heliopolis": {"name": "Heliopolis", "coords": [31.3225, 30.0866], "type": "district"},
    "nasr_city": {"name": "Nasr City", "coords": [31.3656, 30.0511], "type": "district"},
    "shubra": {"name": "Shubra", "coords": [31.2422, 30.0986], "type": "district"},
    "mohandessin": {"name": "Mohandessin", "coords": [31.2003, 30.0609], "type": "district"},
    "dokki": {"name": "Dokki", "coords": [31.2125, 30.0392], "type": "district"},
    "ain_shams": {"name": "Ain Shams", "coords": [31.3194, 30.1311], "type": "district"},
    "zeitoun": {"name": "Zeitoun", "coords": [31.3000, 30.1167], "type": "district"},
    "abbassia": {"name": "Abbassia", "coords": [31.2833, 30.0722], "type": "district"},
    "imbaba": {"name": "Imbaba", "coords": [31.2078, 30.0758], "type": "district"},
    "dar_el_salam": {"name": "Dar El Salam", "coords": [31.2417, 29.9833], "type": "district"},
    "6october": {"name": "6th October City", "coords": [30.9167, 29.9389], "type": "satellite"},
    "new_cairo": {"name": "New Cairo", "coords": [31.4700, 30.0300], "type": "satellite"},
    "helwan": {"name": "Helwan", "coords": [31.3340, 29.8500], "type": "satellite"},
    # Additional hubs for more routes
    "zamalek": {"name": "Zamalek", "coords": [31.2194, 30.0609], "type": "district"},
    "garden_city": {"name": "Garden City", "coords": [31.2311, 30.0356], "type": "district"},
    "sayeda_zeinab": {"name": "Sayeda Zeinab", "coords": [31.2356, 30.0287], "type": "district"},
    "el_matareya": {"name": "El Matareya", "coords": [31.3133, 30.1214], "type": "district"},
}

# ============================================================================
# OVERLAPPING ROUTE PATTERNS - KEY FOR HARD MODE
# ============================================================================

# These routes INTENTIONALLY share road segments (overlapping)
# This tests whether DBSCAN can distinguish routes that partially overlap

OVERLAPPING_ROUTE_GROUPS = [
    # Group 1: Routes from Ramses going different directions (share Ramses area)
    [
        ("ramses", "giza"),
        ("ramses", "tahrir"),
        ("ramses", "dokki"),
        ("ramses", "mohandessin"),
    ],
    
    # Group 2: Routes through Tahrir (all pass through Tahrir)
    [
        ("ramses", "tahrir"),
        ("tahrir", "maadi"),
        ("tahrir", "garden_city"),
        ("ataba", "tahrir"),
    ],
    
    # Group 3: Routes to Heliopolis area (share eastern corridor)
    [
        ("ramses", "heliopolis"),
        ("ramses", "nasr_city"),
        ("abbassia", "heliopolis"),
        ("abbassia", "nasr_city"),
    ],
    
    # Group 4: Giza corridor (share western roads)
    [
        ("tahrir", "giza"),
        ("dokki", "giza"),
        ("mohandessin", "giza"),
        ("tahrir", "6october"),  # Goes through Giza
    ],
    
    # Group 5: Southern corridor (share southern roads)
    [
        ("tahrir", "maadi"),
        ("maadi", "helwan"),
        ("sayeda_zeinab", "maadi"),
        ("garden_city", "maadi"),
    ],
    
    # Group 6: Northern routes (share northern roads)
    [
        ("ramses", "shubra"),
        ("shubra", "imbaba"),
        ("ataba", "shubra"),
    ],
    
    # Group 7: New Cairo corridor
    [
        ("nasr_city", "new_cairo"),
        ("heliopolis", "new_cairo"),
    ],
    
    # Group 8: Cross-city routes (longer, more distinct)
    [
        ("giza", "heliopolis"),
        ("6october", "nasr_city"),
        ("helwan", "shubra"),
    ],
]

# Flatten to get all route patterns
ROUTE_PATTERNS = []
for group in OVERLAPPING_ROUTE_GROUPS:
    for route in group:
        if route not in ROUTE_PATTERNS:
            ROUTE_PATTERNS.append(route)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_gps_noise(lat, lon, noise_meters):
    """Add realistic GPS noise with variable intensity"""
    noise_lat = np.random.normal(0, noise_meters / 111000)
    noise_lon = np.random.normal(0, noise_meters / (111000 * np.cos(np.radians(lat))))
    return lat + noise_lat, lon + noise_lon

def add_systematic_drift(lat, lon, trip_num, drift_factor=0.0001):
    """
    Add systematic drift to simulate different drivers taking slightly different paths
    This makes trips from same route slightly different
    """
    # Each trip has a consistent offset (simulates driver preference)
    np.random.seed(trip_num * 42)  # Reproducible per trip
    drift_lat = np.random.uniform(-drift_factor, drift_factor)
    drift_lon = np.random.uniform(-drift_factor, drift_factor)
    return lat + drift_lat, lon + drift_lon

def interpolate_trajectory(coords, duration_seconds, sampling_rate, noise_level, trip_num):
    """
    Interpolate coordinates with variable noise and drift
    """
    if len(coords) < 2:
        return [(coords[0][0], coords[0][1], 0)]
    
    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111000
        distances.append(distances[-1] + dist)
    
    total_distance = distances[-1]
    if total_distance == 0:
        return [(coords[0][0], coords[0][1], 0)]
    
    num_points = int(duration_seconds / sampling_rate) + 1
    trajectory = []
    
    for i in range(num_points):
        t = i * sampling_rate
        progress = min(t / duration_seconds, 1.0)
        target_dist = progress * total_distance
        
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                seg_start = distances[j-1]
                seg_end = distances[j]
                seg_progress = (target_dist - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0
                
                lat = coords[j-1][0] + seg_progress * (coords[j][0] - coords[j-1][0])
                lon = coords[j-1][1] + seg_progress * (coords[j][1] - coords[j-1][1])
                
                # Add systematic drift (driver variation)
                lat, lon = add_systematic_drift(lat, lon, trip_num)
                
                # Add random GPS noise
                lat_noisy, lon_noisy = add_gps_noise(lat, lon, noise_level)
                trajectory.append((lat_noisy, lon_noisy, t))
                break
        else:
            lat, lon = coords[-1]
            lat, lon = add_systematic_drift(lat, lon, trip_num)
            lat_noisy, lon_noisy = add_gps_noise(lat, lon, noise_level)
            trajectory.append((lat_noisy, lon_noisy, t))
    
    return trajectory

# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================

class CairoHardModeGenerator:
    def __init__(self, api_key):
        self.client = openrouteservice.Client(key=api_key)
        self.all_trajectories = []
        self.route_metadata = []
        
    def get_route(self, origin_coords, dest_coords):
        """Get driving route from OpenRouteService"""
        try:
            routes = self.client.directions(
                coordinates=[origin_coords, dest_coords],
                profile='driving-car',
                format='geojson',
                instructions=False
            )
            
            if routes and 'features' in routes and len(routes['features']) > 0:
                feature = routes['features'][0]
                coords = feature['geometry']['coordinates']
                coords_latlon = [(c[1], c[0]) for c in coords]
                
                props = feature['properties']['summary']
                duration = props['duration']
                distance = props['distance']
                
                return {
                    'coordinates': coords_latlon,
                    'duration': duration,
                    'distance': distance
                }
        except Exception as e:
            print(f"  Error getting route: {e}")
            return None
        
        return None
    
    def generate_route_trajectories(self, route_id, origin_key, dest_key, 
                                     num_trips=TRIPS_PER_ROUTE, overlap_group=None):
        """Generate trips for a route with variable noise"""
        
        origin = CAIRO_HUBS[origin_key]
        dest = CAIRO_HUBS[dest_key]
        
        overlap_str = f" [Overlap Group {overlap_group}]" if overlap_group else ""
        print(f"\nRoute {route_id}: {origin['name']} → {dest['name']}{overlap_str}")
        
        route_data = self.get_route(origin['coords'], dest['coords'])
        
        if not route_data:
            print(f"  ❌ Failed to get route")
            return []
        
        print(f"  ✓ Distance: {route_data['distance']/1000:.1f} km, Duration: {route_data['duration']/60:.1f} min")
        
        trajectories = []
        base_time = datetime(2024, 1, 15, 6, 0, 0)
        
        for trip_num in range(num_trips):
            # Randomly select noise level for this trip
            noise_level = random.choice(GPS_NOISE_LEVELS)
            
            # Vary trip time
            hour_offset = random.choice([0, 1, 2, 7, 8, 9, 12, 13, 17, 18, 19])
            day_offset = random.randint(0, 13)  # 2 weeks
            trip_start = base_time + timedelta(days=day_offset, hours=hour_offset, 
                                                minutes=random.randint(0, 59))
            
            # Traffic variation (wider range)
            traffic_factor = random.uniform(0.7, 1.8)
            trip_duration = route_data['duration'] * traffic_factor
            
            # Generate trajectory with noise and drift
            traj_points = interpolate_trajectory(
                route_data['coordinates'],
                trip_duration,
                SAMPLING_RATE_SECONDS,
                noise_level,
                trip_num
            )
            
            trip_id = f"R{route_id:03d}_T{trip_num:03d}"
            
            for lat, lon, time_offset in traj_points:
                timestamp = trip_start + timedelta(seconds=time_offset)
                trajectories.append({
                    'trip_id': trip_id,
                    'route_id': route_id,
                    'origin': origin_key,
                    'destination': dest_key,
                    'overlap_group': overlap_group,
                    'noise_level': noise_level,
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'time_offset_sec': time_offset
                })
        
        print(f"  ✓ Generated {num_trips} trips (noise: {min(GPS_NOISE_LEVELS)}-{max(GPS_NOISE_LEVELS)}m)")
        
        # Store metadata
        self.route_metadata.append({
            'route_id': route_id,
            'origin': origin_key,
            'destination': dest_key,
            'overlap_group': overlap_group,
            'distance_km': route_data['distance'] / 1000,
            'base_duration_min': route_data['duration'] / 60
        })
        
        return trajectories
    
    def generate_all_routes(self):
        """Generate all overlapping routes"""
        
        print("="*70)
        print("CAIRO ROUTE GENERATION - HARD MODE")
        print("="*70)
        print(f"Total route patterns: {len(ROUTE_PATTERNS)}")
        print(f"Trips per route: {TRIPS_PER_ROUTE}")
        print(f"GPS noise levels: {GPS_NOISE_LEVELS} meters")
        print(f"Features: Overlapping routes, variable noise, driver drift")
        print("="*70)
        
        all_trajectories = []
        route_id = 1
        
        # Generate routes by overlap group
        for group_idx, group in enumerate(OVERLAPPING_ROUTE_GROUPS):
            print(f"\n--- Overlap Group {group_idx + 1} ({len(group)} routes) ---")
            
            for origin_key, dest_key in group:
                trajectories = self.generate_route_trajectories(
                    route_id=route_id,
                    origin_key=origin_key,
                    dest_key=dest_key,
                    overlap_group=group_idx + 1
                )
                all_trajectories.extend(trajectories)
                route_id += 1
                
                # Rate limiting
                time.sleep(1.5)
        
        self.all_trajectories = all_trajectories
        return all_trajectories
    
    def save_data(self, output_dir=OUTPUT_DIR):
        """Save generated data"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.all_trajectories:
            print("No trajectories to save!")
            return
        
        df = pd.DataFrame(self.all_trajectories)
        
        # Save full dataset
        full_path = os.path.join(output_dir, 'cairo_hard_trajectories.csv')
        df.to_csv(full_path, index=False)
        print(f"\n✓ Saved: {full_path}")
        print(f"  Total records: {len(df):,}")
        
        # Save route metadata
        meta_df = pd.DataFrame(self.route_metadata)
        meta_path = os.path.join(output_dir, 'cairo_hard_route_metadata.csv')
        meta_df.to_csv(meta_path, index=False)
        print(f"✓ Saved: {meta_path}")
        
        # Save overlap group info
        overlap_info = []
        for group_idx, group in enumerate(OVERLAPPING_ROUTE_GROUPS):
            for origin, dest in group:
                overlap_info.append({
                    'overlap_group': group_idx + 1,
                    'origin': origin,
                    'destination': dest,
                    'description': f"Routes sharing {CAIRO_HUBS[origin]['name']} area"
                })
        overlap_df = pd.DataFrame(overlap_info)
        overlap_path = os.path.join(output_dir, 'overlap_groups.csv')
        overlap_df.to_csv(overlap_path, index=False)
        print(f"✓ Saved: {overlap_path}")
        
        # Summary statistics
        print("\n" + "="*70)
        print("HARD MODE DATA SUMMARY")
        print("="*70)
        print(f"Routes: {df['route_id'].nunique()}")
        print(f"Trips: {df['trip_id'].nunique()}")
        print(f"GPS Points: {len(df):,}")
        print(f"Overlap Groups: {len(OVERLAPPING_ROUTE_GROUPS)}")
        print(f"\nNoise distribution:")
        noise_counts = df.groupby('trip_id')['noise_level'].first().value_counts()
        for noise, count in noise_counts.items():
            print(f"  {noise}m noise: {count} trips")
        
        return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ERROR: Please set your OpenRouteService API key!")
        exit(1)
    
    generator = CairoHardModeGenerator(API_KEY)
    trajectories = generator.generate_all_routes()
    df = generator.save_data()
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print(f"\nNext step:")
    print(f"  python preprocessing/cairo_preprocessing.py  # Update input path first!")
    print(f"  python clustering/cairo_dbscan_routes.py")