"""
Cairo Microbus Route Generator using OpenRouteService
D-Nerve Project - Synthetic GPS Trajectory Data Generation

This script generates realistic GPS trajectory data for Cairo's informal
transit network using OpenRouteService (free alternative to Google Maps).

Author: D-Nerve Team
Date: January 2026
"""

import openrouteservice
from openrouteservice.directions import directions
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# OpenRouteService API Key
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjMxMGI1M2E2YjliOTRiYWJhMGM3NGIwZWNmMTAyMWMzIiwiaCI6Im11cm11cjY0In0="  



# Output directories
OUTPUT_DIR = "data/cairo/raw"
PROCESSED_DIR = "data/cairo/processed"

# Route generation parameters
NUM_ROUTES = 50  # Number of unique routes to generate (start small, increase later)
TRIPS_PER_ROUTE = 10  # Number of trips per route (simulates multiple drivers/times)
GPS_NOISE_METERS = 15  # Realistic GPS noise (5-20 meters typical)
SAMPLING_RATE_SECONDS = 5  # GPS point every 5 seconds

# ============================================================================
# CAIRO TRANSIT HUBS
# ============================================================================

CAIRO_HUBS = {
    # Major Terminals (high connectivity)
    "ramses": {"name": "Ramses Square", "coords": [31.2466, 30.0619], "type": "major"},
    "tahrir": {"name": "Tahrir Square", "coords": [31.2357, 30.0444], "type": "major"},
    "giza": {"name": "Giza Square", "coords": [31.2089, 30.0131], "type": "major"},
    "abdel_moneim": {"name": "Abdel Moneim Riad", "coords": [31.2340, 30.0453], "type": "major"},
    "ataba": {"name": "Ataba Square", "coords": [31.2469, 30.0531], "type": "major"},
    
    # District Hubs (medium connectivity)
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
    
    # Satellite Cities (lower frequency, longer routes)
    "6october": {"name": "6th October City", "coords": [30.9167, 29.9389], "type": "satellite"},
    "new_cairo": {"name": "New Cairo", "coords": [31.4700, 30.0300], "type": "satellite"},
    "helwan": {"name": "Helwan", "coords": [31.3340, 29.8500], "type": "satellite"},
}

# Realistic Cairo microbus route patterns
# Format: (origin_key, destination_key, frequency_weight)
ROUTE_PATTERNS = [
    # Major terminal connections (highest frequency)
    ("ramses", "giza", 10),
    ("ramses", "tahrir", 10),
    ("tahrir", "giza", 10),
    ("ramses", "ataba", 8),
    ("ataba", "tahrir", 8),
    
    # Downtown to districts
    ("ramses", "shubra", 8),
    ("ramses", "heliopolis", 7),
    ("ramses", "nasr_city", 7),
    ("tahrir", "maadi", 8),
    ("tahrir", "dokki", 7),
    ("tahrir", "mohandessin", 7),
    ("giza", "dokki", 6),
    ("giza", "mohandessin", 6),
    ("ataba", "abbassia", 6),
    ("ataba", "zeitoun", 5),
    
    # District to district
    ("heliopolis", "nasr_city", 6),
    ("nasr_city", "ain_shams", 5),
    ("maadi", "dar_el_salam", 5),
    ("shubra", "imbaba", 5),
    ("dokki", "mohandessin", 5),
    ("abbassia", "heliopolis", 5),
    
    # Satellite city connections
    ("giza", "6october", 4),
    ("tahrir", "6october", 3),
    ("nasr_city", "new_cairo", 4),
    ("heliopolis", "new_cairo", 4),
    ("maadi", "helwan", 5),
    ("tahrir", "helwan", 3),
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_gps_noise(lat, lon, noise_meters=GPS_NOISE_METERS):
    """Add realistic GPS noise to coordinates."""
    # Convert meters to approximate degrees
    # 1 degree latitude ≈ 111,000 meters
    # 1 degree longitude ≈ 111,000 * cos(latitude) meters
    noise_lat = np.random.normal(0, noise_meters / 111000)
    noise_lon = np.random.normal(0, noise_meters / (111000 * np.cos(np.radians(lat))))
    return lat + noise_lat, lon + noise_lon


def decode_polyline(encoded):
    """Decode Google-style encoded polyline to list of coordinates."""
    # ORS returns coordinates directly, but this is backup for encoded routes
    decoded = []
    index = 0
    lat = 0
    lng = 0
    
    while index < len(encoded):
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if result & 1 else result >> 1
        lat += dlat
        
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if result & 1 else result >> 1
        lng += dlng
        
        decoded.append((lat / 1e5, lng / 1e5))
    
    return decoded


def interpolate_trajectory(coords, duration_seconds, sampling_rate=SAMPLING_RATE_SECONDS):
    """
    Interpolate coordinates to create GPS points at regular time intervals.
    
    Args:
        coords: List of (lat, lon) tuples from route
        duration_seconds: Total trip duration
        sampling_rate: Seconds between GPS points
    
    Returns:
        List of (lat, lon, timestamp_offset) tuples
    """
    if len(coords) < 2:
        return [(coords[0][0], coords[0][1], 0)]
    
    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        # Haversine distance (simplified)
        dist = np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111000  # approx meters
        distances.append(distances[-1] + dist)
    
    total_distance = distances[-1]
    if total_distance == 0:
        return [(coords[0][0], coords[0][1], 0)]
    
    # Generate points at regular time intervals
    num_points = int(duration_seconds / sampling_rate) + 1
    trajectory = []
    
    for i in range(num_points):
        t = i * sampling_rate
        progress = min(t / duration_seconds, 1.0)  # 0 to 1
        target_dist = progress * total_distance
        
        # Find segment
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                # Interpolate within segment
                seg_start = distances[j-1]
                seg_end = distances[j]
                seg_progress = (target_dist - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0
                
                lat = coords[j-1][0] + seg_progress * (coords[j][0] - coords[j-1][0])
                lon = coords[j-1][1] + seg_progress * (coords[j][1] - coords[j-1][1])
                
                # Add GPS noise
                lat_noisy, lon_noisy = add_gps_noise(lat, lon)
                trajectory.append((lat_noisy, lon_noisy, t))
                break
        else:
            # At the end
            lat, lon = coords[-1]
            lat_noisy, lon_noisy = add_gps_noise(lat, lon)
            trajectory.append((lat_noisy, lon_noisy, t))
    
    return trajectory


# ============================================================================
# MAIN ROUTE GENERATION
# ============================================================================

class CairoRouteGenerator:
    def __init__(self, api_key):
        self.client = openrouteservice.Client(key=api_key)
        self.generated_routes = []
        self.all_trajectories = []
        
    def get_route(self, origin_coords, dest_coords):
        """
        Get driving route from OpenRouteService.
        
        Args:
            origin_coords: [longitude, latitude] (ORS format)
            dest_coords: [longitude, latitude]
        
        Returns:
            dict with 'coordinates', 'duration', 'distance' or None if failed
        """
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
                # Convert from [lon, lat] to [lat, lon]
                coords_latlon = [(c[1], c[0]) for c in coords]
                
                props = feature['properties']['summary']
                duration = props['duration']  # seconds
                distance = props['distance']  # meters
                
                return {
                    'coordinates': coords_latlon,
                    'duration': duration,
                    'distance': distance
                }
        except Exception as e:
            print(f"  Error getting route: {e}")
            return None
        
        return None
    
    def generate_route_trajectories(self, route_id, origin_key, dest_key, num_trips=TRIPS_PER_ROUTE):
        """Generate multiple trip trajectories for a single route."""
        
        origin = CAIRO_HUBS[origin_key]
        dest = CAIRO_HUBS[dest_key]
        
        print(f"\nRoute {route_id}: {origin['name']} → {dest['name']}")
        
        # Get base route from ORS
        route_data = self.get_route(origin['coords'], dest['coords'])
        
        if not route_data:
            print(f"  ❌ Failed to get route")
            return []
        
        print(f"  ✓ Distance: {route_data['distance']/1000:.1f} km, Base duration: {route_data['duration']/60:.1f} min")
        
        trajectories = []
        base_time = datetime(2024, 1, 15, 6, 0, 0)  # Start date for simulation
        
        for trip_num in range(num_trips):
            # Vary the trip time (different times of day)
            hour_offset = random.choice([0, 1, 2, 7, 8, 9, 12, 13, 17, 18, 19])  # Peak hours
            day_offset = random.randint(0, 6)  # Different days
            trip_start = base_time + timedelta(days=day_offset, hours=hour_offset, minutes=random.randint(0, 59))
            
            # Add traffic variation to duration (±30%)
            traffic_factor = random.uniform(0.8, 1.5)
            trip_duration = route_data['duration'] * traffic_factor
            
            # Generate trajectory points
            traj_points = interpolate_trajectory(
                route_data['coordinates'],
                trip_duration,
                SAMPLING_RATE_SECONDS
            )
            
            # Create trajectory records
            trip_id = f"R{route_id:03d}_T{trip_num:03d}"
            
            for lat, lon, time_offset in traj_points:
                timestamp = trip_start + timedelta(seconds=time_offset)
                trajectories.append({
                    'trip_id': trip_id,
                    'route_id': route_id,
                    'origin': origin_key,
                    'destination': dest_key,
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'time_offset_sec': time_offset
                })
        
        print(f"  ✓ Generated {num_trips} trips, {len(trajectories)} GPS points")
        return trajectories
    
    def generate_all_routes(self, num_routes=NUM_ROUTES):
        """Generate trajectories for multiple routes."""
        
        print("="*60)
        print("CAIRO MICROBUS ROUTE GENERATION")
        print("="*60)
        print(f"Target routes: {num_routes}")
        print(f"Trips per route: {TRIPS_PER_ROUTE}")
        print(f"GPS noise: ±{GPS_NOISE_METERS} meters")
        print(f"Sampling rate: {SAMPLING_RATE_SECONDS} seconds")
        print("="*60)
        
        # Select routes based on weighted patterns
        weights = [p[2] for p in ROUTE_PATTERNS]
        total_weight = sum(weights)
        probs = [w/total_weight for w in weights]
        
        selected_patterns = np.random.choice(
            len(ROUTE_PATTERNS), 
            size=min(num_routes, len(ROUTE_PATTERNS)),
            replace=False,
            p=probs
        )
        
        all_trajectories = []
        
        for i, pattern_idx in enumerate(selected_patterns):
            origin_key, dest_key, _ = ROUTE_PATTERNS[pattern_idx]
            
            trajectories = self.generate_route_trajectories(
                route_id=i+1,
                origin_key=origin_key,
                dest_key=dest_key
            )
            
            all_trajectories.extend(trajectories)
            
            # Rate limiting (ORS allows 40 requests/minute)
            time.sleep(1.5)
        
        self.all_trajectories = all_trajectories
        return all_trajectories
    
    def save_data(self, output_dir=OUTPUT_DIR):
        """Save generated trajectories to CSV files."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.all_trajectories:
            print("No trajectories to save!")
            return
        
        df = pd.DataFrame(self.all_trajectories)
        
        # Save full dataset
        full_path = os.path.join(output_dir, 'cairo_trajectories_full.csv')
        df.to_csv(full_path, index=False)
        print(f"\n✓ Saved full dataset: {full_path}")
        print(f"  Total records: {len(df)}")
        
        # Save individual trip files (mimics Beijing T-Drive format)
        trips_dir = os.path.join(output_dir, 'trips')
        os.makedirs(trips_dir, exist_ok=True)
        
        for trip_id in df['trip_id'].unique():
            trip_df = df[df['trip_id'] == trip_id][['latitude', 'longitude', 'timestamp']]
            trip_path = os.path.join(trips_dir, f'{trip_id}.csv')
            trip_df.to_csv(trip_path, index=False)
        
        print(f"✓ Saved {df['trip_id'].nunique()} individual trip files to: {trips_dir}")
        
        # Save route summary
        route_summary = df.groupby('route_id').agg({
            'origin': 'first',
            'destination': 'first',
            'trip_id': 'nunique',
            'latitude': 'count'
        }).rename(columns={'trip_id': 'num_trips', 'latitude': 'total_points'})
        
        summary_path = os.path.join(output_dir, 'route_summary.csv')
        route_summary.to_csv(summary_path)
        print(f"✓ Saved route summary: {summary_path}")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if API key is set
    if API_KEY == "YOUR_API_KEY_HERE":
        print("❌ ERROR: Please set your OpenRouteService API key!")
        print("   Edit this file and replace 'YOUR_API_KEY_HERE' with your key")
        print("   Or load it from file (see commented code)")
        exit(1)
    
    # Create generator
    generator = CairoRouteGenerator(API_KEY)
    
    # Generate routes (start with fewer for testing)
    trajectories = generator.generate_all_routes(num_routes=NUM_ROUTES)
    
    # Save data
    df = generator.save_data()
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Routes generated: {df['route_id'].nunique()}")
    print(f"Total trips: {df['trip_id'].nunique()}")
    print(f"Total GPS points: {len(df)}")
    print(f"\nNext step: Run preprocessing and train DBSCAN model")
