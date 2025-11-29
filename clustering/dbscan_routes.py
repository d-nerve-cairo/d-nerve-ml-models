"""
DBSCAN-based route discovery

Clusters similar trajectories using:
- Hausdorff distance for trajectory similarity
- DBSCAN to find route clusters
- Representative route extraction

Author: Group 2 - ML Team
Environment: dnervenv
Platform: Windows
Target: F1 ≥ 0.85
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix imports on Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config, create_output_dir

# DBSCAN Parameters (these will be tuned later)
EPSILON = 300  # meters - maximum distance between trajectories in same cluster
MIN_SAMPLES = 5  # minimum trips to form a route

def extract_trajectory_points(df, trip_id):
    """
    Extract lat/lon points for a single trip
    Returns: numpy array of shape (n_points, 2) - [[lon, lat], ...]
    """
    trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
    coords = trip_data[['longitude', 'latitude']].values
    return coords

def hausdorff_distance_meters(traj1, traj2):
    """
    Calculate Hausdorff distance between two trajectories
    
    Args:
        traj1, traj2: numpy arrays of shape (n_points, 2) with [lon, lat]
    
    Returns:
        Distance in meters (approximate)
    """
    # Hausdorff distance in degrees
    dist_deg = max(
        directed_hausdorff(traj1, traj2)[0],
        directed_hausdorff(traj2, traj1)[0]
    )
    
    # Convert degrees to meters (approximate at Beijing latitude ~40°)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 85 km at 40°N
    dist_meters = dist_deg * 100000  # rough approximation
    
    return dist_meters

def compute_distance_matrix(df, trip_ids, sample_size=None):
    """
    Compute pairwise distance matrix for all trips
    
    Args:
        df: DataFrame with GPS points
        trip_ids: List of trip IDs to compare
        sample_size: If set, only use first N trips (for faster testing)
    
    Returns:
        Distance matrix (n_trips x n_trips)
    """
    if sample_size:
        trip_ids = trip_ids[:sample_size]
    
    n_trips = len(trip_ids)
    print(f"\nComputing distance matrix for {n_trips} trips...")
    print("(This may take 5-10 minutes)\n")
    
    dist_matrix = np.zeros((n_trips, n_trips))
    
    # Extract all trajectories first
    print("Extracting trajectories...")
    trajectories = {}
    for trip_id in tqdm(trip_ids, desc="Loading trips"):
        trajectories[trip_id] = extract_trajectory_points(df, trip_id)
    
    # Compute pairwise distances
    print("\nComputing pairwise Hausdorff distances...")
    total_comparisons = (n_trips * (n_trips - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="Computing distances") as pbar:
        for i in range(n_trips):
            for j in range(i+1, n_trips):
                dist = hausdorff_distance_meters(
                    trajectories[trip_ids[i]],
                    trajectories[trip_ids[j]]
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
                pbar.update(1)
    
    print("\n Distance matrix complete\n")
    return dist_matrix, trip_ids

def cluster_trajectories(dist_matrix, eps, min_samples):
    """
    Apply DBSCAN clustering
    
    Args:
        dist_matrix: Precomputed distance matrix
        eps: Maximum distance between trajectories (meters)
        min_samples: Minimum trips to form route
    
    Returns:
        Cluster labels (-1 = noise/outlier, 0+ = route cluster)
    """
    print(f"Running DBSCAN (ε={eps}m, MinPts={min_samples})...")
    
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    labels = clustering.fit_predict(dist_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n Clustering complete:")
    print(f"   Routes discovered: {n_clusters}")
    print(f"   Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    return labels

def extract_route_centerlines(df, trip_ids, labels):
    """
    For each cluster, compute representative route (average trajectory)
    
    Returns:
        Dictionary mapping route_id to route information
    """
    routes = {}
    
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label
    
    print(f"\nExtracting centerlines for {len(unique_labels)} routes...")
    
    for route_id in unique_labels:
        # Get all trips in this cluster
        cluster_trips = [trip_ids[i] for i, label in enumerate(labels) if label == route_id]
        
        print(f"  Route {route_id}: {len(cluster_trips)} supporting trips")
        
        # Extract all trajectories
        trajectories = [extract_trajectory_points(df, trip_id) for trip_id in cluster_trips]
        
        # Use first trajectory as representative (simplification)
        # TODO: Implement proper averaging/interpolation
        centerline = trajectories[0]
        
        routes[f"route_{route_id:03d}"] = {
            'centerline': centerline,
            'num_trips': len(cluster_trips),
            'trip_ids': cluster_trips
        }
    
    return routes

def visualize_routes(routes, output_dir):
    """
    Plot discovered routes on map
    """
    output_dir = Path(output_dir)
    create_output_dir(output_dir)
    
    print(f"\n Visualizing routes...")
    
    # Plot all routes together
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(routes), 20)))
    
    for idx, (route_id, route_data) in enumerate(routes.items()):
        centerline = route_data['centerline']
        color = colors[idx % 20]
        
        plt.plot(
            centerline[:, 0],  # longitude
            centerline[:, 1],  # latitude
            linewidth=2.5,
            alpha=0.7,
            label=f"{route_id} ({route_data['num_trips']} trips)",
            color=color
        )
    
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(f'Discovered Routes in Beijing (n={len(routes)})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = output_dir / "discovered_routes_all.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f" Saved: {output_file}")
    
    plt.close()
    
    # Plot top 5 routes individually
    sorted_routes = sorted(routes.items(), key=lambda x: x[1]['num_trips'], reverse=True)
    
    for route_id, route_data in sorted_routes[:5]:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        centerline = route_data['centerline']
        
        # Plot centerline
        ax.plot(centerline[:, 0], centerline[:, 1], 
                'b-', linewidth=3, label='Route centerline', zorder=3)
        
        # Mark start and end
        ax.scatter(centerline[0, 0], centerline[0, 1], 
                   c='green', s=250, marker='o', label='Start', zorder=4, 
                   edgecolor='white', linewidth=2)
        ax.scatter(centerline[-1, 0], centerline[-1, 1], 
                   c='red', s=250, marker='s', label='End', zorder=4, 
                   edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'{route_id} - {route_data["num_trips"]} supporting trips', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        output_file = output_dir / f"{route_id}_detail.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f" Top 5 route plots saved to {output_dir}/")

def save_results(routes, labels, trip_ids, output_dir, parameters):
    """Save clustering results"""
    import pickle
    
    output_dir = Path(output_dir)
    create_output_dir(output_dir)
    
    results = {
        'routes': routes,
        'labels': labels,
        'trip_ids': trip_ids,
        'parameters': parameters
    }
    
    output_file = output_dir / "route_discovery_results.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f" Results saved to {output_file}")

def run_route_discovery(input_file, output_dir, sample_trips=500):
    """
    Complete DBSCAN route discovery pipeline
    
    Args:
        input_file: Path to trips parquet file
        output_dir: Where to save results
        sample_trips: Number of trips to use (use fewer for testing)
    """
    print("="*70)
    print("DBSCAN ROUTE DISCOVERY")
    print("="*70)
    
    # Load trips
    print(f"\n Loading trips from {input_file}...")
    df = pd.read_parquet(input_file)
    
    trip_ids = df['trip_id'].unique().tolist()
    print(f"Total trips available: {len(trip_ids)}")
    
    if sample_trips and len(trip_ids) > sample_trips:
        print(f"Using first {sample_trips} trips for analysis (faster processing)")
        trip_ids_subset = trip_ids[:sample_trips]
    else:
        trip_ids_subset = trip_ids
    
    # Compute distance matrix
    dist_matrix, used_trip_ids = compute_distance_matrix(df, trip_ids_subset, sample_size=sample_trips)
    
    # Cluster with DBSCAN
    labels = cluster_trajectories(dist_matrix, eps=EPSILON, min_samples=MIN_SAMPLES)
    
    # Extract routes
    routes = extract_route_centerlines(df, used_trip_ids, labels)
    
    # Visualize
    if len(routes) > 0:
        visualize_routes(routes, output_dir)
    else:
        print("\n  No routes discovered! Try adjusting DBSCAN parameters.")
    
    # Save results
    parameters = {
        'epsilon': EPSILON,
        'min_samples': MIN_SAMPLES,
        'num_routes': len(routes),
        'num_trips_analyzed': len(used_trip_ids)
    }
    save_results(routes, labels, used_trip_ids, output_dir, parameters)
    
    print("\n" + "="*70)
    print(" ROUTE DISCOVERY COMPLETE")
    print("="*70)
    print(f"Discovered {len(routes)} routes from {len(used_trip_ids)} trips")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Check visualizations in outputs/route_discovery/")
    print(f"  2. If too few/many routes, tune EPSILON and MIN_SAMPLES")
    print(f"  3. Proceed to ETA prediction model")
    
    return routes

if __name__ == "__main__":
    config = load_config()
    
    INPUT_FILE = Path(config['data']['processed_dir']) / "tdrive_100taxis_trips.parquet"
    OUTPUT_DIR = Path(config['outputs']['route_discovery_dir'])
    
    # Start with 500 trips for speed (can use all 2,181 later)
    routes = run_route_discovery(INPUT_FILE, OUTPUT_DIR, sample_trips=500)