"""
Beijing T-Drive Real Data Clustering
Uses existing tdrive_100taxis_trips.parquet

Author: D-Nerve Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/processed/tdrive_100taxis_trips.parquet"
OUTPUT_DIR = Path("outputs/beijing_real_clustering")

# Sampling for faster computation
MAX_TRIPS = 500  # Use 500 trips for reasonable computation time

# DBSCAN parameters
EPSILON_VALUES = [300, 400, 500, 600, 800, 1000, 1500]  # meters
MIN_SAMPLES_VALUES = [2, 3, 4, 5]

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def load_data():
    """Load real T-Drive data"""
    print("="*70)
    print("BEIJING T-DRIVE REAL DATA CLUSTERING")
    print("="*70)
    
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"  Total GPS points: {len(df):,}")
    print(f"  Total trips: {df['trip_id'].nunique():,}")
    print(f"  Total taxis: {df['taxi_id'].nunique()}")
    
    return df


def filter_valid_trips(df, min_points=10, max_points=500):
    """Filter trips by point count"""
    print(f"\nFiltering trips ({min_points}-{max_points} points)...")
    
    trip_sizes = df.groupby('trip_id').size()
    valid_trips = trip_sizes[(trip_sizes >= min_points) & (trip_sizes <= max_points)].index
    
    df_filtered = df[df['trip_id'].isin(valid_trips)]
    
    print(f"  Valid trips: {len(valid_trips):,}")
    print(f"  GPS points: {len(df_filtered):,}")
    
    return df_filtered


def sample_trips(df, max_trips=MAX_TRIPS):
    """Sample trips for faster computation"""
    unique_trips = df['trip_id'].unique()
    
    if len(unique_trips) > max_trips:
        print(f"\nSampling {max_trips} trips from {len(unique_trips)}...")
        np.random.seed(42)
        selected = np.random.choice(unique_trips, max_trips, replace=False)
        df = df[df['trip_id'].isin(selected)]
        print(f"  Selected: {df['trip_id'].nunique()} trips")
    
    return df


def extract_trajectories(df):
    """Extract trajectory arrays"""
    print("\nExtracting trajectories...")
    
    trajectories = {}
    for trip_id in tqdm(df['trip_id'].unique(), desc="  Extracting"):
        trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
        coords = trip_data[['latitude', 'longitude']].values
        trajectories[trip_id] = coords
    
    print(f"  ✓ Extracted {len(trajectories)} trajectories")
    return trajectories


def compute_distance_matrix(trajectories):
    """Compute pairwise Hausdorff distances"""
    print("\nComputing distance matrix...")
    
    trip_ids = list(trajectories.keys())
    n = len(trip_ids)
    distance_matrix = np.zeros((n, n))
    
    total_pairs = n * (n - 1) // 2
    
    with tqdm(total=total_pairs, desc="  Distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                traj_i = trajectories[trip_ids[i]]
                traj_j = trajectories[trip_ids[j]]
                
                d1 = directed_hausdorff(traj_i, traj_j)[0]
                d2 = directed_hausdorff(traj_j, traj_i)[0]
                dist = max(d1, d2) * 111000  # degrees to meters
                
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                pbar.update(1)
    
    print(f"  ✓ Distance matrix: {n}x{n}")
    return distance_matrix, trip_ids


def run_dbscan(distance_matrix, eps, min_samples):
    """Run DBSCAN clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    return dbscan.fit_predict(distance_matrix)


def evaluate(distance_matrix, labels):
    """Evaluate clustering"""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)
    
    silhouette = np.nan
    if n_clusters >= 2 and noise_ratio < 0.9:
        mask = labels != -1
        if mask.sum() >= 2:
            try:
                silhouette = silhouette_score(
                    distance_matrix[mask][:, mask],
                    labels[mask],
                    metric='precomputed'
                )
            except:
                pass
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': noise_ratio,
        'silhouette': silhouette
    }


def tune_hyperparameters(distance_matrix):
    """Grid search for best parameters"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    results = []
    best_score = -1
    best_params = None
    
    for eps in EPSILON_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            labels = run_dbscan(distance_matrix, eps, min_samples)
            metrics = evaluate(distance_matrix, labels)
            metrics['epsilon'] = eps
            metrics['min_samples'] = min_samples
            results.append(metrics)
            
            # Print progress
            sil_str = f", Silhouette: {metrics['silhouette']:.3f}" if not np.isnan(metrics['silhouette']) else ""
            print(f"  ε={eps}m, MinPts={min_samples}: {metrics['n_clusters']} clusters, {metrics['noise_ratio']*100:.1f}% noise{sil_str}")
            
            # Track best
            if not np.isnan(metrics['silhouette']) and metrics['silhouette'] > best_score:
                best_score = metrics['silhouette']
                best_params = metrics.copy()
    
    # Fallback if no valid silhouette
    if best_params is None:
        valid = [r for r in results if r['n_clusters'] >= 2]
        if valid:
            best_params = min(valid, key=lambda x: x['noise_ratio'])
        else:
            best_params = results[0]
    
    print(f"\n{'='*50}")
    print(f"Best: ε={best_params['epsilon']}m, MinPts={best_params['min_samples']}")
    print(f"  Clusters: {best_params['n_clusters']}")
    print(f"  Noise: {best_params['noise_ratio']*100:.1f}%")
    if not np.isnan(best_params['silhouette']):
        print(f"  Silhouette: {best_params['silhouette']:.3f}")
    
    return best_params, pd.DataFrame(results)


def analyze_clusters(labels, trip_ids, trajectories, df):
    """Analyze discovered clusters"""
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    trip_to_cluster = dict(zip(trip_ids, labels))
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    
    print(f"\nDiscovered {len(unique_clusters)} route clusters:")
    print("-"*70)
    
    cluster_stats = []
    for cluster_id in unique_clusters[:15]:  # Show top 15
        cluster_trips = [tid for tid, c in trip_to_cluster.items() if c == cluster_id]
        
        # Get representative trajectory
        rep_traj = trajectories[cluster_trips[0]]
        
        # Calculate cluster centroid
        all_points = np.vstack([trajectories[t] for t in cluster_trips])
        centroid = all_points.mean(axis=0)
        
        stats = {
            'cluster_id': cluster_id,
            'num_trips': len(cluster_trips),
            'start_lat': rep_traj[0, 0],
            'start_lon': rep_traj[0, 1],
            'end_lat': rep_traj[-1, 0],
            'end_lon': rep_traj[-1, 1],
            'centroid_lat': centroid[0],
            'centroid_lon': centroid[1]
        }
        cluster_stats.append(stats)
        
        print(f"  Cluster {cluster_id}: {len(cluster_trips)} trips | "
              f"({stats['start_lat']:.4f}, {stats['start_lon']:.4f}) → "
              f"({stats['end_lat']:.4f}, {stats['end_lon']:.4f})")
    
    if len(unique_clusters) > 15:
        print(f"  ... and {len(unique_clusters) - 15} more clusters")
    
    noise_count = list(labels).count(-1)
    print(f"\nNoise: {noise_count} trips ({noise_count/len(labels)*100:.1f}%)")
    
    return pd.DataFrame(cluster_stats), trip_to_cluster


def visualize(trajectories, labels, trip_ids, results_df, output_dir):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trip_to_cluster = dict(zip(trip_ids, labels))
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    
    # 1. Route map
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_clusters))))
    
    # Plot clusters
    for idx, cluster_id in enumerate(unique_clusters[:20]):
        cluster_trips = [tid for tid, c in trip_to_cluster.items() if c == cluster_id]
        for trip_id in cluster_trips[:3]:  # 3 trips per cluster
            traj = trajectories[trip_id]
            ax.plot(traj[:, 1], traj[:, 0], color=colors[idx % 20], alpha=0.6, linewidth=1)
    
    # Plot noise
    noise_trips = [tid for tid, c in trip_to_cluster.items() if c == -1]
    for trip_id in noise_trips[:100]:
        traj = trajectories[trip_id]
        ax.plot(traj[:, 1], traj[:, 0], color='lightgray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Beijing Taxi Routes ({len(unique_clusters)} clusters discovered)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beijing_real_routes_map.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: beijing_real_routes_map.png")
    
    # 2. Cluster size distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cluster_sizes = [len([t for t, c in trip_to_cluster.items() if c == cid]) 
                    for cid in unique_clusters]
    
    ax.hist(cluster_sizes, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(cluster_sizes), color='red', linestyle='--', 
               label=f'Mean: {np.mean(cluster_sizes):.1f}')
    ax.set_xlabel('Trips per Cluster')
    ax.set_ylabel('Frequency')
    ax.set_title('Cluster Size Distribution', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beijing_cluster_sizes.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: beijing_cluster_sizes.png")
    
    # 3. Hyperparameter heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = results_df.pivot(index='epsilon', columns='min_samples', values='n_clusters')
    im = ax.imshow(pivot.values, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(MIN_SAMPLES_VALUES)))
    ax.set_xticklabels(MIN_SAMPLES_VALUES)
    ax.set_yticks(range(len(EPSILON_VALUES)))
    ax.set_yticklabels(EPSILON_VALUES)
    ax.set_xlabel('MinSamples')
    ax.set_ylabel('Epsilon (m)')
    ax.set_title('Number of Clusters by Parameters', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Clusters')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beijing_hyperparameters.png', dpi=150)
    plt.close()
    print(f"  ✓ Saved: beijing_hyperparameters.png")


def save_results(cluster_stats, results_df, best_params, output_dir):
    """Save all results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cluster_stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
    results_df.to_csv(output_dir / 'hyperparameter_results.csv', index=False)
    
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Beijing T-Drive Real Data Clustering Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Routes discovered: {best_params['n_clusters']}\n")
        f.write(f"Noise ratio: {best_params['noise_ratio']*100:.1f}%\n")
        f.write(f"Silhouette score: {best_params['silhouette']:.3f}\n" if not np.isnan(best_params['silhouette']) else "Silhouette score: N/A\n")
        f.write(f"\nBest Parameters:\n")
        f.write(f"  Epsilon: {best_params['epsilon']}m\n")
        f.write(f"  MinSamples: {best_params['min_samples']}\n")
    
    print(f"\n✓ Results saved to {output_dir}")


def main():
    """Main pipeline"""
    
    # Load data
    df = load_data()
    
    # Filter valid trips
    df = filter_valid_trips(df)
    
    # Sample for computation
    df = sample_trips(df)
    
    # Extract trajectories
    trajectories = extract_trajectories(df)
    
    # Compute distance matrix
    distance_matrix, trip_ids = compute_distance_matrix(trajectories)
    
    # Tune hyperparameters
    best_params, results_df = tune_hyperparameters(distance_matrix)
    
    # Final clustering
    print("\n" + "="*70)
    print("FINAL CLUSTERING")
    print("="*70)
    
    labels = run_dbscan(distance_matrix, best_params['epsilon'], best_params['min_samples'])
    
    # Analyze
    cluster_stats, trip_to_cluster = analyze_clusters(labels, trip_ids, trajectories, df)
    
    # Visualize
    visualize(trajectories, labels, trip_ids, results_df, OUTPUT_DIR)
    
    # Save
    save_results(cluster_stats, results_df, best_params, OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*70)
    print("BEIJING REAL DATA CLUSTERING COMPLETE")
    print("="*70)
    print(f"\n🎯 Results:")
    print(f"  Routes discovered: {best_params['n_clusters']}")
    print(f"  Noise ratio: {best_params['noise_ratio']*100:.1f}%")
    if not np.isnan(best_params['silhouette']):
        print(f"  Silhouette score: {best_params['silhouette']:.3f}")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
