"""
Beijing T-Drive Route Discovery using DBSCAN
Real-world validation of the D-Nerve clustering pipeline

Since we don't have ground truth labels for real taxi data,
we use internal validation metrics:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Cluster quality analysis

Author: D-Nerve Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "data/beijing_tdrive/processed/beijing_trips.csv"
OUTPUT_DIR = Path("outputs/beijing_clustering")

# DBSCAN parameters to test
EPSILON_VALUES = [200, 300, 400, 500, 600, 800]  # meters
MIN_SAMPLES_VALUES = [2, 3, 4, 5]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_trip_data(input_file):
    """Load preprocessed trip data"""
    print("="*70)
    print("BEIJING T-DRIVE ROUTE DISCOVERY")
    print("="*70)
    
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['timestamp'])
    
    print(f"  GPS points: {len(df):,}")
    print(f"  Unique trips: {df['trip_id'].nunique():,}")
    print(f"  Unique taxis: {df['taxi_id'].nunique():,}")
    
    return df

# ============================================================================
# TRAJECTORY EXTRACTION
# ============================================================================

def extract_trajectories(df):
    """Extract trajectory arrays for each trip"""
    print("\nExtracting trajectories...")
    
    trajectories = {}
    trip_ids = df['trip_id'].unique()
    
    for trip_id in tqdm(trip_ids, desc="  Extracting"):
        trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
        coords = trip_data[['latitude', 'longitude']].values
        trajectories[trip_id] = coords
    
    print(f"  ✓ Extracted {len(trajectories)} trajectories")
    
    return trajectories

# ============================================================================
# DISTANCE COMPUTATION
# ============================================================================

def compute_distance_matrix(trajectories):
    """
    Compute pairwise Hausdorff distances between trajectories
    """
    print("\nComputing distance matrix...")
    
    trip_ids = list(trajectories.keys())
    n = len(trip_ids)
    
    # Initialize distance matrix
    distance_matrix = np.zeros((n, n))
    
    # Compute pairwise distances
    total_pairs = n * (n - 1) // 2
    
    with tqdm(total=total_pairs, desc="  Computing distances") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                traj_i = trajectories[trip_ids[i]]
                traj_j = trajectories[trip_ids[j]]
                
                # Hausdorff distance (symmetric)
                d1 = directed_hausdorff(traj_i, traj_j)[0]
                d2 = directed_hausdorff(traj_j, traj_i)[0]
                dist = max(d1, d2)
                
                # Convert to meters (approximate)
                dist_meters = dist * 111000  # degrees to meters
                
                distance_matrix[i, j] = dist_meters
                distance_matrix[j, i] = dist_meters
                
                pbar.update(1)
    
    print(f"  ✓ Distance matrix complete ({n}x{n})")
    
    return distance_matrix, trip_ids

# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================

def run_dbscan(distance_matrix, epsilon, min_samples):
    """Run DBSCAN with given parameters"""
    
    dbscan = DBSCAN(
        eps=epsilon,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    labels = dbscan.fit_predict(distance_matrix)
    
    return labels


def evaluate_clustering(distance_matrix, labels):
    """
    Evaluate clustering quality using internal metrics
    (No ground truth needed)
    """
    metrics = {}
    
    # Number of clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels)
    
    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise
    metrics['noise_ratio'] = noise_ratio
    
    # Only compute metrics if we have valid clusters
    if n_clusters >= 2 and noise_ratio < 0.9:
        # Filter out noise for metrics
        mask = labels != -1
        if mask.sum() >= 2:
            try:
                # Silhouette Score (-1 to 1, higher is better)
                metrics['silhouette'] = silhouette_score(
                    distance_matrix[mask][:, mask], 
                    labels[mask],
                    metric='precomputed'
                )
            except:
                metrics['silhouette'] = np.nan
            
            try:
                # Calinski-Harabasz Index (higher is better)
                # Need feature matrix, not distance - use MDS approximation
                from sklearn.manifold import MDS
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, max_iter=100)
                coords = mds.fit_transform(distance_matrix[mask][:, mask])
                metrics['calinski_harabasz'] = calinski_harabasz_score(coords, labels[mask])
                metrics['davies_bouldin'] = davies_bouldin_score(coords, labels[mask])
            except:
                metrics['calinski_harabasz'] = np.nan
                metrics['davies_bouldin'] = np.nan
    else:
        metrics['silhouette'] = np.nan
        metrics['calinski_harabasz'] = np.nan
        metrics['davies_bouldin'] = np.nan
    
    return metrics

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_hyperparameters(distance_matrix, trip_ids):
    """
    Grid search for best DBSCAN parameters
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    results = []
    
    for eps in EPSILON_VALUES:
        for min_samples in MIN_SAMPLES_VALUES:
            print(f"\nTesting ε={eps}m, MinPts={min_samples}...")
            
            labels = run_dbscan(distance_matrix, eps, min_samples)
            metrics = evaluate_clustering(distance_matrix, labels)
            
            metrics['epsilon'] = eps
            metrics['min_samples'] = min_samples
            results.append(metrics)
            
            print(f"  Clusters: {metrics['n_clusters']}, Noise: {metrics['noise_ratio']*100:.1f}%", end="")
            if not np.isnan(metrics['silhouette']):
                print(f", Silhouette: {metrics['silhouette']:.3f}")
            else:
                print()
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters (maximize silhouette, reasonable cluster count)
    valid_results = results_df[
        (results_df['silhouette'].notna()) &
        (results_df['n_clusters'] >= 3) &
        (results_df['noise_ratio'] < 0.5)
    ]
    
    if len(valid_results) > 0:
        best_idx = valid_results['silhouette'].idxmax()
        best_params = results_df.loc[best_idx]
    else:
        # Fallback: pick reasonable defaults
        best_params = results_df[results_df['n_clusters'] >= 2].iloc[0] if len(results_df[results_df['n_clusters'] >= 2]) > 0 else results_df.iloc[0]
    
    print("\n" + "-"*50)
    print(f"Best Parameters:")
    print(f"  Epsilon: {best_params['epsilon']}m")
    print(f"  MinSamples: {best_params['min_samples']}")
    print(f"  Clusters: {best_params['n_clusters']}")
    print(f"  Silhouette: {best_params['silhouette']:.3f}" if not np.isnan(best_params['silhouette']) else "  Silhouette: N/A")
    
    return best_params, results_df

# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

def analyze_clusters(df, labels, trip_ids, trajectories):
    """
    Analyze discovered route clusters
    """
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    # Create mapping
    trip_to_cluster = dict(zip(trip_ids, labels))
    df['cluster'] = df['trip_id'].map(trip_to_cluster)
    
    # Cluster statistics
    cluster_stats = []
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    
    print(f"\nDiscovered {len(unique_clusters)} route clusters:")
    print("-"*70)
    
    for cluster_id in unique_clusters[:10]:  # Show top 10
        cluster_trips = [tid for tid, c in trip_to_cluster.items() if c == cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Get representative trajectory (first trip)
        rep_trip = cluster_trips[0]
        rep_traj = trajectories[rep_trip]
        
        stats = {
            'cluster_id': cluster_id,
            'num_trips': len(cluster_trips),
            'total_points': len(cluster_data),
            'start_lat': rep_traj[0, 0],
            'start_lon': rep_traj[0, 1],
            'end_lat': rep_traj[-1, 0],
            'end_lon': rep_traj[-1, 1],
            'avg_points_per_trip': len(cluster_data) / len(cluster_trips)
        }
        cluster_stats.append(stats)
        
        print(f"  Cluster {cluster_id}: {len(cluster_trips)} trips, "
              f"Start: ({stats['start_lat']:.4f}, {stats['start_lon']:.4f}) → "
              f"End: ({stats['end_lat']:.4f}, {stats['end_lon']:.4f})")
    
    if len(unique_clusters) > 10:
        print(f"  ... and {len(unique_clusters) - 10} more clusters")
    
    # Noise analysis
    noise_trips = [tid for tid, c in trip_to_cluster.items() if c == -1]
    print(f"\nNoise points: {len(noise_trips)} trips ({len(noise_trips)/len(trip_ids)*100:.1f}%)")
    
    return pd.DataFrame(cluster_stats), df

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(df, trajectories, labels, trip_ids, results_df, output_dir):
    """Create visualizations"""
    print("\nGenerating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Hyperparameter heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Silhouette heatmap
    ax1 = axes[0]
    pivot = results_df.pivot(index='epsilon', columns='min_samples', values='silhouette')
    im = ax1.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(len(MIN_SAMPLES_VALUES)))
    ax1.set_xticklabels(MIN_SAMPLES_VALUES)
    ax1.set_yticks(range(len(EPSILON_VALUES)))
    ax1.set_yticklabels(EPSILON_VALUES)
    ax1.set_xlabel('MinSamples')
    ax1.set_ylabel('Epsilon (m)')
    ax1.set_title('Silhouette Score by Parameters', fontweight='bold')
    plt.colorbar(im, ax=ax1)
    
    # Cluster count heatmap
    ax2 = axes[1]
    pivot2 = results_df.pivot(index='epsilon', columns='min_samples', values='n_clusters')
    im2 = ax2.imshow(pivot2.values, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(MIN_SAMPLES_VALUES)))
    ax2.set_xticklabels(MIN_SAMPLES_VALUES)
    ax2.set_yticks(range(len(EPSILON_VALUES)))
    ax2.set_yticklabels(EPSILON_VALUES)
    ax2.set_xlabel('MinSamples')
    ax2.set_ylabel('Epsilon (m)')
    ax2.set_title('Number of Clusters by Parameters', fontweight='bold')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: hyperparameter_analysis.png")
    
    # 2. Cluster visualization (map)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique clusters (excluding noise)
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_clusters))))
    
    # Plot each cluster
    trip_to_cluster = dict(zip(trip_ids, labels))
    
    for idx, cluster_id in enumerate(unique_clusters[:20]):  # Plot up to 20 clusters
        cluster_trips = [tid for tid, c in trip_to_cluster.items() if c == cluster_id]
        
        for trip_id in cluster_trips[:5]:  # Plot up to 5 trips per cluster
            traj = trajectories[trip_id]
            ax.plot(traj[:, 1], traj[:, 0], color=colors[idx % 20], alpha=0.5, linewidth=1)
    
    # Plot noise in gray
    noise_trips = [tid for tid, c in trip_to_cluster.items() if c == -1]
    for trip_id in noise_trips[:50]:  # Limit noise display
        traj = trajectories[trip_id]
        ax.plot(traj[:, 1], traj[:, 0], color='lightgray', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Discovered Routes in Beijing ({len(unique_clusters)} clusters)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'beijing_routes_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: beijing_routes_map.png")
    
    # 3. Cluster size distribution
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
    plt.savefig(output_dir / 'cluster_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: cluster_size_distribution.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df, cluster_stats, results_df, best_params, output_dir):
    """Save all results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save clustered data
    df.to_csv(output_dir / 'beijing_clustered_trips.csv', index=False)
    
    # Save cluster statistics
    cluster_stats.to_csv(output_dir / 'cluster_statistics.csv', index=False)
    
    # Save hyperparameter results
    results_df.to_csv(output_dir / 'hyperparameter_results.csv', index=False)
    
    # Save summary
    with open(output_dir / 'clustering_summary.txt', 'w') as f:
        f.write("Beijing T-Drive Clustering Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total trips analyzed: {df['trip_id'].nunique()}\n")
        f.write(f"Routes discovered: {int(best_params['n_clusters'])}\n")
        f.write(f"Noise ratio: {best_params['noise_ratio']*100:.1f}%\n")
        f.write(f"Silhouette score: {best_params['silhouette']:.3f}\n" if not np.isnan(best_params['silhouette']) else "Silhouette score: N/A\n")
        f.write(f"\nBest Parameters:\n")
        f.write(f"  Epsilon: {best_params['epsilon']}m\n")
        f.write(f"  MinSamples: {int(best_params['min_samples'])}\n")
    
    print(f"\n✓ Results saved to {output_dir}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_beijing_clustering():
    """Run complete clustering pipeline"""
    
    # Check if data exists
    if not Path(INPUT_FILE).exists():
        print(f"Data file not found: {INPUT_FILE}")
        print("Please run beijing_preprocessing.py first!")
        return
    
    # Load data
    df = load_trip_data(INPUT_FILE)
    
    # Extract trajectories
    trajectories = extract_trajectories(df)
    
    # Compute distance matrix
    distance_matrix, trip_ids = compute_distance_matrix(trajectories)
    
    # Tune hyperparameters
    best_params, results_df = tune_hyperparameters(distance_matrix, trip_ids)
    
    # Run final clustering with best parameters
    print("\n" + "="*70)
    print("FINAL CLUSTERING")
    print("="*70)
    
    labels = run_dbscan(
        distance_matrix, 
        int(best_params['epsilon']), 
        int(best_params['min_samples'])
    )
    
    # Analyze clusters
    cluster_stats, df = analyze_clusters(df, labels, trip_ids, trajectories)
    
    # Visualize
    visualize_results(df, trajectories, labels, trip_ids, results_df, OUTPUT_DIR)
    
    # Save results
    save_results(df, cluster_stats, results_df, best_params, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("BEIJING T-DRIVE CLUSTERING COMPLETE")
    print("="*70)
    print(f"\nKey Results:")
    print(f"  Routes discovered: {int(best_params['n_clusters'])}")
    print(f"  Silhouette score: {best_params['silhouette']:.3f}" if not np.isnan(best_params['silhouette']) else "  Silhouette score: N/A")
    print(f"  Noise ratio: {best_params['noise_ratio']*100:.1f}%")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return df, labels, cluster_stats


if __name__ == "__main__":
    run_beijing_clustering()
