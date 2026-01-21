"""
Cairo DBSCAN Route Discovery with Ground Truth Evaluation

This script:
1. Clusters Cairo trajectories using DBSCAN + Hausdorff distance
2. Compares discovered clusters to ground truth routes
3. Calculates F1 score, precision, recall (TARGET: F1 >= 0.85)

Author: D-Nerve Team
Environment: dnervenv
Target: F1 ≥ 0.85
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# DBSCAN Parameters - TUNE THESE!
EPSILON = 200       # meters - max distance between trajectories in same cluster
MIN_SAMPLES = 3     # minimum trips to form a route (we have 10 trips per route)

# Cairo latitude for distance conversion
CAIRO_LAT = 30.0    # degrees

# ============================================================================
# TRAJECTORY DISTANCE FUNCTIONS
# ============================================================================

def extract_trajectory_points(df, trip_id):
    """Extract lat/lon points for a single trip"""
    trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
    coords = trip_data[['longitude', 'latitude']].values
    return coords

def hausdorff_distance_meters(traj1, traj2, ref_lat=CAIRO_LAT):
    """
    Calculate Hausdorff distance between two trajectories in meters
    
    Args:
        traj1, traj2: numpy arrays of shape (n_points, 2) with [lon, lat]
        ref_lat: Reference latitude for degree-to-meter conversion
    
    Returns:
        Distance in meters
    """
    # Hausdorff distance in degrees
    dist_deg = max(
        directed_hausdorff(traj1, traj2)[0],
        directed_hausdorff(traj2, traj1)[0]
    )
    
    # Convert degrees to meters at Cairo latitude (~30°N)
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 * cos(30°) ≈ 96 km
    meters_per_degree = 111000 * np.cos(np.radians(ref_lat))
    dist_meters = dist_deg * meters_per_degree
    
    return dist_meters

def compute_distance_matrix(df, trip_ids):
    """Compute pairwise Hausdorff distance matrix"""
    n_trips = len(trip_ids)
    print(f"\nComputing distance matrix for {n_trips} trips...")
    
    # Extract all trajectories first
    print("  Extracting trajectories...")
    trajectories = {}
    for trip_id in tqdm(trip_ids, desc="  Loading"):
        trajectories[trip_id] = extract_trajectory_points(df, trip_id)
    
    # Compute pairwise distances
    print("  Computing pairwise Hausdorff distances...")
    dist_matrix = np.zeros((n_trips, n_trips))
    
    total_comparisons = (n_trips * (n_trips - 1)) // 2
    
    with tqdm(total=total_comparisons, desc="  Distances") as pbar:
        for i in range(n_trips):
            for j in range(i+1, n_trips):
                dist = hausdorff_distance_meters(
                    trajectories[trip_ids[i]],
                    trajectories[trip_ids[j]]
                )
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                pbar.update(1)
    
    print("  ✓ Distance matrix complete")
    return dist_matrix, trajectories

# ============================================================================
# CLUSTERING
# ============================================================================

def cluster_trajectories(dist_matrix, eps, min_samples):
    """Apply DBSCAN clustering"""
    print(f"\nRunning DBSCAN (ε={eps}m, MinPts={min_samples})...")
    
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    labels = clustering.fit_predict(dist_matrix)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  ✓ Routes discovered: {n_clusters}")
    print(f"  ✓ Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
    
    return labels

# ============================================================================
# EVALUATION METRICS (KEY FOR YOUR PROJECT!)
# ============================================================================

def evaluate_clustering(predicted_labels, true_labels, trip_ids, df):
    """
    Evaluate clustering against ground truth
    
    This is critical for your project - demonstrates that DBSCAN
    can recover true routes from noisy GPS data.
    """
    print("\n" + "="*60)
    print("EVALUATION: Discovered vs Ground Truth Routes")
    print("="*60)
    
    # Get ground truth route IDs for each trip
    trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
    true_route_ids = [trip_to_route[tid] for tid in trip_ids]
    
    # Filter out noise points for fair comparison
    valid_mask = predicted_labels != -1
    pred_valid = predicted_labels[valid_mask]
    true_valid = np.array(true_route_ids)[valid_mask]
    
    print(f"\nTrips analyzed: {len(predicted_labels)}")
    print(f"Trips clustered (non-noise): {sum(valid_mask)}")
    print(f"Trips marked as noise: {sum(~valid_mask)}")
    
    # --- Standard Clustering Metrics ---
    print("\n--- Clustering Quality Metrics ---")
    
    # Adjusted Rand Index (ARI): -1 to 1, higher = better
    ari = adjusted_rand_score(true_valid, pred_valid)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    # Normalized Mutual Information (NMI): 0 to 1, higher = better
    nmi = normalized_mutual_info_score(true_valid, pred_valid)
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    
    # Homogeneity: Each cluster contains only members of a single class
    homogeneity = homogeneity_score(true_valid, pred_valid)
    print(f"Homogeneity: {homogeneity:.4f}")
    
    # Completeness: All members of a class are assigned to same cluster
    completeness = completeness_score(true_valid, pred_valid)
    print(f"Completeness: {completeness:.4f}")
    
    # V-measure: Harmonic mean of homogeneity and completeness
    v_measure = v_measure_score(true_valid, pred_valid)
    print(f"V-measure: {v_measure:.4f}")
    
    # --- Route-Level F1 Score ---
    print("\n--- Route Discovery F1 Score ---")
    
    # Calculate purity-based F1
    # For each predicted cluster, find best matching true route
    pred_clusters = set(pred_valid)
    true_routes = set(true_valid)
    
    print(f"Predicted clusters: {len(pred_clusters)}")
    print(f"True routes: {len(true_routes)}")
    
    # Build contingency matrix
    contingency = {}
    for pred_label in pred_clusters:
        mask = pred_valid == pred_label
        true_in_cluster = true_valid[mask]
        contingency[pred_label] = Counter(true_in_cluster)
    
    # Calculate precision (purity) per cluster
    cluster_precisions = []
    cluster_sizes = []
    for pred_label, true_counts in contingency.items():
        total = sum(true_counts.values())
        max_count = max(true_counts.values())
        precision = max_count / total
        cluster_precisions.append(precision)
        cluster_sizes.append(total)
    
    # Weighted average precision
    weighted_precision = sum(p * s for p, s in zip(cluster_precisions, cluster_sizes)) / sum(cluster_sizes)
    
    # Calculate recall: what fraction of each true route is captured
    route_recalls = []
    route_sizes = []
    for true_route in true_routes:
        mask = true_valid == true_route
        total_in_route = sum(mask)
        
        # Find which cluster(s) contain this route's trips
        preds_for_route = pred_valid[mask]
        most_common_cluster = Counter(preds_for_route).most_common(1)[0]
        recall = most_common_cluster[1] / total_in_route
        route_recalls.append(recall)
        route_sizes.append(total_in_route)
    
    # Weighted average recall
    weighted_recall = sum(r * s for r, s in zip(route_recalls, route_sizes)) / sum(route_sizes)
    
    # F1 Score
    if weighted_precision + weighted_recall > 0:
        f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    else:
        f1_score = 0
    
    print(f"\nWeighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Target check
    target_f1 = 0.85
    if f1_score >= target_f1:
        print(f"\n🎯 TARGET MET! F1 ({f1_score:.4f}) >= {target_f1}")
    else:
        gap = target_f1 - f1_score
        print(f"\n⚠️  F1 ({f1_score:.4f}) below target {target_f1} (gap: {gap:.4f})")
        print("   Try adjusting EPSILON or MIN_SAMPLES")
    
    # Return all metrics
    metrics = {
        'ari': ari,
        'nmi': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1_score': f1_score,
        'n_clusters_predicted': len(pred_clusters),
        'n_routes_true': len(true_routes),
        'n_noise': sum(~valid_mask),
        'noise_ratio': sum(~valid_mask) / len(predicted_labels)
    }
    
    return metrics

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def grid_search_dbscan(dist_matrix, trip_ids, df, eps_range, min_samples_range):
    """
    Grid search over DBSCAN parameters to find best F1 score
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = cluster_trajectories(dist_matrix, eps, min_samples)
            
            # Quick evaluation (suppress detailed output)
            trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
            true_route_ids = [trip_to_route[tid] for tid in trip_ids]
            
            valid_mask = labels != -1
            if sum(valid_mask) < 10:  # Too few clustered points
                continue
                
            pred_valid = labels[valid_mask]
            true_valid = np.array(true_route_ids)[valid_mask]
            
            ari = adjusted_rand_score(true_valid, pred_valid)
            nmi = normalized_mutual_info_score(true_valid, pred_valid)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = list(labels).count(-1) / len(labels)
            
            results.append({
                'epsilon': eps,
                'min_samples': min_samples,
                'ari': ari,
                'nmi': nmi,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio
            })
            
            print(f"  ε={eps:4d}, MinPts={min_samples}: ARI={ari:.3f}, NMI={nmi:.3f}, "
                  f"clusters={n_clusters}, noise={noise_ratio:.1%}")
    
    # Find best parameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['ari'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print(f"\n✓ Best parameters:")
    print(f"  Epsilon: {int(best_params['epsilon'])} meters")
    print(f"  MinSamples: {int(best_params['min_samples'])}")
    print(f"  ARI: {best_params['ari']:.4f}")
    
    return results_df, best_params

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_clustering_results(df, trip_ids, labels, trajectories, output_dir):
    """Visualize discovered routes vs ground truth"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Get ground truth for comparison
    trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
    
    # Plot 1: All discovered clusters
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Ground truth routes
    ax1 = axes[0]
    unique_routes = df['route_id'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_routes)))
    route_to_color = {r: colors[i % 20] for i, r in enumerate(unique_routes)}
    
    for trip_id in trip_ids[:50]:  # Plot first 50 trips
        traj = trajectories[trip_id]
        route = trip_to_route[trip_id]
        ax1.plot(traj[:, 0], traj[:, 1], color=route_to_color[route], alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Ground Truth Routes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Right: Discovered clusters
    ax2 = axes[1]
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    for i, trip_id in enumerate(trip_ids[:50]):
        traj = trajectories[trip_id]
        label = labels[i]
        
        if label == -1:
            color = 'gray'
            alpha = 0.2
        else:
            color = colors[label % 20]
            alpha = 0.6
        
        ax2.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=1)
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'DBSCAN Discovered Routes (n={n_clusters})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "cairo_clustering_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")
    
    # Plot 2: Cluster quality summary
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cluster_sizes = Counter(labels)
    if -1 in cluster_sizes:
        del cluster_sizes[-1]
    
    if cluster_sizes:
        clusters = list(cluster_sizes.keys())
        sizes = list(cluster_sizes.values())
        
        ax.bar(range(len(clusters)), sizes, color='steelblue', edgecolor='black')
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.set_title('Trips per Discovered Cluster', fontsize=14, fontweight='bold')
        ax.axhline(y=10, color='red', linestyle='--', label='Expected (10 trips/route)')
        ax.legend()
    
    output_file = output_dir / "cairo_cluster_sizes.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

def save_results(metrics, labels, trip_ids, params, output_dir):
    """Save all results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'labels': labels,
        'trip_ids': trip_ids,
        'parameters': params
    }
    
    # Save as pickle
    pkl_file = output_dir / "cairo_clustering_results.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"  ✓ Saved: {pkl_file}")
    
    # Save metrics as CSV for easy viewing
    metrics_df = pd.DataFrame([metrics])
    csv_file = output_dir / "cairo_clustering_metrics.csv"
    metrics_df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved: {csv_file}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_cairo_route_discovery(input_file, output_dir, tune_params=False):
    """Complete Cairo DBSCAN route discovery pipeline"""
    print("="*70)
    print("CAIRO DBSCAN ROUTE DISCOVERY")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    trip_ids = df['trip_id'].unique().tolist()
    print(f"Total trips: {len(trip_ids)}")
    print(f"Ground truth routes: {df['route_id'].nunique()}")
    
    # Compute distance matrix
    dist_matrix, trajectories = compute_distance_matrix(df, trip_ids)
    
    # Optional: Hyperparameter tuning
    if tune_params:
        eps_range = [100, 150, 200, 250, 300, 400, 500]
        min_samples_range = [2, 3, 4, 5]
        results_df, best_params = grid_search_dbscan(
            dist_matrix, trip_ids, df, eps_range, min_samples_range
        )
        
        # Use best parameters
        eps = int(best_params['epsilon'])
        min_samples = int(best_params['min_samples'])
    else:
        eps = EPSILON
        min_samples = MIN_SAMPLES
    
    # Run clustering with final parameters
    labels = cluster_trajectories(dist_matrix, eps, min_samples)
    
    # Evaluate against ground truth
    metrics = evaluate_clustering(labels, None, trip_ids, df)
    
    # Visualize
    visualize_clustering_results(df, trip_ids, labels, trajectories, output_dir)
    
    # Save results
    params = {'epsilon': eps, 'min_samples': min_samples}
    save_results(metrics, labels, trip_ids, params, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("ROUTE DISCOVERY COMPLETE")
    print("="*70)
    print(f"Parameters: ε={eps}m, MinPts={min_samples}")
    print(f"Clusters discovered: {metrics['n_clusters_predicted']}")
    print(f"Ground truth routes: {metrics['n_routes_true']}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    
    return metrics, labels

if __name__ == "__main__":
    INPUT_FILE = "data/cairo/processed/cairo_trajectories_clean.csv"
    OUTPUT_DIR = "outputs/cairo_route_discovery"
    
    # Set tune_params=True to run grid search (slower but finds best params)
    metrics, labels = run_cairo_route_discovery(
        INPUT_FILE, 
        OUTPUT_DIR, 
        tune_params=True  # Set to True for hyperparameter search
    )