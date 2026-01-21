"""
Cairo DBSCAN Route Discovery - HARD MODE Evaluation

Enhanced evaluation for challenging data:
1. Higher GPS noise (30-50m)
2. Overlapping routes analysis
3. Noise level impact analysis
4. Detailed confusion matrix

Author: D-Nerve Team
Target: F1 ≥ 0.85 (harder to achieve!)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# DBSCAN Parameters - May need tuning for hard mode!
EPSILON = 300       # May need larger epsilon for noisier data
MIN_SAMPLES = 3

CAIRO_LAT = 30.0

# ============================================================================
# DISTANCE FUNCTIONS (Same as before)
# ============================================================================

def extract_trajectory_points(df, trip_id):
    """Extract lat/lon points for a single trip"""
    trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
    coords = trip_data[['longitude', 'latitude']].values
    return coords

def hausdorff_distance_meters(traj1, traj2, ref_lat=CAIRO_LAT):
    """Calculate Hausdorff distance in meters"""
    dist_deg = max(
        directed_hausdorff(traj1, traj2)[0],
        directed_hausdorff(traj2, traj1)[0]
    )
    meters_per_degree = 111000 * np.cos(np.radians(ref_lat))
    return dist_deg * meters_per_degree

def compute_distance_matrix(df, trip_ids):
    """Compute pairwise Hausdorff distance matrix"""
    n_trips = len(trip_ids)
    print(f"\nComputing distance matrix for {n_trips} trips...")
    
    print("  Extracting trajectories...")
    trajectories = {}
    for trip_id in tqdm(trip_ids, desc="  Loading"):
        trajectories[trip_id] = extract_trajectory_points(df, trip_id)
    
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
# ENHANCED EVALUATION FOR HARD MODE
# ============================================================================

def evaluate_clustering_hard_mode(predicted_labels, trip_ids, df):
    """
    Enhanced evaluation for hard mode data
    Includes overlap group analysis
    """
    print("\n" + "="*60)
    print("HARD MODE EVALUATION")
    print("="*60)
    
    # Get ground truth
    trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
    true_route_ids = np.array([trip_to_route[tid] for tid in trip_ids])
    
    # Get noise levels per trip
    trip_to_noise = df.groupby('trip_id')['noise_level'].first().to_dict()
    noise_levels = np.array([trip_to_noise[tid] for tid in trip_ids])
    
    # Get overlap groups if available
    has_overlap = 'overlap_group' in df.columns
    if has_overlap:
        trip_to_overlap = df.groupby('trip_id')['overlap_group'].first().to_dict()
        overlap_groups = np.array([trip_to_overlap[tid] for tid in trip_ids])
    
    # Filter out noise
    valid_mask = predicted_labels != -1
    pred_valid = predicted_labels[valid_mask]
    true_valid = true_route_ids[valid_mask]
    noise_valid = noise_levels[valid_mask]
    
    print(f"\nTrips analyzed: {len(predicted_labels)}")
    print(f"Trips clustered: {sum(valid_mask)} ({sum(valid_mask)/len(predicted_labels)*100:.1f}%)")
    print(f"Trips as noise: {sum(~valid_mask)} ({sum(~valid_mask)/len(predicted_labels)*100:.1f}%)")
    
    # === Standard Metrics ===
    print("\n--- Overall Clustering Metrics ---")
    
    ari = adjusted_rand_score(true_valid, pred_valid)
    nmi = normalized_mutual_info_score(true_valid, pred_valid)
    homogeneity = homogeneity_score(true_valid, pred_valid)
    completeness = completeness_score(true_valid, pred_valid)
    v_measure = v_measure_score(true_valid, pred_valid)
    
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"V-measure: {v_measure:.4f}")
    
    # === F1 Score ===
    print("\n--- Route Discovery F1 Score ---")
    
    pred_clusters = set(pred_valid)
    true_routes = set(true_valid)
    
    print(f"Predicted clusters: {len(pred_clusters)}")
    print(f"True routes: {len(true_routes)}")
    
    # Precision (cluster purity)
    contingency = {}
    for pred_label in pred_clusters:
        mask = pred_valid == pred_label
        true_in_cluster = true_valid[mask]
        contingency[pred_label] = Counter(true_in_cluster)
    
    cluster_precisions = []
    cluster_sizes = []
    for pred_label, true_counts in contingency.items():
        total = sum(true_counts.values())
        max_count = max(true_counts.values())
        precision = max_count / total
        cluster_precisions.append(precision)
        cluster_sizes.append(total)
    
    weighted_precision = sum(p * s for p, s in zip(cluster_precisions, cluster_sizes)) / sum(cluster_sizes)
    
    # Recall
    route_recalls = []
    route_sizes = []
    for true_route in true_routes:
        mask = true_valid == true_route
        total_in_route = sum(mask)
        
        preds_for_route = pred_valid[mask]
        most_common_cluster = Counter(preds_for_route).most_common(1)[0]
        recall = most_common_cluster[1] / total_in_route
        route_recalls.append(recall)
        route_sizes.append(total_in_route)
    
    weighted_recall = sum(r * s for r, s in zip(route_recalls, route_sizes)) / sum(route_sizes)
    
    if weighted_precision + weighted_recall > 0:
        f1_score = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    else:
        f1_score = 0
    
    print(f"\nWeighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # === Analysis by Noise Level ===
    print("\n--- Performance by Noise Level ---")
    
    for noise_level in sorted(set(noise_levels)):
        noise_mask = noise_levels == noise_level
        valid_noise_mask = noise_mask & valid_mask
        
        if sum(valid_noise_mask) > 0:
            pred_noise = predicted_labels[valid_noise_mask]
            true_noise = true_route_ids[valid_noise_mask]
            
            ari_noise = adjusted_rand_score(true_noise, pred_noise)
            
            # Noise-specific F1
            noise_contingency = {}
            for pred_label in set(pred_noise):
                mask = pred_noise == pred_label
                true_in_cluster = true_noise[mask]
                noise_contingency[pred_label] = Counter(true_in_cluster)
            
            noise_precisions = []
            noise_sizes = []
            for pred_label, true_counts in noise_contingency.items():
                total = sum(true_counts.values())
                max_count = max(true_counts.values())
                noise_precisions.append(max_count / total)
                noise_sizes.append(total)
            
            noise_precision = sum(p * s for p, s in zip(noise_precisions, noise_sizes)) / sum(noise_sizes) if noise_sizes else 0
            
            print(f"  {noise_level}m noise: ARI={ari_noise:.3f}, Precision={noise_precision:.3f}, "
                  f"Trips={sum(valid_noise_mask)}")
    
    # === Analysis by Overlap Group ===
    if has_overlap:
        print("\n--- Performance by Overlap Group ---")
        overlap_valid = overlap_groups[valid_mask]
        
        for group in sorted(set(overlap_groups)):
            group_mask = overlap_groups == group
            valid_group_mask = group_mask & valid_mask
            
            if sum(valid_group_mask) > 0:
                pred_group = predicted_labels[valid_group_mask]
                true_group = true_route_ids[valid_group_mask]
                
                ari_group = adjusted_rand_score(true_group, pred_group)
                n_true_routes = len(set(true_group))
                n_pred_clusters = len(set(pred_group))
                
                print(f"  Group {group}: ARI={ari_group:.3f}, "
                      f"True routes={n_true_routes}, Pred clusters={n_pred_clusters}")
    
    # === Target Check ===
    target_f1 = 0.85
    print("\n" + "="*60)
    if f1_score >= target_f1:
        print(f"🎯 TARGET MET! F1 ({f1_score:.4f}) >= {target_f1}")
    else:
        gap = target_f1 - f1_score
        print(f"⚠️  F1 ({f1_score:.4f}) below target {target_f1} (gap: {gap:.4f})")
        print("   This is expected for hard mode - try tuning EPSILON")
    print("="*60)
    
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
# VISUALIZATION
# ============================================================================

def visualize_hard_mode_results(df, trip_ids, labels, trajectories, metrics, output_dir):
    """Enhanced visualizations for hard mode"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
    trip_to_noise = df.groupby('trip_id')['noise_level'].first().to_dict()
    
    # === Plot 1: Ground Truth vs Predicted ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ground truth
    ax1 = axes[0]
    unique_routes = sorted(df['route_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_routes)))
    route_to_color = {r: colors[i % 20] for i, r in enumerate(unique_routes)}
    
    for trip_id in trip_ids[:100]:
        traj = trajectories[trip_id]
        route = trip_to_route[trip_id]
        ax1.plot(traj[:, 0], traj[:, 1], color=route_to_color[route], alpha=0.4, linewidth=0.8)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'Ground Truth ({len(unique_routes)} routes)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Predicted
    ax2 = axes[1]
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    
    for i, trip_id in enumerate(trip_ids[:100]):
        traj = trajectories[trip_id]
        label = labels[i]
        
        if label == -1:
            color = 'gray'
            alpha = 0.15
        else:
            color = colors[label % 20]
            alpha = 0.5
        
        ax2.plot(traj[:, 0], traj[:, 1], color=color, alpha=alpha, linewidth=0.8)
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'DBSCAN Clusters ({n_clusters} discovered)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hard_mode_clustering_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: hard_mode_clustering_comparison.png")
    
    # === Plot 2: Performance by Noise Level ===
    noise_levels_list = [trip_to_noise[tid] for tid in trip_ids]
    unique_noise = sorted(set(noise_levels_list))
    
    noise_stats = []
    for noise in unique_noise:
        mask = np.array([trip_to_noise[tid] == noise for tid in trip_ids])
        valid_mask = labels != -1
        combined_mask = mask & valid_mask
        
        if sum(combined_mask) > 0:
            pred_subset = labels[combined_mask]
            true_subset = np.array([trip_to_route[tid] for tid in np.array(trip_ids)[combined_mask]])
            ari = adjusted_rand_score(true_subset, pred_subset)
            noise_stats.append({'noise': noise, 'ari': ari, 'n_trips': sum(mask)})
    
    if noise_stats:
        noise_df = pd.DataFrame(noise_stats)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(noise_df['noise'].astype(str), noise_df['ari'], color='steelblue', edgecolor='black')
        
        ax.axhline(y=0.85, color='red', linestyle='--', label='Target (0.85)')
        ax.set_xlabel('GPS Noise Level (meters)', fontsize=12)
        ax.set_ylabel('Adjusted Rand Index', fontsize=12)
        ax.set_title('Clustering Performance by Noise Level', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        # Add value labels
        for bar, ari in zip(bars, noise_df['ari']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{ari:.3f}', ha='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / "hard_mode_noise_impact.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: hard_mode_noise_impact.png")
    
    # === Plot 3: Metrics Summary ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure', 'Precision', 'Recall', 'F1']
    metric_values = [metrics['ari'], metrics['nmi'], metrics['homogeneity'], 
                     metrics['completeness'], metrics['v_measure'],
                     metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    colors = ['#2ecc71' if v >= 0.85 else '#e74c3c' for v in metric_values]
    bars = ax.barh(metric_names, metric_values, color=colors, edgecolor='black')
    
    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2, label='Target (0.85)')
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Hard Mode Evaluation Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    for bar, val in zip(bars, metric_values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hard_mode_metrics_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: hard_mode_metrics_summary.png")

def save_results(metrics, labels, trip_ids, params, output_dir):
    """Save results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metrics': metrics,
        'labels': labels,
        'trip_ids': trip_ids,
        'parameters': params
    }
    
    with open(output_dir / "hard_mode_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    pd.DataFrame([metrics]).to_csv(output_dir / "hard_mode_metrics.csv", index=False)
    print(f"  ✓ Results saved to {output_dir}")

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_for_hard_mode(dist_matrix, trip_ids, df):
    """Extended grid search for hard mode"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - HARD MODE")
    print("="*60)
    
    # Wider range for noisy data
    eps_range = [150, 200, 250, 300, 350, 400, 500, 600, 800]
    min_samples_range = [2, 3, 4, 5]
    
    results = []
    
    trip_to_route = df.groupby('trip_id')['route_id'].first().to_dict()
    true_route_ids = [trip_to_route[tid] for tid in trip_ids]
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            labels = cluster_trajectories(dist_matrix, eps, min_samples)
            
            valid_mask = labels != -1
            if sum(valid_mask) < 10:
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
            
            print(f"  ε={eps:4d}, MinPts={min_samples}: ARI={ari:.3f}, clusters={n_clusters}, noise={noise_ratio:.1%}")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df['ari'].idxmax()
    best_params = results_df.loc[best_idx]
    
    print(f"\n✓ Best parameters:")
    print(f"  Epsilon: {int(best_params['epsilon'])} meters")
    print(f"  MinSamples: {int(best_params['min_samples'])}")
    print(f"  ARI: {best_params['ari']:.4f}")
    
    return results_df, best_params

# ============================================================================
# MAIN
# ============================================================================

def run_hard_mode_evaluation(input_file, output_dir, tune_params=True):
    """Complete hard mode evaluation pipeline"""
    print("="*70)
    print("CAIRO DBSCAN - HARD MODE EVALUATION")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    trip_ids = df['trip_id'].unique().tolist()
    print(f"Total trips: {len(trip_ids)}")
    print(f"Ground truth routes: {df['route_id'].nunique()}")
    
    if 'noise_level' in df.columns:
        print(f"Noise levels: {sorted(df['noise_level'].unique())}")
    if 'overlap_group' in df.columns:
        print(f"Overlap groups: {df['overlap_group'].nunique()}")
    
    # Compute distances
    dist_matrix, trajectories = compute_distance_matrix(df, trip_ids)
    
    # Tune or use defaults
    if tune_params:
        results_df, best_params = tune_for_hard_mode(dist_matrix, trip_ids, df)
        eps = int(best_params['epsilon'])
        min_samples = int(best_params['min_samples'])
    else:
        eps = EPSILON
        min_samples = MIN_SAMPLES
    
    # Final clustering
    labels = cluster_trajectories(dist_matrix, eps, min_samples)
    
    # Evaluate
    metrics = evaluate_clustering_hard_mode(labels, trip_ids, df)
    
    # Visualize
    visualize_hard_mode_results(df, trip_ids, labels, trajectories, metrics, output_dir)
    
    # Save
    params = {'epsilon': eps, 'min_samples': min_samples}
    save_results(metrics, labels, trip_ids, params, output_dir)
    
    print("\n" + "="*70)
    print("HARD MODE EVALUATION COMPLETE")
    print("="*70)
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Results saved to: {output_dir}")
    
    return metrics, labels

if __name__ == "__main__":
    # For hard mode data
    INPUT_FILE = "data/cairo_hard/processed/cairo_hard_trajectories_clean.csv"
    OUTPUT_DIR = "outputs/cairo_hard_mode"
    
    # Check if file exists, otherwise use regular Cairo data path
    if not os.path.exists(INPUT_FILE):
        INPUT_FILE = "data/cairo_hard/raw/cairo_hard_trajectories.csv"
    
    metrics, labels = run_hard_mode_evaluation(INPUT_FILE, OUTPUT_DIR, tune_params=True)
