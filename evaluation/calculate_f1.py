"""
Calculate F1 score for route discovery evaluation

Compares discovered routes against ground truth (manual labeling)

Author: Group 2 - ML Team
Environment: dnervenv
Platform: Windows
Target: F1 ≥ 0.85
"""

import pickle
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict

# Fix imports on Windows
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config

def load_results(results_file):
    """Load DBSCAN clustering results"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results

def create_ground_truth_labels(trip_ids, df):
    """
    Create ground truth route labels
    
    For T-Drive dataset, we'll use a simple heuristic:
    - Trips with similar start/end points are same route
    - Grid-based approach for simplicity
    
    In real Cairo data, you'd manually label routes
    """
    print("\n  Creating ground truth labels...")
    print("(Using spatial clustering as proxy for ground truth)")
    
    # Extract start and end points for each trip
    trip_endpoints = {}
    
    for trip_id in trip_ids:
        trip_data = df[df['trip_id'] == trip_id].sort_values('timestamp')
        
        if len(trip_data) < 2:
            continue
        
        start_lon = trip_data.iloc[0]['longitude']
        start_lat = trip_data.iloc[0]['latitude']
        end_lon = trip_data.iloc[-1]['longitude']
        end_lat = trip_data.iloc[-1]['latitude']
        
        trip_endpoints[trip_id] = {
            'start': (start_lon, start_lat),
            'end': (end_lon, end_lat)
        }
    
    # Simple grid-based clustering (0.01 degree ~1km resolution)
    GRID_SIZE = 0.01
    
    def get_grid_cell(lon, lat):
        return (int(lon / GRID_SIZE), int(lat / GRID_SIZE))
    
    # Assign ground truth labels based on start-end grid cells
    ground_truth = {}
    route_counter = 0
    cell_to_route = {}
    
    for trip_id, endpoints in trip_endpoints.items():
        start_cell = get_grid_cell(*endpoints['start'])
        end_cell = get_grid_cell(*endpoints['end'])
        cell_pair = (start_cell, end_cell)
        
        if cell_pair not in cell_to_route:
            cell_to_route[cell_pair] = route_counter
            route_counter += 1
        
        ground_truth[trip_id] = cell_to_route[cell_pair]
    
    print(f" Ground truth created: {route_counter} unique route patterns")
    
    return ground_truth

def calculate_f1_score(predicted_labels, ground_truth, trip_ids):
    """
    Calculate F1 score for route clustering
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    print("\n Calculating F1 Score...")
    
    # Create mapping: trip_id -> predicted cluster
    predicted = {}
    for i, trip_id in enumerate(trip_ids):
        predicted[trip_id] = predicted_labels[i]
    
    # Only evaluate trips that have both predicted and ground truth labels
    common_trips = set(predicted.keys()) & set(ground_truth.keys())
    
    if len(common_trips) == 0:
        print(" No common trips found!")
        return 0.0
    
    print(f"Evaluating {len(common_trips)} trips")
    
    # Calculate pairwise precision and recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    trip_list = list(common_trips)
    
    for i in range(len(trip_list)):
        for j in range(i+1, len(trip_list)):
            trip_i = trip_list[i]
            trip_j = trip_list[j]
            
            # Ground truth: are they in same route?
            same_route_gt = (ground_truth[trip_i] == ground_truth[trip_j])
            
            # Predicted: are they in same cluster? (exclude noise -1)
            same_cluster_pred = (
                predicted[trip_i] == predicted[trip_j] and 
                predicted[trip_i] != -1
            )
            
            if same_route_gt and same_cluster_pred:
                true_positives += 1
            elif not same_route_gt and same_cluster_pred:
                false_positives += 1
            elif same_route_gt and not same_cluster_pred:
                false_negatives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"True Positives:  {true_positives:,}")
    print(f"False Positives: {false_positives:,}")
    print(f"False Negatives: {false_negatives:,}")
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"\n{'='*60}")
    
    # Check if meets target
    TARGET_F1 = 0.85
    if f1 >= TARGET_F1:
        print(f" TARGET ACHIEVED! F1 = {f1:.4f} ≥ {TARGET_F1}")
    else:
        print(f"\n  Below target: F1 = {f1:.4f} < {TARGET_F1}")
        print(f"   Gap: {(TARGET_F1 - f1)*100:.2f}%")
        print(f"\nSuggestions to improve:")
        print(f"  1. Tune EPSILON (try {300 if precision < recall else 200}m)")
        print(f"  2. Tune MIN_SAMPLES (try {3 if precision > recall else 7})")
        print(f"  3. Use more sophisticated distance metric")
    
    print(f"{'='*60}\n")
    
    return f1

def main():
    """Main evaluation function"""
    print("="*60)
    print("ROUTE DISCOVERY EVALUATION")
    print("="*60)
    
    config = load_config()
    
    # Load DBSCAN results
    results_file = Path(config['outputs']['route_discovery_dir']) / "route_discovery_results.pkl"
    
    if not results_file.exists():
        print(f"\n Results file not found: {results_file}")
        print("Run DBSCAN first: python clustering\\dbscan_routes.py")
        return
    
    print(f"\n Loading results from {results_file}...")
    results = load_results(results_file)
    
    predicted_labels = results['labels']
    trip_ids = results['trip_ids']
    parameters = results['parameters']
    
    print(f" Loaded {len(trip_ids)} trip predictions")
    print(f"Parameters: ε={parameters['epsilon']}m, MinPts={parameters['min_samples']}")
    
    # Load trip data
    trips_file = Path(config['data']['processed_dir']) / "tdrive_100taxis_trips.parquet"
    print(f"\n Loading trip data from {trips_file}...")
    df = pd.read_parquet(trips_file)
    
    # Create ground truth
    ground_truth = create_ground_truth_labels(trip_ids, df)
    
    # Calculate F1 score
    f1_score = calculate_f1_score(predicted_labels, ground_truth, trip_ids)
    
    # Save evaluation results
    output_file = Path(config['outputs']['route_discovery_dir']) / "evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"DBSCAN Route Discovery Evaluation\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Epsilon: {parameters['epsilon']}m\n")
        f.write(f"  Min Samples: {parameters['min_samples']}\n")
        f.write(f"  Trips Analyzed: {len(trip_ids)}\n\n")
        f.write(f"Results:\n")
        f.write(f"  F1 Score: {f1_score:.4f}\n")
        f.write(f"  Routes Discovered: {parameters['num_routes']}\n")
    
    print(f" Evaluation saved to {output_file}")
    
    print("\n EVALUATION COMPLETE\n")
    
    return f1_score

if __name__ == "__main__":
    main()