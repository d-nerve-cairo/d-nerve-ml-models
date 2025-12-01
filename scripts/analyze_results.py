"""
Quick analysis of DBSCAN route discovery results

Author: Group 2
"""

import pickle
import pandas as pd
from pathlib import Path
import os
import sys

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config

def main():
    config = load_config()
    
    # Load results
    results_file = Path(config['outputs']['route_discovery_dir']) / "route_discovery_results.pkl"
    
    if not results_file.exists():
        print(f" Results file not found: {results_file}")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    routes = results['routes']
    labels = results['labels']
    parameters = results['parameters']
    
    print("="*60)
    print("DBSCAN ROUTE DISCOVERY ANALYSIS")
    print("="*60)
    
    print(f"\nParameters Used:")
    print(f"  Epsilon (ε): {parameters['epsilon']} meters")
    print(f"  Min Samples: {parameters['min_samples']} trips")
    print(f"  Trips Analyzed: {parameters['num_trips_analyzed']}")
    
    print(f"\nClustering Results:")
    print(f"  Routes Discovered: {parameters['num_routes']}")
    noise_count = list(labels).count(-1)
    print(f"  Noise Points: {noise_count} ({noise_count/len(labels)*100:.1f}%)")
    print(f"  Clustered Points: {len(labels) - noise_count} ({(len(labels)-noise_count)/len(labels)*100:.1f}%)")
    
    if len(routes) > 0:
        print(f"\nRoute Statistics:")
        route_sizes = [r['num_trips'] for r in routes.values()]
        print(f"  Largest route: {max(route_sizes)} trips")
        print(f"  Smallest route: {min(route_sizes)} trips")
        print(f"  Average: {sum(route_sizes)/len(route_sizes):.1f} trips per route")
        print(f"  Total trips in routes: {sum(route_sizes)}")
        
        print(f"\nTop 10 Routes by Trip Count:")
        sorted_routes = sorted(routes.items(), key=lambda x: x[1]['num_trips'], reverse=True)
        for i, (route_id, route_data) in enumerate(sorted_routes[:10], 1):
            print(f"  {i:2d}. {route_id}: {route_data['num_trips']:3d} trips")
    
    print("\n" + "="*60)
    print(" ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nVisualizations available in:")
    print(f"  outputs/route_discovery/discovered_routes_all.png")
    print(f"  outputs/route_discovery/route_XXX_detail.png")

if __name__ == "__main__":
    main()

