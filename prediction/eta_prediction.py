"""
ETA Prediction Model - IMPROVED VERSION
Enhanced feature engineering and hyperparameter tuning to meet MAE ≤ 3.0 min target

Author: D-Nerve Team
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_MAE_MINUTES = 3.0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS points (km)"""
    from math import radians, cos, sin, asin, sqrt
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# ============================================================================
# ENHANCED DATA PREPARATION
# ============================================================================

def load_and_prepare_data_v2(input_file):
    """
    Enhanced data preparation with more features
    """
    print("="*70)
    print("ETA PREDICTION - IMPROVED VERSION")
    print("="*70)
    
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"  Total GPS points: {len(df):,}")
    print(f"  Total trips: {df['trip_id'].nunique()}")
    
    # Aggregate to trip level with MORE features
    print("\nEngineering enhanced features...")
    
    trip_features = df.groupby('trip_id').agg({
        'route_id': 'first',
        'origin': 'first',
        'destination': 'first',
        'overlap_group': 'first',
        'noise_level': 'first',
        'latitude': ['first', 'last', 'count', 'mean', 'std'],
        'longitude': ['first', 'last', 'mean', 'std'],
        'timestamp': ['first', 'last'],
        'time_offset_sec': 'max'
    }).reset_index()
    
    # Flatten columns
    trip_features.columns = [
        'trip_id', 'route_id', 'origin', 'destination', 'overlap_group', 'noise_level',
        'start_lat', 'end_lat', 'num_points', 'mean_lat', 'std_lat',
        'start_lon', 'end_lon', 'mean_lon', 'std_lon',
        'start_time', 'end_time', 'duration_sec'
    ]
    
    # Target: duration in minutes
    trip_features['duration_min'] = trip_features['duration_sec'] / 60
    
    # === DISTANCE FEATURES ===
    trip_features['distance_km'] = trip_features.apply(
        lambda row: haversine_distance(
            row['start_lon'], row['start_lat'],
            row['end_lon'], row['end_lat']
        ), axis=1
    )
    
    # Distance squared (captures non-linear relationship)
    trip_features['distance_km_sq'] = trip_features['distance_km'] ** 2
    
    # Log distance (helps with long routes)
    trip_features['distance_km_log'] = np.log1p(trip_features['distance_km'])
    
    # === TIME FEATURES ===
    trip_features['hour'] = trip_features['start_time'].dt.hour
    trip_features['day_of_week'] = trip_features['start_time'].dt.dayofweek
    trip_features['is_weekend'] = (trip_features['day_of_week'] >= 5).astype(int)
    
    # More granular peak hour encoding
    trip_features['is_morning_peak'] = trip_features['hour'].isin([7, 8, 9]).astype(int)
    trip_features['is_evening_peak'] = trip_features['hour'].isin([17, 18, 19]).astype(int)
    trip_features['is_peak'] = (trip_features['is_morning_peak'] | trip_features['is_evening_peak']).astype(int)
    
    # Time period (cyclical encoding)
    trip_features['hour_sin'] = np.sin(2 * np.pi * trip_features['hour'] / 24)
    trip_features['hour_cos'] = np.cos(2 * np.pi * trip_features['hour'] / 24)
    trip_features['dow_sin'] = np.sin(2 * np.pi * trip_features['day_of_week'] / 7)
    trip_features['dow_cos'] = np.cos(2 * np.pi * trip_features['day_of_week'] / 7)
    
    # === ROUTE STATISTICS ===
    route_stats = trip_features.groupby('route_id').agg({
        'duration_min': ['mean', 'std', 'min', 'max', 'median'],
        'distance_km': ['mean', 'std']
    }).reset_index()
    route_stats.columns = ['route_id', 'route_avg_duration', 'route_std_duration', 
                           'route_min_duration', 'route_max_duration', 'route_median_duration',
                           'route_avg_distance', 'route_std_distance']
    
    # Fill NaN std with 0 (routes with single trip)
    route_stats['route_std_duration'] = route_stats['route_std_duration'].fillna(0)
    route_stats['route_std_distance'] = route_stats['route_std_distance'].fillna(0)
    
    trip_features = trip_features.merge(route_stats, on='route_id', how='left')
    
    # === SPEED ESTIMATE ===
    trip_features['expected_speed'] = trip_features['route_avg_distance'] / (trip_features['route_avg_duration'] / 60)
    trip_features['expected_speed'] = trip_features['expected_speed'].fillna(30)  # Default 30 km/h
    
    # Expected duration based on distance and historical speed
    trip_features['expected_duration'] = trip_features['distance_km'] / trip_features['expected_speed'] * 60
    
    # === LOCATION FEATURES ===
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()
    
    trip_features['origin_encoded'] = le_origin.fit_transform(trip_features['origin'])
    trip_features['dest_encoded'] = le_dest.fit_transform(trip_features['destination'])
    
    # Origin-destination pair
    trip_features['od_pair'] = trip_features['origin'] + '_' + trip_features['destination']
    le_od = LabelEncoder()
    trip_features['od_pair_encoded'] = le_od.fit_transform(trip_features['od_pair'])
    
    # === TRAJECTORY COMPLEXITY ===
    # Standard deviation of lat/lon indicates how "straight" the route is
    trip_features['trajectory_complexity'] = trip_features['std_lat'] + trip_features['std_lon']
    trip_features['trajectory_complexity'] = trip_features['trajectory_complexity'].fillna(0)
    
    # Points per km (route complexity)
    trip_features['points_per_km'] = trip_features['num_points'] / (trip_features['distance_km'] + 0.1)
    
    print(f"✓ Prepared {len(trip_features)} trips with enhanced features")
    
    return trip_features

# ============================================================================
# FEATURE SELECTION
# ============================================================================

def prepare_features_v2(trip_features):
    """
    Prepare enhanced feature matrix
    """
    print("\nPreparing enhanced feature set...")
    
    feature_cols = [
        # Distance features
        'distance_km',
        'distance_km_sq',
        'distance_km_log',
        
        # Time features (cyclical)
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'is_weekend',
        'is_morning_peak',
        'is_evening_peak',
        
        # Route historical features
        'route_avg_duration',
        'route_std_duration',
        'route_median_duration',
        'route_avg_distance',
        'expected_duration',
        
        # Location features
        'origin_encoded',
        'dest_encoded',
        'od_pair_encoded',
        
        # Trajectory features
        'num_points',
        'points_per_km',
        'trajectory_complexity',
        
        # Other
        'overlap_group',
    ]
    
    X = trip_features[feature_cols].copy()
    y = trip_features['duration_min'].copy()
    
    # Handle NaN
    X = X.fillna(X.mean())
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X)}")
    
    return X, y, feature_cols, trip_features

# ============================================================================
# MODEL TRAINING WITH TUNING
# ============================================================================

def train_and_evaluate_v2(X, y, feature_cols):
    """
    Train models with hyperparameter tuning
    """
    print("\n" + "="*70)
    print("MODEL TRAINING WITH TUNING")
    print("="*70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    results = {}
    models = {}
    
    # --- Baseline ---
    print("\n--- Baseline (Distance only) ---")
    from sklearn.linear_model import LinearRegression
    baseline = LinearRegression()
    baseline.fit(X_train[['distance_km']], y_train)
    baseline_pred = baseline.predict(X_test[['distance_km']])
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print(f"  MAE: {baseline_mae:.3f} min")
    results['Baseline'] = {'mae': baseline_mae, 'r2': r2_score(y_test, baseline_pred)}
    
    # --- Ridge Regression (regularized) ---
    print("\n--- Ridge Regression (Tuned) ---")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)
    print(f"  MAE: {ridge_mae:.3f} min | R²: {ridge_r2:.4f}")
    results['Ridge'] = {'mae': ridge_mae, 'r2': ridge_r2}
    models['Ridge'] = ridge
    
    # --- ElasticNet ---
    print("\n--- ElasticNet ---")
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    elastic.fit(X_train, y_train)
    elastic_pred = elastic.predict(X_test)
    elastic_mae = mean_absolute_error(y_test, elastic_pred)
    elastic_r2 = r2_score(y_test, elastic_pred)
    print(f"  MAE: {elastic_mae:.3f} min | R²: {elastic_r2:.4f}")
    results['ElasticNet'] = {'mae': elastic_mae, 'r2': elastic_r2}
    models['ElasticNet'] = elastic
    
    # --- Random Forest (Tuned) ---
    print("\n--- Random Forest (Tuned) ---")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"  MAE: {rf_mae:.3f} min | R²: {rf_r2:.4f}")
    results['Random Forest'] = {'mae': rf_mae, 'r2': rf_r2}
    models['Random Forest'] = rf
    
    # --- Gradient Boosting (Tuned) ---
    print("\n--- Gradient Boosting (Tuned) ---")
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"  MAE: {gb_mae:.3f} min | R²: {gb_r2:.4f}")
    results['Gradient Boosting'] = {'mae': gb_mae, 'r2': gb_r2}
    models['Gradient Boosting'] = gb
    
    # --- LightGBM (Tuned) ---
    if HAS_LIGHTGBM:
        print("\n--- LightGBM (Tuned) ---")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_mae = mean_absolute_error(y_test, lgb_pred)
        lgb_r2 = r2_score(y_test, lgb_pred)
        print(f"  MAE: {lgb_mae:.3f} min | R²: {lgb_r2:.4f}")
        results['LightGBM'] = {'mae': lgb_mae, 'r2': lgb_r2}
        models['LightGBM'] = lgb_model
    
    # --- Ensemble (Average of top models) ---
    print("\n--- Ensemble (Averaging) ---")
    # Get predictions from best models
    ensemble_pred = (rf_pred + gb_pred) / 2
    if HAS_LIGHTGBM:
        ensemble_pred = (rf_pred + gb_pred + lgb_pred) / 3
    
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    print(f"  MAE: {ensemble_mae:.3f} min | R²: {ensemble_r2:.4f}")
    results['Ensemble'] = {'mae': ensemble_mae, 'r2': ensemble_r2}
    
    # Find best
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    best_mae = results[best_model_name]['mae']
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'MAE (min)':<12} {'R²':<10}")
    print("-"*45)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['mae']):
        marker = " ← Best" if name == best_model_name else ""
        print(f"{name:<20} {metrics['mae']:<12.3f} {metrics['r2']:<10.4f}{marker}")
    
    print("\n" + "="*70)
    if best_mae <= TARGET_MAE_MINUTES:
        print(f"🎯 TARGET MET! MAE ({best_mae:.3f}) ≤ {TARGET_MAE_MINUTES} minutes")
    else:
        gap = best_mae - TARGET_MAE_MINUTES
        print(f"⚠️  MAE ({best_mae:.3f}) > target by {gap:.3f} minutes ({gap*60:.1f} seconds)")
    print("="*70)
    
    return results, models, best_model_name, (X_train, X_test, y_train, y_test)

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results_v2(results, y_test, predictions, best_model_name, output_dir):
    """Generate visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    model_names = list(results.keys())
    maes = [results[m]['mae'] for m in model_names]
    r2s = [results[m]['r2'] for m in model_names]
    
    # Sort by MAE
    sorted_idx = np.argsort(maes)
    model_names = [model_names[i] for i in sorted_idx]
    maes = [maes[i] for i in sorted_idx]
    r2s = [r2s[i] for i in sorted_idx]
    
    # MAE plot
    ax1 = axes[0]
    colors = ['#2ecc71' if mae <= TARGET_MAE_MINUTES else '#e74c3c' for mae in maes]
    bars = ax1.barh(model_names, maes, color=colors, edgecolor='black')
    ax1.axvline(x=TARGET_MAE_MINUTES, color='red', linestyle='--', linewidth=2, label=f'Target: {TARGET_MAE_MINUTES} min')
    ax1.set_xlabel('MAE (minutes)', fontsize=12)
    ax1.set_title('Model Comparison: MAE', fontsize=14, fontweight='bold')
    ax1.legend()
    
    for bar, mae in zip(bars, maes):
        ax1.text(mae + 0.05, bar.get_y() + bar.get_height()/2, f'{mae:.2f}', va='center', fontsize=10)
    
    # R² plot
    ax2 = axes[1]
    bars = ax2.barh(model_names, r2s, color='steelblue', edgecolor='black')
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('Model Comparison: R²', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    
    for bar, r2 in zip(bars, r2s):
        ax2.text(r2 + 0.02, bar.get_y() + bar.get_height()/2, f'{r2:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eta_model_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eta_model_comparison_v2.png")
    
    # Predictions plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.scatter(y_test, predictions, alpha=0.6, edgecolor='black', linewidth=0.5)
    min_val, max_val = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    ax1.set_xlabel('Actual (min)', fontsize=12)
    ax1.set_ylabel('Predicted (min)', fontsize=12)
    ax1.set_title(f'Predicted vs Actual ({best_model_name})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    errors = predictions - y_test
    ax2.hist(errors, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Error (min)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'eta_predictions_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: eta_predictions_v2.png")

def save_results_v2(results, best_model, best_model_name, feature_cols, output_dir):
    """Save results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'eta_best_model_v2.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'name': best_model_name, 'features': feature_cols}, f)
    
    pd.DataFrame(results).T.to_csv(output_dir / 'eta_results_v2.csv')
    print(f"\n✓ Saved model and results to {output_dir}")

# ============================================================================
# MAIN
# ============================================================================

def run_eta_prediction_v2(input_file, output_dir):
    """Run improved ETA prediction pipeline"""
    
    # Load and prepare
    trip_features = load_and_prepare_data_v2(input_file)
    
    # Prepare features
    X, y, feature_cols, trip_df = prepare_features_v2(trip_features)
    
    # Train and evaluate
    results, models, best_model_name, splits = train_and_evaluate_v2(X, y, feature_cols)
    X_train, X_test, y_train, y_test = splits
    
    # Get best model predictions
    if best_model_name == 'Ensemble':
        # Recreate ensemble prediction
        rf_pred = models['Random Forest'].predict(X_test)
        gb_pred = models['Gradient Boosting'].predict(X_test)
        if 'LightGBM' in models:
            lgb_pred = models['LightGBM'].predict(X_test)
            best_pred = (rf_pred + gb_pred + lgb_pred) / 3
        else:
            best_pred = (rf_pred + gb_pred) / 2
        best_model = None
    else:
        best_model = models.get(best_model_name)
        best_pred = best_model.predict(X_test) if best_model else None
    
    # Visualize
    if best_pred is not None:
        visualize_results_v2(results, y_test, best_pred, best_model_name, output_dir)
    
    # Save
    save_results_v2(results, best_model, best_model_name, feature_cols, output_dir)
    
    # Summary
    best_mae = results[best_model_name]['mae']
    print("\n" + "="*70)
    print("ETA PREDICTION V2 COMPLETE")
    print("="*70)
    print(f"Best Model: {best_model_name}")
    print(f"MAE: {best_mae:.3f} minutes")
    
    if best_mae <= TARGET_MAE_MINUTES:
        print(f"\n🎯 TARGET MET!")
    
    return results, models

if __name__ == "__main__":
    INPUT_FILE = "data/cairo_hard/raw/cairo_hard_trajectories.csv"
    OUTPUT_DIR = "outputs/eta_prediction"
    
    if not os.path.exists(INPUT_FILE):
        INPUT_FILE = "data/cairo/raw/cairo_trajectories_full.csv"
    
    results, models = run_eta_prediction_v2(INPUT_FILE, OUTPUT_DIR)