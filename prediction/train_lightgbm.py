"""
Train LightGBM model for ETA prediction

Predicts trip duration (minutes) based on trip features

Author: Group 2 - ML Team
Environment: dnervenv
Target: MAE ≤ 3.0 minutes
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from preprocessing.utils import load_config, create_output_dir

def load_features(features_file):
    """Load engineered features"""
    print(f"\n Loading features from {features_file}...")
    df = pd.read_parquet(features_file)
    print(f" Loaded {len(df)} trips with {len(df.columns)} features")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare train/test split"""
    print(f"\n Preparing train/test split ({int((1-test_size)*100)}%/{int(test_size*100)})...")
    
    # Separate features and target
    feature_cols = ['distance_km', 'num_points', 'start_lon', 'start_lat', 
                    'end_lon', 'end_lat', 'hour', 'day_of_week', 
                    'is_weekend', 'is_rush_hour', 'avg_speed_kph', 
                    'is_on_route', 'route_popularity']
    
    X = df[feature_cols]
    y = df['duration_minutes']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f" Training set: {len(X_train)} trips")
    print(f" Test set: {len(X_test)} trips")
    print(f" Features: {feature_cols}")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_model(X_train, y_train, X_test, y_test):
    """Train LightGBM regression model"""
    print(f"\n Training LightGBM model...")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,  # Changed from 0 to -1
        'random_state': 42
    }
    
    # Train with early stopping
    print("Training in progress...")
    
    # Create callback list
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    # Train model (fixed API call)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=callbacks
    )
    
    # Get evaluation results from model history
    print(f"\n Training complete!")
    print(f"   Best iteration: {model.best_iteration}")
    
    # Calculate final metrics
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"   Training MAE: {train_mae:.2f} minutes")
    print(f"   Test MAE: {test_mae:.2f} minutes")
    
    # Create evals_result for compatibility
    evals_result = {
        'train': {'l1': [train_mae]},
        'test': {'l1': [test_mae]}
    }
    
    return model, evals_result

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print(f"\n Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE):  {mae:.2f} minutes")
    print(f"Root Mean Squared Error:     {rmse:.2f} minutes")
    print(f"R² Score:                    {r2:.4f}")
    print(f"Mean Absolute % Error:       {mape:.2f}%")
    print(f"{'='*60}")
    
    # Check if meets target
    TARGET_MAE = 3.0
    if mae <= TARGET_MAE:
        print(f"\n TARGET ACHIEVED! MAE = {mae:.2f} ≤ {TARGET_MAE} minutes")
    else:
        print(f"\n  Above target: MAE = {mae:.2f} > {TARGET_MAE} minutes")
        print(f"   Gap: {mae - TARGET_MAE:.2f} minutes")
        print(f"\nSuggestions to improve:")
        print(f"  1. Add more features (weather, traffic conditions)")
        print(f"  2. Use historical speed data")
        print(f"  3. Tune hyperparameters with Optuna")
        print(f"  4. Ensemble with other models")
    
    print(f"{'='*60}\n")
    
    return mae, rmse, r2, y_pred

def plot_results(y_test, y_pred, output_dir):
    """Create visualizations"""
    print(f"\n Creating visualizations...")
    
    output_dir = Path(output_dir)
    create_output_dir(output_dir)
    
    # 1. Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Duration (minutes)', fontsize=12)
    plt.ylabel('Predicted Duration (minutes)', fontsize=12)
    plt.title('ETA Prediction: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_predicted.png', dpi=150)
    print(f" Saved: {output_dir / 'actual_vs_predicted.png'}")
    plt.close()
    
    # 2. Error distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    plt.xlabel('Prediction Error (minutes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('ETA Prediction Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=150)
    print(f" Saved: {output_dir / 'error_distribution.png'}")
    plt.close()
    
    # 3. Residuals plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, errors, alpha=0.5, s=20)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Duration (minutes)', fontsize=12)
    plt.ylabel('Residuals (minutes)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150)
    print(f" Saved: {output_dir / 'residuals.png'}")
    plt.close()

def plot_feature_importance(model, feature_cols, output_dir):
    """Plot feature importance"""
    print(f"\n Creating feature importance plot...")
    
    output_dir = Path(output_dir)
    
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance for ETA Prediction', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    print(f" Saved: {output_dir / 'feature_importance.png'}")
    plt.close()
    
    print(f"\nTop 5 Most Important Features:")
    top_features = feature_importance.tail(5)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")

def save_model(model, output_dir, metadata):
    """Save trained model"""
    output_dir = Path(output_dir)
    create_output_dir(output_dir)
    
    # Save model
    model_file = output_dir / 'lightgbm_eta_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n Model saved to {model_file}")
    
    # Save metadata
    metadata_file = output_dir / 'model_metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write("LightGBM ETA Prediction Model\n")
        f.write("="*60 + "\n\n")
        f.write(f"MAE: {metadata['mae']:.2f} minutes\n")
        f.write(f"RMSE: {metadata['rmse']:.2f} minutes\n")
        f.write(f"R² Score: {metadata['r2']:.4f}\n")
        f.write(f"\nTraining samples: {metadata['train_size']}\n")
        f.write(f"Test samples: {metadata['test_size']}\n")
        f.write(f"Features: {metadata['features']}\n")
    
    print(f" Metadata saved to {metadata_file}")

def main():
    """Main training pipeline"""
    print("="*60)
    print("LIGHTGBM ETA PREDICTION MODEL TRAINING")
    print("="*60)
    
    config = load_config()
    
    # Load features
    features_file = Path(config['data']['final_dir']) / 'trip_features.parquet'
    df = load_features(features_file)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)
    
    # Train model
    model, evals_result = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    
    # Visualizations
    output_dir = Path(config['outputs']['eta_model_dir'])
    plot_results(y_test, y_pred, output_dir)
    plot_feature_importance(model, feature_cols, output_dir)
    
    # Save model
    metadata = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'features': feature_cols
    }
    save_model(model, output_dir, metadata)
    
    print("\n" + "="*60)
    print(" MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel and results saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Check visualizations in {output_dir}")
    print(f"  2. Review feature importance")
    print(f"  3. If MAE > 3.0, tune hyperparameters")
    print(f"  4. Integrate model with backend API")
    
    return model, mae

if __name__ == "__main__":
    main()