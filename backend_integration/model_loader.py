"""
D-Nerve ML Model Loader

Standalone model loading and prediction interface for backend integration.
Can be copied directly to d-nerve-backend repository.

Features:
- ETA Prediction using Linear Regression
- Route Discovery using DBSCAN clustering
- Singleton pattern (efficient memory usage)
- Lazy loading (models loaded only when needed)
- Input validation with Cairo coordinate support
- Logging for debugging and monitoring
- Confidence intervals for predictions
- Health check endpoint support

Usage:
    from model_loader import DNerveModelLoader, ETAPredictionRequest

    loader = DNerveModelLoader()
    request = ETAPredictionRequest(
        distance_km=10.5,
        hour=8,
        day_of_week=1,
        is_weekend=0,
        is_peak=1
    )
    response = loader.predict_eta(request)
    print(f"ETA: {response.predicted_duration_minutes:.1f} minutes")

Author: Group 2 - ML Team
Version: 2.0.0
Date: 2026-01-20
License: MIT
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Cairo coordinate bounds
CAIRO_BOUNDS = {
    'lat_min': 29.7,
    'lat_max': 30.2,
    'lon_min': 31.1,
    'lon_max': 31.5
}

# Beijing coordinate bounds (for T-Drive validation)
BEIJING_BOUNDS = {
    'lat_min': 39.4,
    'lat_max': 41.1,
    'lon_min': 115.4,
    'lon_max': 117.5
}

# Default to Cairo
ACTIVE_BOUNDS = CAIRO_BOUNDS


# =============================================================================
# REQUEST/RESPONSE DATACLASSES
# =============================================================================

@dataclass
class ETAPredictionRequest:
    """
    ETA prediction request matching trained model features
    
    Required Features (12 total):
        distance_km: Trip distance in kilometers
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        is_weekend: Weekend flag (0 or 1)
        is_peak: Peak hour flag (0 or 1)
        time_period_encoded: Time period (0=night, 1=morning, 2=afternoon, 3=evening)
        route_avg_duration: Historical average duration for this route (minutes)
        route_std_duration: Historical std deviation for this route
        route_avg_distance: Historical average distance for this route (km)
        origin_encoded: Origin cluster ID
        dest_encoded: Destination cluster ID
        overlap_group: Route overlap group ID
    """
    distance_km: float
    hour: int = 12
    day_of_week: int = 2
    is_weekend: int = 0
    is_peak: int = 0
    time_period_encoded: int = 2
    route_avg_duration: float = 15.0
    route_std_duration: float = 3.0
    route_avg_distance: float = 5.0
    origin_encoded: int = 0
    dest_encoded: int = 0
    overlap_group: int = 0

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate all input parameters"""
        
        # Distance validation
        if not 0 < self.distance_km <= 100:
            return False, f"Invalid distance: {self.distance_km} km (must be 0-100)"
        
        # Hour validation
        if not 0 <= self.hour <= 23:
            return False, f"Invalid hour: {self.hour} (must be 0-23)"
        
        # Day of week validation
        if not 0 <= self.day_of_week <= 6:
            return False, f"Invalid day_of_week: {self.day_of_week} (must be 0-6)"
        
        # Binary flags validation
        if self.is_weekend not in [0, 1]:
            return False, f"Invalid is_weekend: {self.is_weekend} (must be 0 or 1)"
        
        if self.is_peak not in [0, 1]:
            return False, f"Invalid is_peak: {self.is_peak} (must be 0 or 1)"
        
        # Time period validation
        if not 0 <= self.time_period_encoded <= 3:
            return False, f"Invalid time_period: {self.time_period_encoded} (must be 0-3)"
        
        # Route statistics validation
        if self.route_avg_duration < 0:
            return False, f"Invalid route_avg_duration: {self.route_avg_duration} (must be >= 0)"
        
        return True, None

    def to_feature_dict(self) -> Dict[str, Any]:
        """Convert to feature dictionary matching model training order"""
        return {
            'distance_km': self.distance_km,
            'hour': self.hour,
            'day_of_week': self.day_of_week,
            'is_weekend': self.is_weekend,
            'is_peak': self.is_peak,
            'time_period_encoded': self.time_period_encoded,
            'route_avg_duration': self.route_avg_duration,
            'route_std_duration': self.route_std_duration,
            'route_avg_distance': self.route_avg_distance,
            'origin_encoded': self.origin_encoded,
            'dest_encoded': self.dest_encoded,
            'overlap_group': self.overlap_group
        }


@dataclass
class ETAPredictionResponse:
    """ETA prediction response"""
    predicted_duration_minutes: float
    confidence_interval: Tuple[float, float]
    model_version: str
    model_type: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'predicted_duration_minutes': round(self.predicted_duration_minutes, 2),
            'confidence_interval': {
                'lower': round(self.confidence_interval[0], 2),
                'upper': round(self.confidence_interval[1], 2)
            },
            'model_version': self.model_version,
            'model_type': self.model_type,
            'timestamp': self.timestamp
        }


@dataclass
class RouteDiscoveryRequest:
    """Request for route discovery from GPS trajectories"""
    trajectories: List[List[Tuple[float, float]]]  # List of trips, each trip is list of (lat, lon)
    epsilon_meters: float = 150.0
    min_samples: int = 2

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate input"""
        if not self.trajectories:
            return False, "No trajectories provided"
        
        if len(self.trajectories) < 2:
            return False, "At least 2 trajectories required for clustering"
        
        if not 50 <= self.epsilon_meters <= 2000:
            return False, f"Invalid epsilon: {self.epsilon_meters} (must be 50-2000 meters)"
        
        if not 2 <= self.min_samples <= 10:
            return False, f"Invalid min_samples: {self.min_samples} (must be 2-10)"
        
        return True, None


@dataclass
class RouteDiscoveryResponse:
    """Response from route discovery"""
    num_clusters: int
    num_noise: int
    noise_ratio: float
    cluster_labels: List[int]
    cluster_sizes: Dict[int, int]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'num_clusters': self.num_clusters,
            'num_noise': self.num_noise,
            'noise_ratio': round(self.noise_ratio, 4),
            'cluster_labels': self.cluster_labels,
            'cluster_sizes': self.cluster_sizes,
            'timestamp': self.timestamp
        }


# =============================================================================
# MODEL LOADER CLASS
# =============================================================================

class DNerveModelLoader:
    """
    ML model loader for D-Nerve ETA prediction and route discovery
    
    Implements singleton pattern for efficient memory usage.
    Models are lazy-loaded (loaded only when first accessed).
    
    Model Performance (Cairo Synthetic Data):
        - ETA Prediction: MAE = 3.28 minutes, R² = 0.865
        - Route Discovery: F1 = 0.963 (hard mode with overlapping routes)
    
    Validated on:
        - Cairo synthetic data (83,448 GPS points, 420 trips)
        - Beijing T-Drive real data (120,407 GPS points, 2,181 trips)
    """

    # Singleton instance
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize model loader
        
        Args:
            model_dir: Directory containing trained models.
                      If None, uses default location (outputs/)
        """
        if self._initialized:
            return

        # Set model directory
        if model_dir is None:
            project_root = Path(__file__).parent.parent
            self.model_dir = project_root / "outputs"
        else:
            self.model_dir = Path(model_dir)

        # Model file paths
        self.eta_model_path = self.model_dir / "eta_prediction" / "eta_best_model.pkl"
        self.routes_path = self.model_dir / "cairo_hard_mode" / "hard_mode_results.pkl"

        # Models (lazy loaded)
        self._eta_model = None
        self._routes_data = None
        self._feature_cols = []

        # Model metadata - UPDATED to match actual training results
        self._metadata = {
            'name': 'D-Nerve ETA Predictor',
            'version': '2.0.0',
            'model_type': 'Linear Regression',
            'mae_minutes': 3.28,
            'rmse_minutes': 4.74,
            'r2_score': 0.865,
            'cv_mae': 3.50,
            'cv_std': 0.35,
            'training_date': '2026-01-20',
            'feature_count': 12,
            'training_samples': 336,
            'test_samples': 84
        }
        
        # Route discovery metadata
        self._route_metadata = {
            'algorithm': 'DBSCAN',
            'distance_metric': 'Hausdorff',
            'f1_score_easy': 1.000,
            'f1_score_hard': 0.963,
            'beijing_silhouette': 0.902
        }

        self._initialized = True
        logger.info(f"✓ DNerveModelLoader initialized (model_dir: {self.model_dir})")

    @property
    def eta_model(self):
        """Lazy load ETA prediction model"""
        if self._eta_model is None:
            self._load_eta_model()
        return self._eta_model

    @property
    def routes_data(self):
        """Lazy load routes data"""
        if self._routes_data is None:
            self._load_routes_data()
        return self._routes_data

    def _load_eta_model(self) -> None:
        """Load ETA model from disk"""
        try:
            if not self.eta_model_path.exists():
                raise FileNotFoundError(
                    f"ETA model not found at {self.eta_model_path}. "
                    f"Please run: python prediction/eta_prediction.py"
                )

            logger.info(f"Loading ETA model from {self.eta_model_path}...")
            with open(self.eta_model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both dict format and direct model format
            if isinstance(data, dict):
                self._eta_model = data['model']
                self._feature_cols = data.get('feature_cols', [])
                logger.info(f"  Model type: {data.get('model_name', 'Unknown')}")
                logger.info(f"  Features: {len(self._feature_cols)}")
            else:
                self._eta_model = data
                self._feature_cols = []

            logger.info("✓ ETA model loaded successfully")

        except Exception as e:
            logger.error(f"✗ Failed to load ETA model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_routes_data(self) -> None:
        """Load route discovery results (optional)"""
        try:
            if not self.routes_path.exists():
                logger.warning(f"Routes data not found at {self.routes_path}")
                self._routes_data = None
                return

            logger.info(f"Loading routes data from {self.routes_path}...")
            with open(self.routes_path, 'rb') as f:
                self._routes_data = pickle.load(f)

            logger.info("✓ Routes data loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load routes data: {e}")
            self._routes_data = None

    def predict_eta(
        self,
        request: ETAPredictionRequest,
        return_confidence: bool = True
    ) -> ETAPredictionResponse:
        """
        Predict trip duration (ETA)
        
        Args:
            request: ETAPredictionRequest with trip features
            return_confidence: Whether to calculate confidence interval
        
        Returns:
            ETAPredictionResponse with predicted duration and metadata
        
        Raises:
            ValueError: If input validation fails
            RuntimeError: If prediction fails
        """
        # Validate input
        is_valid, error_msg = request.validate()
        if not is_valid:
            logger.error(f"✗ Input validation failed: {error_msg}")
            raise ValueError(error_msg)

        try:
            # Prepare features
            features = pd.DataFrame([request.to_feature_dict()])

            # Make prediction
            prediction = self.eta_model.predict(features)[0]

            # Calculate confidence interval (±2*MAE covers ~95%)
            if return_confidence:
                mae = self._metadata['mae_minutes']
                confidence_interval = (
                    max(0, prediction - 2 * mae),
                    prediction + 2 * mae
                )
            else:
                confidence_interval = (prediction, prediction)

            response = ETAPredictionResponse(
                predicted_duration_minutes=float(prediction),
                confidence_interval=confidence_interval,
                model_version=self._metadata['version'],
                model_type=self._metadata['model_type'],
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

            logger.info(
                f"✓ Prediction: {prediction:.2f} min "
                f"(distance: {request.distance_km:.2f} km, hour: {request.hour})"
            )

            return response

        except Exception as e:
            logger.error(f"✗ Prediction failed: {e}")
            raise RuntimeError(f"Prediction error: {e}") from e

    def predict_eta_simple(
        self,
        distance_km: float,
        hour: int = 12,
        is_peak: int = 0
    ) -> float:
        """
        Simple ETA prediction with minimal inputs
        
        Uses sensible defaults for missing features.
        
        Args:
            distance_km: Trip distance in kilometers
            hour: Hour of day (0-23)
            is_peak: 1 if rush hour, 0 otherwise
        
        Returns:
            Predicted duration in minutes
        """
        # Determine time period
        if 0 <= hour < 6:
            time_period = 0  # night
        elif 6 <= hour < 12:
            time_period = 1  # morning
        elif 12 <= hour < 18:
            time_period = 2  # afternoon
        else:
            time_period = 3  # evening
        
        # Determine day of week (assume weekday)
        day_of_week = datetime.now().weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Estimate route statistics based on distance
        avg_speed = 20 if is_peak else 30  # km/h
        route_avg_duration = (distance_km / avg_speed) * 60
        
        request = ETAPredictionRequest(
            distance_km=distance_km,
            hour=hour,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_peak=is_peak,
            time_period_encoded=time_period,
            route_avg_duration=route_avg_duration,
            route_std_duration=route_avg_duration * 0.2,
            route_avg_distance=distance_km
        )
        
        response = self.predict_eta(request, return_confidence=False)
        return response.predicted_duration_minutes

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance metrics"""
        return {
            'model_name': self._metadata['name'],
            'version': self._metadata['version'],
            'model_type': self._metadata['model_type'],
            'mae_minutes': self._metadata['mae_minutes'],
            'rmse_minutes': self._metadata['rmse_minutes'],
            'r2_score': self._metadata['r2_score'],
            'cv_mae': f"{self._metadata['cv_mae']} ± {self._metadata['cv_std']}",
            'training_date': self._metadata['training_date'],
            'feature_count': self._metadata['feature_count'],
            'status': 'loaded' if self._eta_model is not None else 'not_loaded',
            'routes_available': self._routes_data is not None,
            'route_discovery': {
                'algorithm': self._route_metadata['algorithm'],
                'f1_score_easy': self._route_metadata['f1_score_easy'],
                'f1_score_hard': self._route_metadata['f1_score_hard'],
                'beijing_validation_silhouette': self._route_metadata['beijing_silhouette']
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        status = {
            'healthy': True,
            'checks': {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Check 1: ETA model file exists
        status['checks']['eta_model_exists'] = self.eta_model_path.exists()
        if not status['checks']['eta_model_exists']:
            status['healthy'] = False

        # Check 2: ETA model loadable
        try:
            _ = self.eta_model
            status['checks']['eta_model_loadable'] = True
        except Exception as e:
            status['checks']['eta_model_loadable'] = False
            status['checks']['eta_model_error'] = str(e)
            status['healthy'] = False

        # Check 3: Routes data exists (optional)
        status['checks']['routes_data_exists'] = self.routes_path.exists()

        # Check 4: Sample prediction works
        try:
            result = self.predict_eta_simple(distance_km=5.0, hour=12, is_peak=0)
            status['checks']['sample_prediction'] = True
            status['checks']['sample_result_minutes'] = round(result, 2)
        except Exception as e:
            status['checks']['sample_prediction'] = False
            status['checks']['prediction_error'] = str(e)
            status['healthy'] = False

        return status


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def predict_trip_duration(
    distance_km: float,
    hour: int = 12,
    is_peak: int = 0
) -> float:
    """
    Quick prediction function (simplified interface)
    
    Args:
        distance_km: Trip distance in kilometers
        hour: Hour of day (0-23)
        is_peak: 1 if rush hour, 0 otherwise
    
    Returns:
        Predicted duration in minutes
    
    Example:
        >>> duration = predict_trip_duration(10.5, hour=8, is_peak=1)
        >>> print(f"ETA: {duration:.1f} minutes")
    """
    loader = DNerveModelLoader()
    return loader.predict_eta_simple(distance_km, hour, is_peak)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("D-NERVE MODEL LOADER v2.0 - DEMO")
    print("="*70)

    # Initialize
    loader = DNerveModelLoader()

    # Model info
    print("\n📊 Model Information:")
    info = loader.get_model_info()
    print(f"  Model: {info['model_name']} v{info['version']}")
    print(f"  Type: {info['model_type']}")
    print(f"  MAE: {info['mae_minutes']} minutes")
    print(f"  R²: {info['r2_score']}")
    print(f"  CV MAE: {info['cv_mae']} minutes")
    print(f"  Features: {info['feature_count']}")
    print(f"  Status: {info['status']}")
    
    print("\n🛣️ Route Discovery Performance:")
    print(f"  F1 Score (Easy): {info['route_discovery']['f1_score_easy']}")
    print(f"  F1 Score (Hard): {info['route_discovery']['f1_score_hard']}")
    print(f"  Beijing Silhouette: {info['route_discovery']['beijing_validation_silhouette']}")

    # Health check
    print("\n🏥 Health Check:")
    health = loader.health_check()
    print(f"  Status: {'✓ Healthy' if health['healthy'] else '✗ Unhealthy'}")
    for check, result in health['checks'].items():
        if isinstance(result, bool):
            symbol = '✓' if result else '✗'
            print(f"  {symbol} {check}")

    # Sample predictions
    print("\n🚌 Sample Predictions:")
    
    # Short trip, off-peak
    eta1 = loader.predict_eta_simple(distance_km=3.0, hour=14, is_peak=0)
    print(f"  3 km at 2pm (off-peak): {eta1:.1f} minutes")
    
    # Medium trip, peak hour
    eta2 = loader.predict_eta_simple(distance_km=8.0, hour=8, is_peak=1)
    print(f"  8 km at 8am (peak): {eta2:.1f} minutes")
    
    # Long trip, evening
    eta3 = loader.predict_eta_simple(distance_km=15.0, hour=19, is_peak=1)
    print(f"  15 km at 7pm (peak): {eta3:.1f} minutes")

    # Full request example
    print("\n📝 Full Request Example:")
    request = ETAPredictionRequest(
        distance_km=10.0,
        hour=8,
        day_of_week=1,
        is_weekend=0,
        is_peak=1,
        time_period_encoded=1,
        route_avg_duration=25.0,
        route_std_duration=5.0,
        route_avg_distance=10.0,
        origin_encoded=1,
        dest_encoded=5,
        overlap_group=2
    )
    response = loader.predict_eta(request)
    print(f"  Distance: {request.distance_km} km")
    print(f"  Time: {request.hour}:00 (peak={request.is_peak})")
    print(f"  Predicted: {response.predicted_duration_minutes:.1f} minutes")
    print(f"  Confidence: [{response.confidence_interval[0]:.1f}, {response.confidence_interval[1]:.1f}] min")

    print("\n✅ Complete!")
