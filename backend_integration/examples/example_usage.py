"""
Example usage of DNerveModelLoader

Demonstrates all features for backend integration.
Run: python backend_integration/examples/example_usage.py

Author: Group 2 - ML Team
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend_integration.model_loader import (
    DNerveModelLoader, 
    PredictionRequest, 
    predict_trip_duration
)


def example_1_basic_usage():
    """Example 1: Basic prediction with full control"""
    print("="*70)
    print("EXAMPLE 1: Basic Prediction")
    print("="*70)
    
    # Initialize loader (singleton - loads once, reused everywhere)
    loader = DNerveModelLoader()
    
    # Create prediction request
    request = PredictionRequest(
        distance_km=12.5,
        num_points=35,
        start_lon=116.3975,  # Beijing Tiananmen Square
        start_lat=39.9087,
        end_lon=116.4832,    # Beijing Capital Airport area
        end_lat=39.9897,
        hour=8,              # 8 AM
        day_of_week=1,       # Tuesday (0=Monday, 6=Sunday)
        is_weekend=0,
        is_rush_hour=1,      # Yes, morning rush
        avg_speed_kph=22.0
    )
    
    # Get prediction
    response = loader.predict_eta(request)
    
    # Display results
    print(f"\n Trip Details:")
    print(f"   From: ({request.start_lat:.4f}, {request.start_lon:.4f})")
    print(f"   To:   ({request.end_lat:.4f}, {request.end_lon:.4f})")
    print(f"   Distance: {request.distance_km} km")
    print(f"   Time: {request.hour}:00 ({'Weekday' if not request.is_weekend else 'Weekend'})")
    print(f"   Rush Hour: {'Yes' if request.is_rush_hour else 'No'}")
    print(f"   Avg Speed: {request.avg_speed_kph} km/h")
    
    print(f"\n  Prediction:")
    print(f"   Duration: {response.predicted_duration_minutes:.1f} minutes")
    print(f"   95% Confidence: {response.confidence_interval[0]:.1f} - {response.confidence_interval[1]:.1f} min")
    print(f"   Model Version: {response.model_version}")
    print(f"   Timestamp: {response.timestamp}")
    
    # Convert to JSON (for API response)
    print(f"\n JSON Response:")
    import json
    print(json.dumps(response.to_dict(), indent=2))


def example_2_simplified_function():
    """Example 2: Quick prediction with simplified function"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Simplified Convenience Function")
    print("="*70)
    
    # One-line prediction (hides complexity)
    duration = predict_trip_duration(
        distance_km=15.5,
        start_lon=116.4,
        start_lat=39.9,
        end_lon=116.6,
        end_lat=40.1,
        hour=17,           # 5 PM
        day_of_week=4,     # Friday
        avg_speed_kph=18.5,
        is_rush_hour=1     # Evening rush
    )
    
    print(f"\n  Predicted Duration: {duration:.1f} minutes")
    print(f"   ({duration/60:.1f} hours)")


def example_3_batch_predictions():
    """Example 3: Compare multiple route alternatives"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Predictions (Route Comparison)")
    print("="*70)
    
    loader = DNerveModelLoader()
    
    # Simulate 3 alternative routes (like Google Maps alternatives)
    routes = [
        {
            'name': 'Route A - Highway (Fastest)',
            'distance_km': 10.0,
            'avg_speed_kph': 40.0,
            'description': 'Via Ring Road'
        },
        {
            'name': 'Route B - Main Roads',
            'distance_km': 12.0,
            'avg_speed_kph': 28.0,
            'description': 'Via City Center'
        },
        {
            'name': 'Route C - Local Streets',
            'distance_km': 15.0,
            'avg_speed_kph': 22.0,
            'description': 'Avoiding tolls'
        }
    ]
    
    print("\n  Comparing 3 Routes (8 AM, Weekday):\n")
    
    results = []
    for route in routes:
        request = PredictionRequest(
            distance_km=route['distance_km'],
            num_points=30,
            start_lon=116.3,
            start_lat=39.9,
            end_lon=116.5,
            end_lat=40.0,
            hour=8,
            day_of_week=1,
            is_weekend=0,
            is_rush_hour=1,
            avg_speed_kph=route['avg_speed_kph']
        )
        
        response = loader.predict_eta(request)
        
        results.append({
            'name': route['name'],
            'distance': route['distance_km'],
            'description': route['description'],
            'eta': response.predicted_duration_minutes
        })
        
        print(f"   {route['name']}")
        print(f"      {route['description']}")
        print(f"      Distance: {route['distance_km']} km @ {route['avg_speed_kph']} km/h")
        print(f"      ETA: {response.predicted_duration_minutes:.1f} minutes")
        print()
    
    # Find fastest
    fastest = min(results, key=lambda x: x['eta'])
    print(f"    Recommended: {fastest['name']} ({fastest['eta']:.1f} min)")


def example_4_error_handling():
    """Example 4: Proper error handling for production"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Error Handling & Validation")
    print("="*70)
    
    loader = DNerveModelLoader()
    
    # Test Case 1: Invalid distance
    print("\n Test 1: Invalid distance (negative)")
    try:
        invalid_request = PredictionRequest(
            distance_km=-10.0,  # Invalid!
            num_points=30,
            start_lon=116.4,
            start_lat=39.9,
            end_lon=116.5,
            end_lat=40.0,
            hour=12,
            day_of_week=2,
            is_weekend=0,
            is_rush_hour=0,
            avg_speed_kph=30.0
        )
        response = loader.predict_eta(invalid_request)
    except ValueError as e:
        print(f"    Caught validation error: {e}")
    
    # Test Case 2: Invalid coordinates
    print("\n Test 2: Invalid coordinates (out of bounds)")
    try:
        invalid_request = PredictionRequest(
            distance_km=10.0,
            num_points=30,
            start_lon=200.0,  # Invalid!
            start_lat=39.9,
            end_lon=116.5,
            end_lat=40.0,
            hour=12,
            day_of_week=2,
            is_weekend=0,
            is_rush_hour=0,
            avg_speed_kph=30.0
        )
        response = loader.predict_eta(invalid_request)
    except ValueError as e:
        print(f"    Caught validation error: {e}")
    
    # Test Case 3: Invalid hour
    print("\n Test 3: Invalid hour (> 23)")
    try:
        invalid_request = PredictionRequest(
            distance_km=10.0,
            num_points=30,
            start_lon=116.4,
            start_lat=39.9,
            end_lon=116.5,
            end_lat=40.0,
            hour=25,  # Invalid!
            day_of_week=2,
            is_weekend=0,
            is_rush_hour=0,
            avg_speed_kph=30.0
        )
        response = loader.predict_eta(invalid_request)
    except ValueError as e:
        print(f"    Caught validation error: {e}")


def example_5_model_info():
    """Example 5: Get model metadata and health status"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Model Information & Health Check")
    print("="*70)
    
    loader = DNerveModelLoader()
    
    # Get model info
    print("\n Model Information:")
    info = loader.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Health check
    print("\n Health Check:")
    health = loader.health_check()
    
    if health['healthy']:
        print("    Status: Healthy")
    else:
        print("    Status: Unhealthy")
    
    print("\n   Detailed Checks:")
    for check, result in health['checks'].items():
        if isinstance(result, bool):
            status = "Success" if result else "Failed"
            print(f"      {status} {check}")
        else:
            print(f"        {check}: {result}")


def example_6_fastapi_integration():
    """Example 6: FastAPI integration pattern"""
    print("\n" + "="*70)
    print("EXAMPLE 6: FastAPI Integration Pattern")
    print("="*70)
    
    fastapi_code = '''
# ============================================================
# COPY THIS TO YOUR d-nerve-backend REPOSITORY
# File: app/routers/eta.py
# ============================================================

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
from app.ml.model_loader import DNerveModelLoader, PredictionRequest

router = APIRouter(
    prefix="/api/v1",
    tags=["ETA Prediction"]
)

# Initialize model loader (singleton - loads once at startup)
model_loader = DNerveModelLoader()


class ETARequest(BaseModel):
    """ETA prediction request schema"""
    distance_km: float = Field(..., gt=0, le=200, description="Trip distance in km")
    start_lon: float = Field(..., description="Starting longitude")
    start_lat: float = Field(..., description="Starting latitude")
    end_lon: float = Field(..., description="Ending longitude")
    end_lat: float = Field(..., description="Ending latitude")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon)")
    avg_speed_kph: float = Field(..., gt=0, le=200, description="Average speed")
    num_points: int = Field(30, ge=10, le=1000, description="GPS points")
    is_rush_hour: int = Field(0, ge=0, le=1, description="Rush hour flag")
    
    class Config:
        schema_extra = {
            "example": {
                "distance_km": 12.5,
                "start_lon": 116.4,
                "start_lat": 39.9,
                "end_lon": 116.5,
                "end_lat": 40.0,
                "hour": 8,
                "day_of_week": 1,
                "avg_speed_kph": 22.0,
                "num_points": 35,
                "is_rush_hour": 1
            }
        }


class ETAResponse(BaseModel):
    """ETA prediction response schema"""
    predicted_duration_minutes: float
    confidence_interval: dict
    model_version: str
    timestamp: str


@router.post(
    "/predict-eta",
    response_model=ETAResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict trip duration",
    description="Predict ETA for a trip based on distance, time, and traffic conditions"
)
async def predict_eta(request: ETARequest):
    """
    Predict trip ETA using LightGBM model
    
    - **distance_km**: Trip distance in kilometers
    - **start_lon/lat**: Starting coordinates
    - **end_lon/lat**: Ending coordinates
    - **hour**: Hour of day (0-23)
    - **day_of_week**: 0=Monday, 6=Sunday
    - **avg_speed_kph**: Expected average speed
    - **is_rush_hour**: 1 if rush hour, 0 otherwise
    
    Returns predicted duration with 95% confidence interval.
    """
    try:
        # Convert FastAPI request to ML request
        ml_request = PredictionRequest(
            distance_km=request.distance_km,
            num_points=request.num_points,
            start_lon=request.start_lon,
            start_lat=request.start_lat,
            end_lon=request.end_lon,
            end_lat=request.end_lat,
            hour=request.hour,
            day_of_week=request.day_of_week,
            is_weekend=1 if request.day_of_week >= 5 else 0,
            is_rush_hour=request.is_rush_hour,
            avg_speed_kph=request.avg_speed_kph
        )
        
        # Get prediction
        response = model_loader.predict_eta(ml_request)
        
        # Return JSON response
        return response.to_dict()
        
    except ValueError as e:
        # Validation error (bad input)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Returns ML model metadata and performance metrics"
)
async def get_model_info():
    """Get ML model information"""
    return model_loader.get_model_info()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="ML health check",
    description="Check if ML models are loaded and operational"
)
async def health_check():
    """Health check for ML models"""
    health = model_loader.health_check()
    
    if not health['healthy']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models not healthy"
        )
    
    return health


# ============================================================
# END OF FASTAPI CODE
# ============================================================
'''
    
    print(fastapi_code)
    
    print("\n" + "="*70)
    print("To use in your backend:")
    print("  1. Copy code above to: d-nerve-backend/app/routers/eta.py")
    print("  2. Register in main.py: app.include_router(eta.router)")
    print("  3. Test: POST http://localhost:8000/api/v1/predict-eta")
    print("="*70)


def main():
    """Run all examples"""
    print("\n" + ""*35)
    print(" "*20 + "D-NERVE MODEL LOADER")
    print(" "*25 + "Usage Examples")
    print(""*35 + "\n")
    
    # Run examples
    example_1_basic_usage()
    example_2_simplified_function()
    example_3_batch_predictions()
    example_4_error_handling()
    example_5_model_info()
    example_6_fastapi_integration()
    
    # Summary
    print("\n" + "="*70)
    print(" ALL EXAMPLES COMPLETE")
    print("="*70)
    
    print("\n What you learned:")
    print("   1. Basic prediction with full control")
    print("   2. Simplified function for quick usage")
    print("   3. Batch predictions for route comparison")
    print("   4. Error handling and validation")
    print("   5. Model info and health checks")
    print("   6. FastAPI integration pattern")
    
    print("\n Next steps for Backend Team:")
    print("   1. Copy model_loader.py to d-nerve-backend/app/ml/")
    print("   2. Copy model files to d-nerve-backend/app/ml/models/")
    print("   3. Create FastAPI endpoint using Example 6 pattern")
    print("   4. Test with Postman")
    print("   5. Read ../docs/BACKEND_INTEGRATION_GUIDE.md for details")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()