# D-Nerve ML API Specification

**Version:** 2.0.0  
**Last Updated:** January 20, 2026  
**Team:** Group 2 - Machine Learning

This document defines the API contract between ML models and backend.

---

## Overview

### Base URL
```
Production: https://api.d-nerve.com/api/v1
Development: http://localhost:8000/api/v1
```

### Model Performance Summary

| Component | Metric | Value |
|-----------|--------|-------|
| ETA Prediction | MAE | 3.28 minutes |
| ETA Prediction | R² | 0.865 |
| ETA Prediction | Model | Linear Regression |
| Route Discovery (Easy) | F1 Score | 1.000 |
| Route Discovery (Hard) | F1 Score | 0.963 |
| Beijing Validation | Silhouette | 0.902 |

---

## Endpoints

### 1. Predict ETA

Predict trip duration based on distance, time, and route characteristics.

**Endpoint:** `POST /predict-eta`

**Request Body:**
```json
{
  "distance_km": 10.0,
  "hour": 8,
  "day_of_week": 1,
  "is_weekend": 0,
  "is_peak": 1,
  "time_period_encoded": 1,
  "route_avg_duration": 25.0,
  "route_std_duration": 5.0,
  "route_avg_distance": 10.0,
  "origin_encoded": 1,
  "dest_encoded": 5,
  "overlap_group": 2
}
```

**Field Descriptions:**

| Field | Type | Required | Range | Description |
|-------|------|----------|-------|-------------|
| `distance_km` | float | Yes | 0-100 | Trip distance in kilometers |
| `hour` | integer | Yes | 0-23 | Hour of day (24-hour format) |
| `day_of_week` | integer | Yes | 0-6 | Day (0=Monday, 6=Sunday) |
| `is_weekend` | integer | Yes | 0-1 | Weekend flag |
| `is_peak` | integer | Yes | 0-1 | Peak hour flag |
| `time_period_encoded` | integer | No | 0-3 | 0=night, 1=morning, 2=afternoon, 3=evening |
| `route_avg_duration` | float | No | ≥0 | Historical avg duration (minutes) |
| `route_std_duration` | float | No | ≥0 | Historical std deviation |
| `route_avg_distance` | float | No | ≥0 | Historical avg distance (km) |
| `origin_encoded` | integer | No | ≥0 | Origin cluster ID |
| `dest_encoded` | integer | No | ≥0 | Destination cluster ID |
| `overlap_group` | integer | No | ≥0 | Route overlap group ID |

**Success Response (200 OK):**
```json
{
  "predicted_duration_minutes": 28.45,
  "confidence_interval": {
    "lower": 21.89,
    "upper": 35.01
  },
  "model_version": "2.0.0",
  "model_type": "Linear Regression",
  "timestamp": "2026-01-20T10:30:45.123456Z"
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "Invalid distance: -10.0 km (must be 0-100)"
}
```

---

### 2. Simple ETA Prediction

Simplified prediction endpoint requiring only essential parameters.

**Endpoint:** `POST /predict-eta/simple`

**Request Body:**
```json
{
  "distance_km": 10.0,
  "hour": 8,
  "is_peak": 1
}
```

**Field Descriptions:**

| Field | Type | Required | Range | Description |
|-------|------|----------|-------|-------------|
| `distance_km` | float | Yes | 0-100 | Trip distance in kilometers |
| `hour` | integer | No | 0-23 | Hour of day (default: 12) |
| `is_peak` | integer | No | 0-1 | Peak hour flag (default: 0) |

**Success Response (200 OK):**
```json
{
  "predicted_duration_minutes": 28.45,
  "model_version": "2.0.0",
  "timestamp": "2026-01-20T10:30:45.123456Z"
}
```

---

### 3. Get Model Information

Get ML model metadata and performance metrics.

**Endpoint:** `GET /model-info`

**Success Response (200 OK):**
```json
{
  "model_name": "D-Nerve ETA Predictor",
  "version": "2.0.0",
  "model_type": "Linear Regression",
  "mae_minutes": 3.28,
  "rmse_minutes": 4.74,
  "r2_score": 0.865,
  "cv_mae": "3.50 ± 0.35",
  "training_date": "2026-01-20",
  "feature_count": 12,
  "status": "loaded",
  "routes_available": true,
  "route_discovery": {
    "algorithm": "DBSCAN",
    "f1_score_easy": 1.0,
    "f1_score_hard": 0.963,
    "beijing_validation_silhouette": 0.902
  }
}
```

---

### 4. Health Check

Check if ML models are operational.

**Endpoint:** `GET /health`

**Success Response (200 OK - Healthy):**
```json
{
  "healthy": true,
  "checks": {
    "eta_model_exists": true,
    "eta_model_loadable": true,
    "routes_data_exists": true,
    "sample_prediction": true,
    "sample_result_minutes": 15.23
  },
  "timestamp": "2026-01-20T10:30:45.123456Z"
}
```

**Error Response (503 Service Unavailable):**
```json
{
  "healthy": false,
  "checks": {
    "eta_model_exists": false,
    "eta_model_loadable": false,
    "eta_model_error": "FileNotFoundError: Model not found",
    "routes_data_exists": true,
    "sample_prediction": false
  },
  "timestamp": "2026-01-20T10:30:45.123456Z"
}
```

---

## Request/Response Schemas

### ETAPredictionRequest Schema (OpenAPI)
```yaml
ETAPredictionRequest:
  type: object
  required:
    - distance_km
    - hour
    - day_of_week
    - is_weekend
    - is_peak
  properties:
    distance_km:
      type: number
      minimum: 0
      maximum: 100
      example: 10.0
    hour:
      type: integer
      minimum: 0
      maximum: 23
      example: 8
    day_of_week:
      type: integer
      minimum: 0
      maximum: 6
      example: 1
    is_weekend:
      type: integer
      enum: [0, 1]
      example: 0
    is_peak:
      type: integer
      enum: [0, 1]
      example: 1
    time_period_encoded:
      type: integer
      minimum: 0
      maximum: 3
      default: 2
      example: 1
    route_avg_duration:
      type: number
      minimum: 0
      default: 15.0
      example: 25.0
    route_std_duration:
      type: number
      minimum: 0
      default: 3.0
      example: 5.0
    route_avg_distance:
      type: number
      minimum: 0
      default: 5.0
      example: 10.0
    origin_encoded:
      type: integer
      minimum: 0
      default: 0
      example: 1
    dest_encoded:
      type: integer
      minimum: 0
      default: 0
      example: 5
    overlap_group:
      type: integer
      minimum: 0
      default: 0
      example: 2
```

### ETAPredictionResponse Schema
```yaml
ETAPredictionResponse:
  type: object
  properties:
    predicted_duration_minutes:
      type: number
      example: 28.45
    confidence_interval:
      type: object
      properties:
        lower:
          type: number
          example: 21.89
        upper:
          type: number
          example: 35.01
    model_version:
      type: string
      example: "2.0.0"
    model_type:
      type: string
      example: "Linear Regression"
    timestamp:
      type: string
      format: date-time
      example: "2026-01-20T10:30:45.123456Z"
```

---

## Error Codes

| Status Code | Error Type | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | Bad Request | Invalid input parameters | Check request body against schema |
| 422 | Unprocessable Entity | Validation error | Fix data types/ranges |
| 500 | Internal Server Error | Model prediction failed | Check logs, verify model files |
| 503 | Service Unavailable | ML models not loaded | Run health check, restart service |

---

## Examples

### Example 1: Morning Commute (Rush Hour)

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict-eta" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 12.0,
    "hour": 8,
    "day_of_week": 1,
    "is_weekend": 0,
    "is_peak": 1,
    "time_period_encoded": 1,
    "route_avg_duration": 30.0,
    "route_std_duration": 6.0,
    "route_avg_distance": 12.0
  }'
```

**Response:**
```json
{
  "predicted_duration_minutes": 32.15,
  "confidence_interval": {
    "lower": 25.59,
    "upper": 38.71
  },
  "model_version": "2.0.0",
  "model_type": "Linear Regression",
  "timestamp": "2026-01-20T08:00:00.000000Z"
}
```

### Example 2: Simple Prediction

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict-eta/simple" \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 5.0,
    "hour": 14,
    "is_peak": 0
  }'
```

**Response:**
```json
{
  "predicted_duration_minutes": 12.34,
  "model_version": "2.0.0",
  "timestamp": "2026-01-20T14:00:00.000000Z"
}
```

### Example 3: Health Check

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

**Response:**
```json
{
  "healthy": true,
  "checks": {
    "eta_model_exists": true,
    "eta_model_loadable": true,
    "routes_data_exists": true,
    "sample_prediction": true,
    "sample_result_minutes": 15.23
  },
  "timestamp": "2026-01-20T10:30:45.123456Z"
}
```

---

## FastAPI Integration

### Complete Router Implementation

```python
# app/routers/eta.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.ml.model_loader import DNerveModelLoader, ETAPredictionRequest

router = APIRouter(prefix="/api/v1", tags=["ETA Prediction"])

# Initialize model loader (singleton)
model_loader = DNerveModelLoader()


class ETARequestFull(BaseModel):
    """Full ETA prediction request"""
    distance_km: float = Field(..., ge=0, le=100, description="Trip distance in km")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend flag")
    is_peak: int = Field(..., ge=0, le=1, description="Peak hour flag")
    time_period_encoded: int = Field(2, ge=0, le=3, description="Time period")
    route_avg_duration: float = Field(15.0, ge=0, description="Historical avg duration")
    route_std_duration: float = Field(3.0, ge=0, description="Historical std duration")
    route_avg_distance: float = Field(5.0, ge=0, description="Historical avg distance")
    origin_encoded: int = Field(0, ge=0, description="Origin cluster ID")
    dest_encoded: int = Field(0, ge=0, description="Destination cluster ID")
    overlap_group: int = Field(0, ge=0, description="Overlap group ID")


class ETARequestSimple(BaseModel):
    """Simple ETA prediction request"""
    distance_km: float = Field(..., ge=0, le=100, description="Trip distance in km")
    hour: int = Field(12, ge=0, le=23, description="Hour of day")
    is_peak: int = Field(0, ge=0, le=1, description="Peak hour flag")


@router.post("/predict-eta")
async def predict_eta(request: ETARequestFull):
    """Full ETA prediction with all features"""
    try:
        ml_request = ETAPredictionRequest(
            distance_km=request.distance_km,
            hour=request.hour,
            day_of_week=request.day_of_week,
            is_weekend=request.is_weekend,
            is_peak=request.is_peak,
            time_period_encoded=request.time_period_encoded,
            route_avg_duration=request.route_avg_duration,
            route_std_duration=request.route_std_duration,
            route_avg_distance=request.route_avg_distance,
            origin_encoded=request.origin_encoded,
            dest_encoded=request.dest_encoded,
            overlap_group=request.overlap_group
        )
        response = model_loader.predict_eta(ml_request)
        return response.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-eta/simple")
async def predict_eta_simple(request: ETARequestSimple):
    """Simple ETA prediction with minimal inputs"""
    try:
        duration = model_loader.predict_eta_simple(
            distance_km=request.distance_km,
            hour=request.hour,
            is_peak=request.is_peak
        )
        return {
            "predicted_duration_minutes": round(duration, 2),
            "model_version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_model_info():
    """Get ML model information"""
    return model_loader.get_model_info()


@router.get("/health")
async def health_check():
    """ML model health check"""
    result = model_loader.health_check()
    if not result['healthy']:
        raise HTTPException(status_code=503, detail=result)
    return result
```

---

## Files Required for Backend

### Copy These Files

| Source (ML Repo) | Destination (Backend Repo) |
|------------------|---------------------------|
| `backend_integration/model_loader.py` | `app/ml/model_loader.py` |
| `outputs/eta_prediction/eta_best_model.pkl` | `app/ml/models/eta_best_model.pkl` |
| `outputs/cairo_hard_mode/hard_mode_results.pkl` | `app/ml/models/hard_mode_results.pkl` (optional) |

### Update Model Path in model_loader.py

After copying, update the model path in `model_loader.py`:

```python
# Change from:
self.eta_model_path = self.model_dir / "eta_prediction" / "eta_best_model.pkl"

# To:
self.eta_model_path = Path(__file__).parent / "models" / "eta_best_model.pkl"
```

---

## Dependencies

```txt
# requirements.txt additions
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-01 | Initial API specification |
| 2.0.0 | 2026-01-20 | Updated model (Linear Regression), correct metrics, Cairo coordinates, Beijing validation |

---

**Maintained by:** Group 2 - ML Team
