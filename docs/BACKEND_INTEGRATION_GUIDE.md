\# Backend Integration Guide



\*\*Complete guide for Group 1 (Backend Team) to integrate ML models\*\*



This document provides step-by-step instructions for integrating the ML models into the d-nerve-backend FastAPI application.



---



\## Table of Contents



1\. \[Overview](#overview)

2\. \[Prerequisites](#prerequisites)

3\. \[Setup Instructions](#setup-instructions)

4\. \[Integration Steps](#integration-steps)

5\. \[Testing](#testing)

6\. \[Deployment](#deployment)

7\. \[Troubleshooting](#troubleshooting)



---



\## Overview



\### What You're Integrating



\*\*ML Models:\*\*

\- \*\*ETA Prediction Model\*\* (LightGBM): Predicts trip duration in minutes

\- \*\*Route Discovery Results\*\* (DBSCAN): 20 discovered routes (optional)



\*\*Performance:\*\*

\- MAE: 9.04 minutes

\- R²: 0.9513 (95.13% variance explained)

\- Average prediction time: < 50ms



---



\## Prerequisites



\### Required Software



\- \[ ] Python 3.11+

\- \[ ] FastAPI

\- \[ ] Pandas, NumPy

\- \[ ] LightGBM

\- \[ ] Git



\### Required Files from ML Repo



From `d-nerve-ml-models` repository:



1\. \*\*Model Loader:\*\*

&nbsp;  - `backend\_integration/model\_loader.py`



2\. \*\*Trained Models:\*\*

&nbsp;  - `outputs/eta\_model/lightgbm\_eta\_model.pkl` (~500 KB)

&nbsp;  - `outputs/route\_discovery/route\_discovery\_results.pkl` (~14 KB)



3\. \*\*Documentation:\*\*

&nbsp;  - `docs/API\_SPECIFICATION.md`

&nbsp;  - `backend\_integration/examples/example\_usage.py`



---



\## Setup Instructions



\### Step 1: Clone ML Repository

```bash

\# Navigate to your projects folder

cd ~/Projects  # Windows: cd %USERPROFILE%\\Projects



\# Clone ML repository

git clone https://github.com/d-nerve-cairo/d-nerve-ml-models.git



\# Navigate into it

cd d-nerve-ml-models



\# Verify files exist

ls backend\_integration/model\_loader.py

ls outputs/eta\_model/lightgbm\_eta\_model.pkl

```



---



\### Step 2: Prepare Backend Repository

```bash

\# Navigate to your backend repo

cd ../d-nerve-backend



\# Create ML directory structure

mkdir -p app/ml/models



\# Verify structure

tree app/ml  # or: ls -R app/ml (Mac/Linux)

```



\*\*Expected structure:\*\*

```

d-nerve-backend/

└── app/

&nbsp;   └── ml/

&nbsp;       └── models/

```



---



\### Step 3: Copy Files



\*\*Copy Model Loader:\*\*

```bash

\# From d-nerve-ml-models to d-nerve-backend

cp ../d-nerve-ml-models/backend\_integration/model\_loader.py app/ml/



\# Windows:

\# copy ..\\d-nerve-ml-models\\backend\_integration\\model\_loader.py app\\ml\\

```



\*\*Copy Trained Models:\*\*

```bash

\# Copy LightGBM model

cp ../d-nerve-ml-models/outputs/eta\_model/lightgbm\_eta\_model.pkl app/ml/models/



\# Copy route discovery results (optional)

cp ../d-nerve-ml-models/outputs/route\_discovery/route\_discovery\_results.pkl app/ml/models/



\# Windows:

\# copy ..\\d-nerve-ml-models\\outputs\\eta\_model\\lightgbm\_eta\_model.pkl app\\ml\\models\\

\# copy ..\\d-nerve-ml-models\\outputs\\route\_discovery\\route\_discovery\_results.pkl app\\ml\\models\\

```



\*\*Verify files:\*\*

```bash

ls app/ml/

\# Should show: model\_loader.py, models/



ls app/ml/models/

\# Should show: lightgbm\_eta\_model.pkl, route\_discovery\_results.pkl

```



---



\### Step 4: Install Dependencies

```bash

\# In d-nerve-backend repo

pip install lightgbm pandas numpy



\# Or add to requirements.txt:

echo "lightgbm>=4.0.0" >> requirements.txt

echo "pandas>=2.0.0" >> requirements.txt

echo "numpy>=1.24.0" >> requirements.txt



pip install -r requirements.txt

```



---



\## Integration Steps



\### Step 1: Create `\_\_init\_\_.py` for ML Module

```bash

\# Create file

touch app/ml/\_\_init\_\_.py  # Mac/Linux

\# type nul > app\\ml\\\_\_init\_\_.py  # Windows

```



---



\### Step 2: Create ETA Router



Create file: `app/routers/eta.py`

```python

"""

ETA Prediction Router



Provides ML-powered trip duration predictions.



Endpoints:

\- POST /api/v1/predict-eta: Predict trip duration

\- GET /api/v1/model-info: Get model metadata

\- GET /api/v1/health: ML health check



Author: Group 1 - Backend Team

Integration: Group 2 - ML Team

"""



from fastapi import APIRouter, HTTPException, status

from pydantic import BaseModel, Field

from typing import Optional

import logging



\# Import ML model loader

from app.ml.model\_loader import DNerveModelLoader, PredictionRequest as MLRequest



\# Configure logging

logger = logging.getLogger(\_\_name\_\_)



\# Create router

router = APIRouter(

&nbsp;   prefix="/api/v1",

&nbsp;   tags=\["ETA Prediction"]

)



\# Initialize model loader (singleton - loads once at startup)

try:

&nbsp;   model\_loader = DNerveModelLoader()

&nbsp;   logger.info(" ML model loader initialized successfully")

except Exception as e:

&nbsp;   logger.error(f" Failed to initialize ML model loader: {e}")

&nbsp;   model\_loader = None





\# ============================================================

\# REQUEST/RESPONSE MODELS

\# ============================================================



class ETARequest(BaseModel):

&nbsp;   """ETA prediction request"""

&nbsp;   distance\_km: float = Field(..., gt=0, le=200, description="Trip distance in km")

&nbsp;   start\_lon: float = Field(..., ge=31.0, le=32.0, description="Starting longitude (Cairo)")

&nbsp;   start\_lat: float = Field(..., ge=29.0, le=31.0, description="Starting latitude (Cairo)")

&nbsp;   end\_lon: float = Field(..., ge=31.0, le=32.0, description="Ending longitude (Cairo)")

&nbsp;   end\_lat: float = Field(..., ge=29.0, le=31.0, description="Ending latitude (Cairo)")

&nbsp;   hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")

&nbsp;   day\_of\_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")

&nbsp;   avg\_speed\_kph: float = Field(..., gt=0, le=200, description="Expected average speed in km/h")

&nbsp;   num\_points: int = Field(30, ge=10, le=1000, description="Number of GPS points")

&nbsp;   is\_rush\_hour: int = Field(0, ge=0, le=1, description="Rush hour flag (0 or 1)")

&nbsp;   

&nbsp;   class Config:

&nbsp;       schema\_extra = {

&nbsp;           "example": {

&nbsp;               "distance\_km": 12.5,

&nbsp;               "start\_lon": 31.2357,

&nbsp;               "start\_lat": 30.0444,

&nbsp;               "end\_lon": 31.3387,

&nbsp;               "end\_lat": 30.0626,

&nbsp;               "hour": 8,

&nbsp;               "day\_of\_week": 1,

&nbsp;               "avg\_speed\_kph": 22.0,

&nbsp;               "num\_points": 35,

&nbsp;               "is\_rush\_hour": 1

&nbsp;           }

&nbsp;       }





class ETAResponse(BaseModel):

&nbsp;   """ETA prediction response"""

&nbsp;   predicted\_duration\_minutes: float

&nbsp;   confidence\_interval: dict

&nbsp;   model\_version: str

&nbsp;   timestamp: str





\# ============================================================

\# ENDPOINTS

\# ============================================================



@router.post(

&nbsp;   "/predict-eta",

&nbsp;   response\_model=ETAResponse,

&nbsp;   status\_code=status.HTTP\_200\_OK,

&nbsp;   summary="Predict trip duration",

&nbsp;   description="Predict ETA for a trip using ML model (LightGBM)"

)

async def predict\_eta(request: ETARequest):

&nbsp;   """

&nbsp;   Predict trip ETA

&nbsp;   

&nbsp;   Returns predicted duration with 95% confidence interval.

&nbsp;   Average prediction time: < 50ms

&nbsp;   """

&nbsp;   # Check if model loaded

&nbsp;   if model\_loader is None:

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_503\_SERVICE\_UNAVAILABLE,

&nbsp;           detail="ML models not initialized. Contact support."

&nbsp;       )

&nbsp;   

&nbsp;   try:

&nbsp;       # Convert FastAPI request to ML request

&nbsp;       ml\_request = MLRequest(

&nbsp;           distance\_km=request.distance\_km,

&nbsp;           num\_points=request.num\_points,

&nbsp;           start\_lon=request.start\_lon,

&nbsp;           start\_lat=request.start\_lat,

&nbsp;           end\_lon=request.end\_lon,

&nbsp;           end\_lat=request.end\_lat,

&nbsp;           hour=request.hour,

&nbsp;           day\_of\_week=request.day\_of\_week,

&nbsp;           is\_weekend=1 if request.day\_of\_week >= 5 else 0,

&nbsp;           is\_rush\_hour=request.is\_rush\_hour,

&nbsp;           avg\_speed\_kph=request.avg\_speed\_kph

&nbsp;       )

&nbsp;       

&nbsp;       # Get prediction

&nbsp;       response = model\_loader.predict\_eta(ml\_request)

&nbsp;       

&nbsp;       logger.info(

&nbsp;           f"ETA prediction: {response.predicted\_duration\_minutes:.1f} min "

&nbsp;           f"(dist: {request.distance\_km:.1f} km, speed: {request.avg\_speed\_kph:.1f} km/h)"

&nbsp;       )

&nbsp;       

&nbsp;       # Return JSON response

&nbsp;       return response.to\_dict()

&nbsp;       

&nbsp;   except ValueError as e:

&nbsp;       # Validation error (bad input)

&nbsp;       logger.warning(f"Validation error: {e}")

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_400\_BAD\_REQUEST,

&nbsp;           detail=f"Invalid input: {str(e)}"

&nbsp;       )

&nbsp;   except Exception as e:

&nbsp;       # Unexpected error

&nbsp;       logger.error(f"Prediction failed: {e}")

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_500\_INTERNAL\_SERVER\_ERROR,

&nbsp;           detail=f"Prediction failed: {str(e)}"

&nbsp;       )





@router.get(

&nbsp;   "/model-info",

&nbsp;   status\_code=status.HTTP\_200\_OK,

&nbsp;   summary="Get model information",

&nbsp;   description="Returns ML model metadata and performance metrics"

)

async def get\_model\_info():

&nbsp;   """Get ML model information"""

&nbsp;   if model\_loader is None:

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_503\_SERVICE\_UNAVAILABLE,

&nbsp;           detail="ML models not initialized"

&nbsp;       )

&nbsp;   

&nbsp;   return model\_loader.get\_model\_info()





@router.get(

&nbsp;   "/health",

&nbsp;   status\_code=status.HTTP\_200\_OK,

&nbsp;   summary="ML health check",

&nbsp;   description="Check if ML models are loaded and operational"

)

async def health\_check():

&nbsp;   """Health check for ML models"""

&nbsp;   if model\_loader is None:

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_503\_SERVICE\_UNAVAILABLE,

&nbsp;           detail="ML models not initialized"

&nbsp;       )

&nbsp;   

&nbsp;   health = model\_loader.health\_check()

&nbsp;   

&nbsp;   if not health\['healthy']:

&nbsp;       raise HTTPException(

&nbsp;           status\_code=status.HTTP\_503\_SERVICE\_UNAVAILABLE,

&nbsp;           detail="ML models not healthy. Check logs for details."

&nbsp;       )

&nbsp;   

&nbsp;   return health

```



\*\*Save as:\*\* `app/routers/eta.py`



---



\### Step 3: Register Router in Main App



Edit `app/main.py`:

```python

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

import logging



\# Import routers

from app.routers import eta  # Add this line



\# Configure logging

logging.basicConfig(

&nbsp;   level=logging.INFO,

&nbsp;   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)



\# Create FastAPI app

app = FastAPI(

&nbsp;   title="D-Nerve API",

&nbsp;   description="Backend API for Cairo informal transit network",

&nbsp;   version="1.0.0"

)



\# CORS middleware

app.add\_middleware(

&nbsp;   CORSMiddleware,

&nbsp;   allow\_origins=\["\*"],  # Adjust for production

&nbsp;   allow\_credentials=True,

&nbsp;   allow\_methods=\["\*"],

&nbsp;   allow\_headers=\["\*"],

)



\# Register routers

app.include\_router(eta.router)  # Add this line



@app.get("/")

async def root():

&nbsp;   return {"message": "D-Nerve API v1.0.0"}

```



---



\### Step 4: Update Model Loader Path (if needed)



If models are in different location, edit `app/ml/model\_loader.py` line 248:

```python

\# Change from:

self.model\_dir = project\_root / "outputs" / "eta\_model"



\# To:

self.model\_dir = Path(\_\_file\_\_).parent / "models"

```



---



\## Testing



\### Test 1: Run Backend

```bash

\# In d-nerve-backend directory

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

```



\*\*Expected output:\*\*

---



\### Test 2: Health Check

```bash

curl http://localhost:8000/api/v1/health

```



\*\*Expected response:\*\*

```json

{

&nbsp; "healthy": true,

&nbsp; "checks": {

&nbsp;   "eta\_model\_exists": true,

&nbsp;   "eta\_model\_loadable": true,

&nbsp;   "routes\_data\_exists": true,

&nbsp;   "sample\_prediction": true

&nbsp; },

&nbsp; "timestamp": "2025-12-01T10:30:45.123456Z"

}

```



---



\### Test 3: Model Info

```bash

curl http://localhost:8000/api/v1/model-info

```



\*\*Expected response:\*\*

```json

{

&nbsp; "model\_name": "LightGBM ETA Predictor",

&nbsp; "version": "1.0.0",

&nbsp; "mae\_minutes": 9.04,

&nbsp; "r2\_score": 0.9513,

&nbsp; ...

}

```



---



\### Test 4: Predict ETA

```bash

curl -X POST "http://localhost:8000/api/v1/predict-eta" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{

&nbsp;   "distance\_km": 12.5,

&nbsp;   "start\_lon": 31.2357,

&nbsp;   "start\_lat": 30.0444,

&nbsp;   "end\_lon": 31.3387,

&nbsp;   "end\_lat": 30.0626,

&nbsp;   "hour": 8,

&nbsp;   "day\_of\_week": 1,

&nbsp;   "avg\_speed\_kph": 22.0,

&nbsp;   "is\_rush\_hour": 1

&nbsp; }'

```



\*\*Expected response:\*\*

```json

{

&nbsp; "predicted\_duration\_minutes": 34.12,

&nbsp; "confidence\_interval": {

&nbsp;   "lower": 16.04,

&nbsp;   "upper": 52.20

&nbsp; },

&nbsp; "model\_version": "1.0.0",

&nbsp; "timestamp": "2025-12-01T08:00:00.000000Z"

}

```



---



\### Test 5: Postman Testing



1\. \*\*Import Collection:\*\* See `API\_SPECIFICATION.md` for Postman JSON

2\. \*\*Test Endpoints:\*\*

&nbsp;  - GET `/health` → Should return healthy

&nbsp;  - GET `/model-info` → Should return model metadata

&nbsp;  - POST `/predict-eta` → Should return prediction



3\. \*\*Test Error Cases:\*\*

&nbsp;  - Invalid distance (negative) → 400 error

&nbsp;  - Invalid coordinates → 400 error

&nbsp;  - Invalid hour (>23) → 400 error



---



\## Deployment



\### Production Checklist



\- \[ ] Update coordinate validation for Cairo (29-31°N, 31-32°E)

\- \[ ] Retrain model with Cairo data (when available)

\- \[ ] Set up logging (Sentry/CloudWatch)

\- \[ ] Configure CORS for production domains

\- \[ ] Add rate limiting

\- \[ ] Add API authentication (JWT)

\- \[ ] Set up monitoring (Prometheus/Grafana)

\- \[ ] Load test (target: 100 req/sec)

\- \[ ] Deploy to cloud (AWS/Azure/Heroku)



---



\## Troubleshooting



\### Issue 1: "FileNotFoundError: Model not found"



\*\*Symptoms:\*\*

\*\*Solution:\*\*

```bash

\# Check file exists

ls app/ml/models/lightgbm\_eta\_model.pkl



\# If not, copy from ML repo

cp ../d-nerve-ml-models/outputs/eta\_model/lightgbm\_eta\_model.pkl app/ml/models/

```



---



\### Issue 2: "ModuleNotFoundError: No module named 'lightgbm'"



\*\*Symptoms:\*\*

\*\*Solution:\*\*

```bash

pip install lightgbm pandas numpy

```



---



\### Issue 3: Health Check Fails



\*\*Symptoms:\*\*

```json

{

&nbsp; "healthy": false,

&nbsp; "checks": {

&nbsp;   "eta\_model\_loadable": false

&nbsp; }

}

```



\*\*Solution:\*\*

1\. Check model file exists and has correct permissions

2\. Check Python version (need 3.11+)

3\. Check lightgbm version (need 4.0+)

4\. Check logs for detailed error



---



\### Issue 4: Prediction Returns 500 Error



\*\*Symptoms:\*\*

```json

{

&nbsp; "detail": "Prediction failed: ..."

}

```



\*\*Solution:\*\*

1\. Check input validation (use API spec schema)

2\. Check logs for stack trace

3\. Test model loader directly:

```python

&nbsp;  from app.ml.model\_loader import DNerveModelLoader

&nbsp;  loader = DNerveModelLoader()

&nbsp;  health = loader.health\_check()

&nbsp;  print(health)

```



---



\### Issue 5: Coordinate Validation Fails (Cairo vs Beijing)



\*\*Symptoms:\*\*

```json

{

&nbsp; "detail": "Invalid start coordinates: (30.0444, 31.2357)"

}

```



\*\*Solution:\*\*



Edit `app/ml/model\_loader.py`, update validation bounds:

```python

\# Line ~106-111, change from Beijing to Cairo:

if not (29.0 <= self.start\_lat <= 31.0 and 31.0 <= self.start\_lon <= 32.0):

&nbsp;   return False, f"Invalid start coordinates: ({self.start\_lat}, {self.start\_lon})"



if not (29.0 <= self.end\_lat <= 31.0 and 31.0 <= self.end\_lon <= 32.0):

&nbsp;   return False, f"Invalid end coordinates: ({self.end\_lat}, {self.end\_lon})"

```



---



\## Performance Optimization



\### Caching Predictions



For frequently requested routes, add caching:

```python

from functools import lru\_cache



@lru\_cache(maxsize=1000)

def get\_cached\_prediction(distance, start\_lon, start\_lat, end\_lon, end\_lat, hour):

&nbsp;   # ... prediction logic

&nbsp;   pass

```



\### Load Testing

```bash

\# Install Apache Bench

sudo apt-get install apache2-utils  # Linux

\# brew install ab  # Mac



\# Test 1000 requests, 10 concurrent

ab -n 1000 -c 10 -p request.json -T application/json \\

&nbsp; http://localhost:8000/api/v1/predict-eta

```



\*\*Target:\*\* > 100 requests/second



---



\## Next Steps



1\. \*\*Integrate with Database:\*\*

&nbsp;  - Store predictions for analytics

&nbsp;  - Cache common routes



2\. \*\*Add Authentication:\*\*

&nbsp;  - JWT tokens for mobile apps

&nbsp;  - API keys for partners



3\. \*\*Mobile Integration:\*\*

&nbsp;  - Provide API endpoint URLs to Group 3

&nbsp;  - Test with Android apps



4\. \*\*Monitoring:\*\*

&nbsp;  - Set up Sentry for error tracking

&nbsp;  - Set up Prometheus for metrics



5\. \*\*Documentation:\*\*

&nbsp;  - Generate Swagger docs (auto with FastAPI)

&nbsp;  - Share API docs with mobile team



---



\## Support



\*\*Questions?\*\*

\- ML Team (Group 2): For model-related issues

\- Backend Team Lead: For integration issues

\- GitHub Issues: For bug reports



\*\*Resources:\*\*

\- API Specification: `API\_SPECIFICATION.md`

\- Example Usage: `backend\_integration/examples/example\_usage.py`

\- ML Repo: https://github.com/d-nerve-cairo/d-nerve-ml-models



---



\*\*Last Updated:\*\* December 1, 2025  

\*\*Version:\*\* 1.0.0  

\*\*Teams:\*\* Group 1 (Backend) + Group 2 (ML)

