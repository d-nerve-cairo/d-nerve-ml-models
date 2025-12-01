\# Backend Integration Package



\*\*Ready-to-use ML model loader for D-Nerve backend team\*\*



This package contains everything Group 1 (Backend) needs to integrate ML models into the FastAPI backend.



---



\##  What's in This Package

```

backend\_integration/

├── model\_loader.py          ← Core model loader (COPY THIS)

├── README.md                ← This file

├── examples/

│   └── example\_usage.py     ← Usage examples

└── tests/

&nbsp;   └── test\_model\_loader.py ← Unit tests

```



---



\##  Quick Start (5 Minutes)



\### Step 1: Test the Model Loader

```bash

\# Make sure you're in ML repo and environment is activated

cd %USERPROFILE%\\Projects\\d-nerve-ml-models

conda activate dnervenv



\# Run the demo

python backend\_integration\\model\_loader.py

```



\*\*Expected output:\*\*

```

======================================================================

D-NERVE MODEL LOADER - DEMO

======================================================================



&nbsp;Model Information:

&nbsp; model\_name: LightGBM ETA Predictor

&nbsp; version: 1.0.0

&nbsp; mae\_minutes: 9.04

&nbsp; r2\_score: 0.9513

&nbsp; ...



&nbsp; Health Check:

&nbsp; Status:  Healthy

&nbsp; ...



&nbsp;Sample Prediction:

&nbsp; Predicted Duration: 25.3 minutes

&nbsp; ...



&nbsp;Demo complete!

```



---



\### Step 2: Run Examples

```bash

python backend\_integration\\examples\\example\_usage.py

```



\*\*This shows:\*\*

\- Basic prediction

\- Simplified function usage

\- Batch predictions

\- Error handling

\- FastAPI integration pattern



---



\### Step 3: Run Tests

```bash

\# Install pytest if not already installed

pip install pytest



\# Run tests

python -m pytest backend\_integration\\tests\\test\_model\_loader.py -v

```



---



\##  For Backend Team (Group 1)



\### Files You Need to Copy



\*\*From `d-nerve-ml-models` repository:\*\*



1\. \*\*Model Loader:\*\*

```

&nbsp;  backend\_integration/model\_loader.py

&nbsp;  → Copy to: d-nerve-backend/app/ml/model\_loader.py

```



2\. \*\*Trained Models:\*\*

```

&nbsp;  outputs/eta\_model/lightgbm\_eta\_model.pkl

&nbsp;  → Copy to: d-nerve-backend/app/ml/models/lightgbm\_eta\_model.pkl

&nbsp;  

&nbsp;  outputs/route\_discovery/route\_discovery\_results.pkl

&nbsp;  → Copy to: d-nerve-backend/app/ml/models/route\_discovery\_results.pkl

```



---



\### Integration Steps



\*\*1. Clone ML Repository (if not done):\*\*

```bash

git clone https://github.com/d-nerve-cairo/d-nerve-ml-models.git

cd d-nerve-ml-models

```



\*\*2. Copy Files to Your Backend Repo:\*\*

```bash

\# In your d-nerve-backend repo

mkdir -p app/ml/models



\# Copy model loader

cp ../d-nerve-ml-models/backend\_integration/model\_loader.py app/ml/



\# Copy trained models

cp ../d-nerve-ml-models/outputs/eta\_model/lightgbm\_eta\_model.pkl app/ml/models/

cp ../d-nerve-ml-models/outputs/route\_discovery/route\_discovery\_results.pkl app/ml/models/

```



\*\*3. Create FastAPI Endpoint:\*\*

```python

\# In d-nerve-backend/app/routers/eta.py



from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from app.ml.model\_loader import DNerveModelLoader, PredictionRequest



router = APIRouter(prefix="/api/v1", tags=\["ETA"])



\# Initialize model loader (singleton - loads once)

model\_loader = DNerveModelLoader()



class ETARequest(BaseModel):

&nbsp;   distance\_km: float

&nbsp;   start\_lon: float

&nbsp;   start\_lat: float

&nbsp;   end\_lon: float

&nbsp;   end\_lat: float

&nbsp;   hour: int

&nbsp;   day\_of\_week: int

&nbsp;   avg\_speed\_kph: float

&nbsp;   num\_points: int = 30

&nbsp;   is\_rush\_hour: int = 0



@router.post("/predict-eta")

async def predict\_eta(request: ETARequest):

&nbsp;   """Predict trip ETA"""

&nbsp;   try:

&nbsp;       ml\_request = PredictionRequest(

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

&nbsp;       response = model\_loader.predict\_eta(ml\_request)

&nbsp;       return response.to\_dict()

&nbsp;       

&nbsp;   except ValueError as e:

&nbsp;       raise HTTPException(status\_code=400, detail=str(e))

&nbsp;   except Exception as e:

&nbsp;       raise HTTPException(status\_code=500, detail=str(e))



@router.get("/health")

async def health\_check():

&nbsp;   """ML model health check"""

&nbsp;   return model\_loader.health\_check()

```



\*\*4. Register Router in Main App:\*\*

```python

\# In d-nerve-backend/app/main.py



from app.routers import eta



app = FastAPI(title="D-Nerve API")



app.include\_router(eta.router)

```



\*\*5. Test with Postman:\*\*

```bash

\# Start backend

uvicorn app.main:app --reload



\# Test health

GET http://localhost:8000/api/v1/health



\# Test prediction

POST http://localhost:8000/api/v1/predict-eta

Body:

{

&nbsp; "distance\_km": 12.5,

&nbsp; "start\_lon": 116.4,

&nbsp; "start\_lat": 39.9,

&nbsp; "end\_lon": 116.5,

&nbsp; "end\_lat": 40.0,

&nbsp; "hour": 8,

&nbsp; "day\_of\_week": 1,

&nbsp; "avg\_speed\_kph": 22.0,

&nbsp; "is\_rush\_hour": 1

}

```



---



\##  Documentation



\- \*\*Full Integration Guide:\*\* See `../docs/BACKEND\_INTEGRATION\_GUIDE.md`

\- \*\*API Specification:\*\* See `../docs/API\_SPECIFICATION.md`

\- \*\*Usage Examples:\*\* See `examples/example\_usage.py`

\- \*\*Tests:\*\* See `tests/test\_model\_loader.py`



---



\##  Troubleshooting



\### Error: "FileNotFoundError: ETA model not found"



\*\*Solution:\*\* Ensure model files are in correct location:

```

d-nerve-backend/

└── app/

&nbsp;   └── ml/

&nbsp;       ├── model\_loader.py

&nbsp;       └── models/

&nbsp;           ├── lightgbm\_eta\_model.pkl  ← Check this exists

&nbsp;           └── route\_discovery\_results.pkl

```



\### Error: "ModuleNotFoundError: No module named 'lightgbm'"



\*\*Solution:\*\* Install dependencies:

```bash

pip install lightgbm pandas numpy

```



\### Error: Invalid coordinates



\*\*Solution:\*\* Adjust coordinate validation in `model\_loader.py` line 106-111 for Cairo bounds:

```python

\# Change from Beijing (39-41°N, 115-118°E)

\# To Cairo (29-31°N, 31-32°E)

if not (29.0 <= self.start\_lat <= 31.0 and 31.0 <= self.start\_lon <= 32.0):

```



---



\##  Support



\*\*Questions?\*\*

\- Create issue on GitHub: https://github.com/d-nerve-cairo/d-nerve-ml-models/issues

\- Contact ML Team (Group 2)

\- Check full documentation in `docs/`



---



\*\*Last Updated:\*\* December 1, 2025  

\*\*Version:\*\* 1.0.0  

\*\*ML Team:\*\* Group 2

