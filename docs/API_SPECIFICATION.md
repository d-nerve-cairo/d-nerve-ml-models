\# D-Nerve ML API Specification



\*\*Version:\*\* 1.0.0  

\*\*Last Updated:\*\* December 1, 2025  

\*\*Team:\*\* Group 2 - Machine Learning



This document defines the exact API contract between ML models and backend.



---



\## Table of Contents



1\. \[Overview](#overview)

2\. \[Endpoints](#endpoints)

3\. \[Request/Response Schemas](#requestresponse-schemas)

4\. \[Error Codes](#error-codes)

5\. \[Examples](#examples)



---



\## Overview



\### Base URL

```

Production: https://api.d-nerve.com/api/v1

Development: http://localhost:8000/api/v1

```



\### Authentication



Currently no authentication required (add JWT later if needed).



\### Content Type



All requests and responses use `application/json`.



---



\## Endpoints



\### 1. Predict ETA



Predict trip duration based on distance, time, and traffic conditions.



\*\*Endpoint:\*\* `POST /predict-eta`



\*\*Request Body:\*\*

```json

{

&nbsp; "distance\_km": 12.5,

&nbsp; "start\_lon": 116.4,

&nbsp; "start\_lat": 39.9,

&nbsp; "end\_lon": 116.5,

&nbsp; "end\_lat": 40.0,

&nbsp; "hour": 8,

&nbsp; "day\_of\_week": 1,

&nbsp; "avg\_speed\_kph": 22.0,

&nbsp; "num\_points": 35,

&nbsp; "is\_rush\_hour": 1

}

```



\*\*Field Descriptions:\*\*



| Field                       | Type      | Required | Range    | Description                                |

|--------------------------|------------|-------------|--------------|-------------------------------------------|

| `distance\_km`       | float      | Yes         | 0-200       | Trip distance in kilometers         |

| `start\_lon`              | float      | Yes         | 115-118\* | Starting longitude                      |

| `start\_lat`               | float      | Yes         | 39-41\*     | Starting latitude                         |

| `end\_lon`              | float      | Yes         | 115-118\* | Ending longitude                       |

| `end\_lat`               | float      | Yes         | 39-41\*     | Ending latitude                          |

| `hour`                    | integer  | Yes         | 0-23         | Hour of day (24-hour format)  |

| `day\_of\_week`     | integer  | Yes         | 0-6           | Day (0=Monday, 6=Sunday)    |

| `avg\_speed\_kph` | float      | Yes         | 0-200       | Expected average speed        |

| `num\_points`        | integer  | No           | 10-1000  | GPS points (default: 30)            |

| `is\_rush\_hour`       | integer  | No           | 0 or 1       | Rush hour flag (default: 0)       |



\*Adjust coordinate ranges for Cairo: lat 29-31, lon 31-32



\*\*Success Response (200 OK):\*\*

```json

{

&nbsp; "predicted\_duration\_minutes": 25.34,

&nbsp; "confidence\_interval": {

&nbsp;   "lower": 7.26,

&nbsp;   "upper": 43.42

&nbsp; },

&nbsp; "model\_version": "1.0.0",

&nbsp; "timestamp": "2025-12-01T10:30:45.123456Z"

}

```



\*\*Error Response (400 Bad Request):\*\*

```json

{

&nbsp; "detail": "Invalid distance: -10.0 km (must be 0-200)"

}

```



\*\*Error Response (500 Internal Server Error):\*\*

```json

{

&nbsp; "detail": "Prediction failed: Model not loaded"

}

```



---



\### 2. Get Model Information



Get ML model metadata and performance metrics.



\*\*Endpoint:\*\* `GET /model-info`



\*\*Request:\*\* None (GET request, no body)



\*\*Success Response (200 OK):\*\*

```json

{

&nbsp; "model\_name": "LightGBM ETA Predictor",

&nbsp; "version": "1.0.0",

&nbsp; "mae\_minutes": 9.04,

&nbsp; "r2\_score": 0.9513,

&nbsp; "training\_date": "2025-12-01",

&nbsp; "feature\_count": 13,

&nbsp; "status": "loaded",

&nbsp; "routes\_available": true

}

```



\*\*Field Descriptions:\*\*



| Field | Type | Description |

|-------|------|-------------|

| `model\_name` | string | Model name |

| `version` | string | Model version |

| `mae\_minutes` | float | Mean Absolute Error (test set) |

| `r2\_score` | float | R² score (0-1, higher is better) |

| `training\_date` | string | Training date (YYYY-MM-DD) |

| `feature\_count` | integer | Number of input features |

| `status` | string | "loaded" or "not\_loaded" |

| `routes\_available` | boolean | Route data available |



---



\### 3. Health Check



Check if ML models are operational.



\*\*Endpoint:\*\* `GET /health`



\*\*Request:\*\* None (GET request, no body)



\*\*Success Response (200 OK - Healthy):\*\*

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



\*\*Error Response (503 Service Unavailable - Unhealthy):\*\*

```json

{

&nbsp; "healthy": false,

&nbsp; "checks": {

&nbsp;   "eta\_model\_exists": false,

&nbsp;   "eta\_model\_loadable": false,

&nbsp;   "eta\_model\_error": "FileNotFoundError: Model not found",

&nbsp;   "routes\_data\_exists": true,

&nbsp;   "sample\_prediction": false,

&nbsp;   "prediction\_error": "Model not loaded"

&nbsp; },

&nbsp; "timestamp": "2025-12-01T10:30:45.123456Z"

}

```



---



\## Request/Response Schemas



\### PredictionRequest Schema

```json

{

&nbsp; "type": "object",

&nbsp; "required": \[

&nbsp;   "distance\_km",

&nbsp;   "start\_lon",

&nbsp;   "start\_lat",

&nbsp;   "end\_lon",

&nbsp;   "end\_lat",

&nbsp;   "hour",

&nbsp;   "day\_of\_week",

&nbsp;   "avg\_speed\_kph"

&nbsp; ],

&nbsp; "properties": {

&nbsp;   "distance\_km": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 0,

&nbsp;     "maximum": 200,

&nbsp;     "example": 12.5

&nbsp;   },

&nbsp;   "start\_lon": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 115,

&nbsp;     "maximum": 118,

&nbsp;     "example": 116.4

&nbsp;   },

&nbsp;   "start\_lat": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 39,

&nbsp;     "maximum": 41,

&nbsp;     "example": 39.9

&nbsp;   },

&nbsp;   "end\_lon": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 115,

&nbsp;     "maximum": 118,

&nbsp;     "example": 116.5

&nbsp;   },

&nbsp;   "end\_lat": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 39,

&nbsp;     "maximum": 41,

&nbsp;     "example": 40.0

&nbsp;   },

&nbsp;   "hour": {

&nbsp;     "type": "integer",

&nbsp;     "minimum": 0,

&nbsp;     "maximum": 23,

&nbsp;     "example": 8

&nbsp;   },

&nbsp;   "day\_of\_week": {

&nbsp;     "type": "integer",

&nbsp;     "minimum": 0,

&nbsp;     "maximum": 6,

&nbsp;     "example": 1

&nbsp;   },

&nbsp;   "avg\_speed\_kph": {

&nbsp;     "type": "number",

&nbsp;     "minimum": 0,

&nbsp;     "maximum": 200,

&nbsp;     "example": 22.0

&nbsp;   },

&nbsp;   "num\_points": {

&nbsp;     "type": "integer",

&nbsp;     "minimum": 10,

&nbsp;     "maximum": 1000,

&nbsp;     "default": 30,

&nbsp;     "example": 35

&nbsp;   },

&nbsp;   "is\_rush\_hour": {

&nbsp;     "type": "integer",

&nbsp;     "enum": \[0, 1],

&nbsp;     "default": 0,

&nbsp;     "example": 1

&nbsp;   }

&nbsp; }

}

```



\### PredictionResponse Schema

```json

{

&nbsp; "type": "object",

&nbsp; "properties": {

&nbsp;   "predicted\_duration\_minutes": {

&nbsp;     "type": "number",

&nbsp;     "example": 25.34

&nbsp;   },

&nbsp;   "confidence\_interval": {

&nbsp;     "type": "object",

&nbsp;     "properties": {

&nbsp;       "lower": {

&nbsp;         "type": "number",

&nbsp;         "example": 7.26

&nbsp;       },

&nbsp;       "upper": {

&nbsp;         "type": "number",

&nbsp;         "example": 43.42

&nbsp;       }

&nbsp;     }

&nbsp;   },

&nbsp;   "model\_version": {

&nbsp;     "type": "string",

&nbsp;     "example": "1.0.0"

&nbsp;   },

&nbsp;   "timestamp": {

&nbsp;     "type": "string",

&nbsp;     "format": "date-time",

&nbsp;     "example": "2025-12-01T10:30:45.123456Z"

&nbsp;   }

&nbsp; }

}

```



---



\## Error Codes



| Status Code | Error Type | Description | Solution |

|-------------|------------|-------------|----------|

| 400 | Bad Request | Invalid input parameters | Check request body against schema |

| 422 | Unprocessable Entity | Validation error | Fix data types/ranges |

| 500 | Internal Server Error | Model prediction failed | Check logs, verify model files |

| 503 | Service Unavailable | ML models not loaded | Run health check, restart service |



\### Common Error Messages



\*\*1. Invalid Distance:\*\*

```json

{

&nbsp; "detail": "Invalid distance: -10.0 km (must be 0-200)"

}

```

\*\*Solution:\*\* Ensure distance is positive and ≤ 200 km.



\*\*2. Invalid Coordinates:\*\*

```json

{

&nbsp; "detail": "Invalid start coordinates: (50.0, 120.0)"

}

```

\*\*Solution:\*\* Use coordinates within Beijing bounds (or Cairo for production).



\*\*3. Invalid Hour:\*\*

```json

{

&nbsp; "detail": "Invalid hour: 25 (must be 0-23)"

}

```

\*\*Solution:\*\* Use 24-hour format (0-23).



\*\*4. Model Not Loaded:\*\*

```json

{

&nbsp; "detail": "Prediction failed: Model not loaded"

}

```

\*\*Solution:\*\* Check model files exist in `app/ml/models/`. Run health check.



---



\## Examples



\### Example 1: Morning Commute (Rush Hour)



\*\*Request:\*\*

```bash

curl -X POST "http://localhost:8000/api/v1/predict-eta" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{

&nbsp;   "distance\_km": 12.5,

&nbsp;   "start\_lon": 116.3975,

&nbsp;   "start\_lat": 39.9087,

&nbsp;   "end\_lon": 116.4832,

&nbsp;   "end\_lat": 39.9897,

&nbsp;   "hour": 8,

&nbsp;   "day\_of\_week": 1,

&nbsp;   "avg\_speed\_kph": 22.0,

&nbsp;   "num\_points": 35,

&nbsp;   "is\_rush\_hour": 1

&nbsp; }'

```



\*\*Response:\*\*

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



\*\*Interpretation:\*\* Trip will take ~34 minutes (95% confidence: 16-52 min)



---



\### Example 2: Evening Trip (Non-Rush)



\*\*Request:\*\*

```bash

curl -X POST "http://localhost:8000/api/v1/predict-eta" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{

&nbsp;   "distance\_km": 8.0,

&nbsp;   "start\_lon": 116.4,

&nbsp;   "start\_lat": 39.9,

&nbsp;   "end\_lon": 116.5,

&nbsp;   "end\_lat": 40.0,

&nbsp;   "hour": 22,

&nbsp;   "day\_of\_week": 3,

&nbsp;   "avg\_speed\_kph": 45.0,

&nbsp;   "is\_rush\_hour": 0

&nbsp; }'

```



\*\*Response:\*\*

```json

{

&nbsp; "predicted\_duration\_minutes": 13.52,

&nbsp; "confidence\_interval": {

&nbsp;   "lower": 0.00,

&nbsp;   "upper": 31.60

&nbsp; },

&nbsp; "model\_version": "1.0.0",

&nbsp; "timestamp": "2025-12-01T22:00:00.000000Z"

}

```



\*\*Interpretation:\*\* Fast trip (~14 minutes) due to late hour and high speed.



---



\### Example 3: Weekend Short Trip



\*\*Request:\*\*

```bash

curl -X POST "http://localhost:8000/api/v1/predict-eta" \\

&nbsp; -H "Content-Type: application/json" \\

&nbsp; -d '{

&nbsp;   "distance\_km": 5.0,

&nbsp;   "start\_lon": 116.4,

&nbsp;   "start\_lat": 39.9,

&nbsp;   "end\_lon": 116.45,

&nbsp;   "end\_lat": 39.95,

&nbsp;   "hour": 14,

&nbsp;   "day\_of\_week": 6,

&nbsp;   "avg\_speed\_kph": 30.0

&nbsp; }'

```



\*\*Response:\*\*

```json

{

&nbsp; "predicted\_duration\_minutes": 10.23,

&nbsp; "confidence\_interval": {

&nbsp;   "lower": 0.00,

&nbsp;   "upper": 28.31

&nbsp; },

&nbsp; "model\_version": "1.0.0",

&nbsp; "timestamp": "2025-12-01T14:00:00.000000Z"

}

```



---



\### Example 4: Health Check



\*\*Request:\*\*

```bash

curl -X GET "http://localhost:8000/api/v1/health"

```



\*\*Response (Healthy):\*\*

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



\### Example 5: Model Info



\*\*Request:\*\*

```bash

curl -X GET "http://localhost:8000/api/v1/model-info"

```



\*\*Response:\*\*

```json

{

&nbsp; "model\_name": "LightGBM ETA Predictor",

&nbsp; "version": "1.0.0",

&nbsp; "mae\_minutes": 9.04,

&nbsp; "r2\_score": 0.9513,

&nbsp; "training\_date": "2025-12-01",

&nbsp; "feature\_count": 13,

&nbsp; "status": "loaded",

&nbsp; "routes\_available": true

}

```



---



\## Postman Collection



Import this JSON into Postman for easy testing:

```json

{

&nbsp; "info": {

&nbsp;   "name": "D-Nerve ML API",

&nbsp;   "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"

&nbsp; },

&nbsp; "item": \[

&nbsp;   {

&nbsp;     "name": "Predict ETA",

&nbsp;     "request": {

&nbsp;       "method": "POST",

&nbsp;       "header": \[

&nbsp;         {

&nbsp;           "key": "Content-Type",

&nbsp;           "value": "application/json"

&nbsp;         }

&nbsp;       ],

&nbsp;       "body": {

&nbsp;         "mode": "raw",

&nbsp;         "raw": "{\\n  \\"distance\_km\\": 12.5,\\n  \\"start\_lon\\": 116.4,\\n  \\"start\_lat\\": 39.9,\\n  \\"end\_lon\\": 116.5,\\n  \\"end\_lat\\": 40.0,\\n  \\"hour\\": 8,\\n  \\"day\_of\_week\\": 1,\\n  \\"avg\_speed\_kph\\": 22.0,\\n  \\"num\_points\\": 35,\\n  \\"is\_rush\_hour\\": 1\\n}"

&nbsp;       },

&nbsp;       "url": {

&nbsp;         "raw": "http://localhost:8000/api/v1/predict-eta",

&nbsp;         "protocol": "http",

&nbsp;         "host": \["localhost"],

&nbsp;         "port": "8000",

&nbsp;         "path": \["api", "v1", "predict-eta"]

&nbsp;       }

&nbsp;     }

&nbsp;   },

&nbsp;   {

&nbsp;     "name": "Health Check",

&nbsp;     "request": {

&nbsp;       "method": "GET",

&nbsp;       "url": {

&nbsp;         "raw": "http://localhost:8000/api/v1/health",

&nbsp;         "protocol": "http",

&nbsp;         "host": \["localhost"],

&nbsp;         "port": "8000",

&nbsp;         "path": \["api", "v1", "health"]

&nbsp;       }

&nbsp;     }

&nbsp;   },

&nbsp;   {

&nbsp;     "name": "Model Info",

&nbsp;     "request": {

&nbsp;       "method": "GET",

&nbsp;       "url": {

&nbsp;         "raw": "http://localhost:8000/api/v1/model-info",

&nbsp;         "protocol": "http",

&nbsp;         "host": \["localhost"],

&nbsp;         "port": "8000",

&nbsp;         "path": \["api", "v1", "model-info"]

&nbsp;       }

&nbsp;     }

&nbsp;   }

&nbsp; ]

}

```



---



\## Implementation Checklist



\*\*Backend Team (Group 1) - Use this checklist:\*\*



\- \[ ] Copy `model\_loader.py` to backend repo

\- \[ ] Copy model files (`.pkl`) to backend repo

\- \[ ] Create FastAPI router with 3 endpoints

\- \[ ] Implement request validation (Pydantic)

\- \[ ] Implement error handling (try-catch)

\- \[ ] Test all endpoints with Postman

\- \[ ] Update coordinate validation for Cairo

\- \[ ] Add logging for debugging

\- \[ ] Deploy to staging environment

\- \[ ] Integration test with mobile apps



---



\## Notes for Cairo Deployment



\*\*When switching from Beijing to Cairo data:\*\*



1\. \*\*Update coordinate validation in `model\_loader.py`:\*\*

```python

&nbsp;  # Change line 106-111 from:

&nbsp;  if not (39.0 <= self.start\_lat <= 41.0 and 115.0 <= self.start\_lon <= 118.0):

&nbsp;  

&nbsp;  # To:

&nbsp;  if not (29.0 <= self.start\_lat <= 31.0 and 31.0 <= self.start\_lon <= 32.0):

```



2\. \*\*Retrain model with Cairo data\*\*

3\. \*\*Update model metadata\*\*

4\. \*\*Test with Cairo coordinates\*\*



---



\## Support



\*\*Questions or issues?\*\*

\- Create issue: https://github.com/d-nerve-cairo/d-nerve-ml-models/issues

\- Contact: Group 2 - ML Team

\- Documentation: See `BACKEND\_INTEGRATION\_GUIDE.md`



---



\*\*Version History:\*\*



| Version | Date | Changes |

|---------|------|---------|

| 1.0.0 | 2025-12-01 | Initial API specification |



---



\*\*Last Updated:\*\* December 1, 2025  

\*\*Maintained by:\*\* Group 2 - ML Team

