# D-Nerve ML Models

Machine learning models for informal transit route discovery and ETA prediction.

## Models
1. **Route Discovery:** DBSCAN clustering (F1 ≥ 0.85)
2. **ETA Prediction:** LightGBM regression (MAE ≤ 3 minutes)

## Tech Stack
- Python 3.11
- scikit-learn (DBSCAN)
- LightGBM
- Pandas, NumPy, GeoPandas

## Setup
```bash
conda create -n dnervenv python=3.11
conda activate dnervenv
pip install -r requirements.txt
jupyter notebook
```

## Notebooks
- `01_data_exploration.ipynb` - EDA
- `02_preprocessing.ipynb` - Cleaning pipeline
- `03_route_discovery.ipynb` - DBSCAN clustering
- `04_eta_prediction.ipynb` - LightGBM training

## Team
Group 2: [Member Names]

## License
MIT
