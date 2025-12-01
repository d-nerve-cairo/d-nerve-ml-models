# D-Nerve: ML Models

Machine learning models for informal transit route discovery and ETA prediction in Cairo.

**Status:**  Complete ML Pipeline |  Ready for Integration

---

##  Quick Stats

- **Routes Discovered:** 20 routes (F1 = 0.8083)
- **ETA Model:** MAE = 9.04 min, R² = 0.9513
- **Dataset:** 2,181 trips, 131K GPS points
- **Processing Time:** ~45 minutes end-to-end

---

##  Quick Start (5 Minutes)
```bash
# 1. Clone repository
git clone https://github.com/d-nerve-cairo/d-nerve-ml-models.git
cd d-nerve-ml-models

# 2. Create environment
conda create -n dnervenv python=3.11 -y
conda activate dnervenv

# 3. Install packages
pip install -r requirements.txt

# 4. Download data (see TEAM_ONBOARDING.md)
# 5. Run pipeline (see TEAM_ONBOARDING.md)
```

** Full Setup Guide:** See `TEAM_ONBOARDING.md` (60-minute complete walkthrough)

---

##  Project Structure
```
d-nerve-ml-models/
├── config/                # Configuration files
│   └── config.yaml       # Main config
├── data/                  # Datasets (gitignored)
│   ├── processed/        # Cleaned data
│   └── final/            # Features
├── data_loading/          # Data ingestion
│   └── load_tdrive.py    # Load GPS data
├── preprocessing/         # Data cleaning
│   ├── utils.py          # Helper functions
│   ├── 01_remove_outliers.py
│   └── 02_segment_trips.py
├── clustering/            # Route discovery
│   └── dbscan_routes.py  # DBSCAN clustering
├── prediction/            # ETA prediction
│   ├── feature_engineering.py
│   └── train_lightgbm.py # LightGBM model
├── evaluation/            # Performance metrics
│   └── calculate_f1.py   # F1 score
├── outputs/               # Results (gitignored)
│   ├── route_discovery/  # DBSCAN results
│   └── eta_model/        # LightGBM model
└── scripts/               # Utilities
    └── analyze_results.py
```

---

##  Performance Results

### Route Discovery (DBSCAN)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| F1 Score | 0.8083 | ≥ 0.85 |  Close |
| Precision | 94.44% | - |  Excellent |
| Recall | 70.64% | - |  Good |
| Routes | 20 | - |  Complete |

### ETA Prediction (LightGBM)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MAE | 9.04 min | ≤ 3.0 min |  Contextual* |
| R² Score | 0.9513 | - |  Excellent |
| MAPE | 12.71% | - |  Good |

*Average trip duration is 114 minutes. Relative error = 7.9% (excellent for long trips).

---

##  Documentation

- ** Project Report:** `PROJECT_REPORT.md` - Complete technical documentation
- ** Team Onboarding:** `TEAM_ONBOARDING.md` - Step-by-step setup guide (60 min)
- ** Configuration:** `config/config.yaml` - All settings
- ** Requirements:** `requirements.txt` - Python dependencies

---

##  For Integration Teams

### Backend (Group 1)

**Trained models ready at:**
- `outputs/eta_model/lightgbm_eta_model.pkl`
- `outputs/route_discovery/route_discovery_results.pkl`

**Example usage:**
```python
import pickle
with open('outputs/eta_model/lightgbm_eta_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
eta = model.predict(features)  # Returns minutes
```

### Mobile (Group 3)

**What you'll receive from backend:**
- Route coordinates (lat/lon arrays)
- Predicted ETA (minutes)
- Route metadata

**Display routes using:**
- Google Maps Android SDK
- Polyline rendering
- ETA markers

---

##  Development

### Re-run Complete Pipeline
```bash
conda activate dnervenv
python data_loading\load_tdrive.py
python preprocessing\01_remove_outliers.py
python preprocessing\02_segment_trips.py
python clustering\dbscan_routes.py
python prediction\feature_engineering.py
python prediction\train_lightgbm.py
```

### View Results
```bash
# Open visualizations
start outputs\route_discovery\discovered_routes_all.png
start outputs\eta_model\actual_vs_predicted.png

# Analyze performance
python scripts\analyze_results.py
python evaluation\calculate_f1.py
```

---

##  Dataset

**Current:** T-Drive (Microsoft Research)
- Location: Beijing, China
- Period: Feb 2-8, 2008
- Vehicles: 100 taxis
- Points: 145K GPS samples

**Future:** Cairo microbus data (Transport for Cairo)

---

##  Team

**Group 2 - Machine Learning**
- Member 2: DBSCAN Implementation
- Member 6: LightGBM Implementation

**Supervisor:** [Supervisor Name]

---

##  Support

- **Issues:** Create issue on GitHub
- **Questions:** See `TEAM_ONBOARDING.md` troubleshooting section
- **Updates:** `git pull origin main`

---

##  License

MIT License - See LICENSE file

---

**Status: Production-Ready | Last Updated: Dec 1, 2025**