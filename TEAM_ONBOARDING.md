\# D-Nerve ML Team Onboarding Guide

\*\*Complete Step-by-Step Setup for New Team Members\*\*



This guide will take you from zero to running the complete ML pipeline in 60 minutes.



---



\##  Prerequisites



Before starting, ensure you have:

\- \[ ] Windows 10/11 PC (or Mac/Linux - adjust commands accordingly)

\- \[ ] Anaconda installed (https://www.anaconda.com/download)

\- \[ ] Git installed (https://git-scm.com/downloads)

\- \[ ] GitHub account

\- \[ ] Internet connection (for downloads)

\- \[ ] At least 10 GB free disk space



---



\##  Time Estimate



| Phase | Time | Description |

|-------|------|-------------|

| Setup Environment | 15 min | Install Anaconda, clone repo |

| Download Data | 10 min | Get T-Drive dataset |

| Run Pipeline | 30 min | Execute all scripts |

| Explore Results | 5 min | View outputs |

| \*\*TOTAL\*\* | \*\*60 min\*\* | End-to-end setup |



---



\##  STEP 1: Install Required Software (15 minutes)



\### 1.1 Install Anaconda



\*\*If not already installed:\*\*



1\. Download Anaconda: https://www.anaconda.com/download

2\. Run installer (`Anaconda3-2024.XX-Windows-x86\_64.exe`)

3\. Follow wizard:

&nbsp;  -  Install for "Just Me"

&nbsp;  -  Accept default location

&nbsp;  -  Check "Add Anaconda3 to PATH" (optional but helpful)

4\. Wait for installation (5-10 minutes)



\*\*Verify installation:\*\*

```cmd

\# Open Command Prompt (Win + R, type "cmd", Enter)

conda --version

\# Should show: conda 24.X.X or similar

```



\### 1.2 Install Git



\*\*If not already installed:\*\*



1\. Download Git: https://git-scm.com/downloads

2\. Run installer

3\. Use default settings (just keep clicking "Next")



\*\*Verify installation:\*\*

```cmd

git --version

\# Should show: git version 2.XX.X

```



---



\##  STEP 2: Clone the Repository (5 minutes)



\### 2.1 Create Projects Folder



\*\*For Windows:\*\*

```cmd

\# Open Command Prompt

\# Navigate to your home directory

cd %USERPROFILE%



\# Create Projects folder

mkdir Projects

cd Projects

```



\*\*For Mac/Linux:\*\*

```bash

\# Open Terminal

cd ~

mkdir -p Projects

cd Projects

```



\*\* Note:\*\* Replace all future instances of `C:\\Users\\LENOVO` with `%USERPROFILE%` (Windows) or `~` (Mac/Linux)



\### 2.2 Clone ML Repository

```cmd

git clone https://github.com/d-nerve-cairo/d-nerve-ml-models.git

cd d-nerve-ml-models

```



\*\*You should see:\*\*

```

Cloning into 'd-nerve-ml-models'...

remote: Enumerating objects: XX, done.

remote: Counting objects: 100% (XX/XX), done.

...

```



\### 2.3 Verify Files



\*\*Windows:\*\*

```cmd

dir

```



\*\*Mac/Linux:\*\*

```bash

ls -la

```



\*\*You should see:\*\*

```

config/

data/

preprocessing/

clustering/

prediction/

evaluation/

scripts/

.gitignore

README.md

requirements.txt

```



---



\##  STEP 3: Setup Python Environment (5 minutes)



\### 3.1 Create Conda Environment

```cmd

\# Make sure you're in the project directory

\# Windows: cd %USERPROFILE%\\Projects\\d-nerve-ml-models

\# Mac/Linux: cd ~/Projects/d-nerve-ml-models



\# Create environment named "dnervenv"

conda create -n dnervenv python=3.11 -y

```



\*\*Wait for packages to download and install (~2 minutes)\*\*



\### 3.2 Activate Environment



\*\*Windows:\*\*

```cmd

conda activate dnervenv

```



\*\*Mac/Linux:\*\*

```bash

conda activate dnervenv

```



\*\*Your prompt should change to:\*\*

```

(dnervenv) \[your-path]\\d-nerve-ml-models>

```



\### 3.3 Install Required Packages

```cmd

\# Install core packages

conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyter -y



\# Install ML-specific packages

pip install lightgbm geopandas pyarrow optuna tqdm pyyaml

```



\*\*This takes 3-5 minutes\*\*



\### 3.4 Verify Installation

```cmd

python -c "import pandas, numpy, sklearn, lightgbm, geopandas; print(' All packages installed successfully!')"

```



\*\*Expected output:\*\*

```

&nbsp;All packages installed successfully!

```



---



\##  STEP 4: Download T-Drive Dataset (10 minutes)



\### 4.1 Create Data Directory



\*\*Windows:\*\*

```cmd

\# Create in your home directory

mkdir %USERPROFILE%\\d-nerve-data\\t-drive

cd %USERPROFILE%\\d-nerve-data\\t-drive

```



\*\*Mac/Linux:\*\*

```bash

mkdir -p ~/d-nerve-data/t-drive

cd ~/d-nerve-data/t-drive

```



\### 4.2 Download Dataset



\*\*Manual Download (Recommended):\*\*



1\. Open browser: https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

2\. Click \*\*"Download"\*\* button

3\. Save `release.zip` to your `d-nerve-data/t-drive/` folder

4\. \*\*Windows:\*\* Right-click `release.zip` → \*\*"Extract All"\*\*

&nbsp;  \*\*Mac:\*\* Double-click `release.zip`

5\. Extract to current folder



\### 4.3 Verify Extraction



\*\*Windows:\*\*

```cmd

dir %USERPROFILE%\\d-nerve-data\\t-drive\\release\\taxi\_log\_2008\_by\_id

```



\*\*Mac/Linux:\*\*

```bash

ls ~/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id | wc -l

```



\*\*You should see 10,357 text files:\*\*

```

1.txt

2.txt

3.txt

...

10357.txt

```



\### 4.4 Update Configuration File



\*\*IMPORTANT:\*\* Update the data path in config to match YOUR system:

```cmd

\# Navigate back to project

cd %USERPROFILE%\\Projects\\d-nerve-ml-models   # Windows

\# cd ~/Projects/d-nerve-ml-models              # Mac/Linux



\# Edit config file

notepad config\\config.yaml   # Windows

\# nano config/config.yaml    # Mac/Linux

```



\*\*Find this line:\*\*

```yaml

tdrive\_path: "C:/Users/LENOVO/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

```



\*\*Change to YOUR path:\*\*



\*\*Windows example:\*\*

```yaml

tdrive\_path: "C:/Users/YourUsername/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

```



\*\*Mac/Linux example:\*\*

```yaml

tdrive\_path: "/Users/yourname/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

```



\*\* Pro tip:\*\* Use forward slashes `/` even on Windows (works in Python)



\*\*Save and close\*\* (Ctrl+S in Notepad, Ctrl+X in nano)



---



\##  STEP 5: Run the Complete ML Pipeline (30 minutes)



Now execute each script in order. Each step builds on the previous one.



\### 5.1 Data Loading (2 minutes)

```cmd

\# Make sure environment is activated

conda activate dnervenv



\# Navigate to project

cd %USERPROFILE%\\Projects\\d-nerve-ml-models   # Windows

\# cd ~/Projects/d-nerve-ml-models              # Mac/Linux



\# Run data loading

\# Windows:

python data\_loading\\load\_tdrive.py



\# Mac/Linux:

\# python data\_loading/load\_tdrive.py

```



\*\*Expected output:\*\*

```

============================================================

T-DRIVE DATA LOADING

============================================================

Loading 100 taxis...

&nbsp;Loaded 100 taxis

&nbsp;Total GPS points: 145,582

&nbsp;Saved to data\\processed\\tdrive\_100taxis.parquet (2.25 MB)

&nbsp;DATA LOADING COMPLETE

```



\*\* Checkpoint:\*\* File created at `data/processed/tdrive\_100taxis.parquet`



---



\### 5.2 Remove Outliers (3 minutes)



\*\*Windows:\*\*

```cmd

python preprocessing\\01\_remove\_outliers.py

```



\*\*Mac/Linux:\*\*

```bash

python preprocessing/01\_remove\_outliers.py

```



\*\*Expected output:\*\*

```

============================================================

PREPROCESSING: OUTLIER REMOVAL

============================================================

Invalid (0,0) points removed: 1

Geographic outliers removed: 7,248

Duplicate timestamps removed: 6,697

Speed outliers removed: 439

&nbsp;Saved to data\\processed\\tdrive\_100taxis\_clean.parquet

&nbsp;CLEANING COMPLETE

Final points: 131,197

```



\*\* Checkpoint:\*\* File created at `data/processed/tdrive\_100taxis\_clean.parquet`



---



\### 5.3 Segment Trips (2 minutes)



\*\*Windows:\*\*

```cmd

python preprocessing\\02\_segment\_trips.py

```



\*\*Mac/Linux:\*\*

```bash

python preprocessing/02\_segment\_trips.py

```



\*\*Expected output:\*\*

```

============================================================

PREPROCESSING: TRIP SEGMENTATION

============================================================

Processing 99 taxis...

Total trips created: 2,181

Avg points per trip: 55.2

&nbsp;Saved to data\\processed\\tdrive\_100taxis\_trips.parquet

&nbsp;TRIP SEGMENTATION COMPLETE

```



\*\* Checkpoint:\*\* File created at `data/processed/tdrive\_100taxis\_trips.parquet`



---



\### 5.4 DBSCAN Route Discovery (15 minutes) ⏰



\*\*This is the longest step - be patient!\*\*



\*\*Windows:\*\*

```cmd

python clustering\\dbscan\_routes.py

```



\*\*Mac/Linux:\*\*

```bash

python clustering/dbscan\_routes.py

```



\*\*Expected output:\*\*

```

======================================================================

DBSCAN ROUTE DISCOVERY

======================================================================

Loading 2181 trips...

Computing distance matrix... (This may take 5-10 minutes)

Computing distances: 100%|████████████| 2377290/2377290 \[12:30<00:00]

&nbsp;Distance matrix complete



Running DBSCAN...

&nbsp;Clustering complete:

&nbsp;  Routes discovered: 20

&nbsp;  Noise points: 1996 (91.5%)



&nbsp;Visualizing routes...

&nbsp;Saved: outputs\\route\_discovery\\discovered\_routes\_all.png

&nbsp;Results saved to outputs\\route\_discovery\\route\_discovery\_results.pkl

======================================================================

&nbsp;ROUTE DISCOVERY COMPLETE

======================================================================

```



\*\* Checkpoint:\*\* Files created in `outputs/route\_discovery/`



---



\### 5.5 Feature Engineering (1 minute)



\*\*Windows:\*\*

```cmd

python prediction\\feature\_engineering.py

```



\*\*Mac/Linux:\*\*

```bash

python prediction/feature\_engineering.py

```



\*\*Expected output:\*\*

```

============================================================

FEATURE ENGINEERING FOR ETA PREDICTION

============================================================

&nbsp;Loaded 2181 trips

&nbsp;Extracted features from 1839 trips

&nbsp;Added route features

&nbsp;Features saved to data\\final\\trip\_features.parquet

&nbsp;FEATURE ENGINEERING COMPLETE

```



\*\* Checkpoint:\*\* File created at `data/final/trip\_features.parquet`



---



\### 5.6 Train LightGBM Model (3 minutes)



\*\*Windows:\*\*

```cmd

python prediction\\train\_lightgbm.py

```



\*\*Mac/Linux:\*\*

```bash

python prediction/train\_lightgbm.py

```



\*\*Expected output:\*\*

```

============================================================

LIGHTGBM ETA PREDICTION MODEL TRAINING

============================================================

&nbsp;Training set: 1471 trips

&nbsp;Test set: 368 trips



&nbsp;Training LightGBM model...

\[100]   train's l1: 7.12    test's l1: 11.71

\[200]   train's l1: 4.39    test's l1: 9.71

...

Early stopping, best iteration is:

\[675]   train's l1: 1.48    test's l1: 9.04



============================================================

MODEL EVALUATION RESULTS

============================================================

Mean Absolute Error (MAE):  9.04 minutes

R² Score:                    0.9513

============================================================



&nbsp;Model saved to outputs\\eta\_model\\lightgbm\_eta\_model.pkl

&nbsp;MODEL TRAINING COMPLETE

```



\*\* Checkpoint:\*\* Files created in `outputs/eta\_model/`



---



\### 5.7 Evaluate Results (1 minute)



\*\*Windows:\*\*

```cmd

python scripts\\analyze\_results.py

python evaluation\\calculate\_f1.py

```



\*\*Mac/Linux:\*\*

```bash

python scripts/analyze\_results.py

python evaluation/calculate\_f1.py

```



\*\*Expected output:\*\*

```

============================================================

DBSCAN ROUTE DISCOVERY ANALYSIS

============================================================

Routes Discovered: 20

Noise Points: 1996 (91.5%)

Top 10 Routes by Trip Count:

&nbsp;  1. route\_005:  22 trips

&nbsp;  2. route\_001:  17 trips

&nbsp;  ...

============================================================



============================================================

ROUTE DISCOVERY EVALUATION

============================================================

F1 Score:  0.8083 (80.83%)

Precision: 0.9444 (94.44%)

Recall:    0.7064 (70.64%)

============================================================

```



---



\##  STEP 6: View Results (5 minutes)



\### 6.1 Open Visualizations



\*\*Windows:\*\*

```cmd

start outputs\\route\_discovery\\discovered\_routes\_all.png

start outputs\\eta\_model\\actual\_vs\_predicted.png

start outputs\\eta\_model\\feature\_importance.png

explorer outputs

```



\*\*Mac:\*\*

```bash

open outputs/route\_discovery/discovered\_routes\_all.png

open outputs/eta\_model/actual\_vs\_predicted.png

open outputs/eta\_model/feature\_importance.png

open outputs

```



\*\*Linux:\*\*

```bash

xdg-open outputs/route\_discovery/discovered\_routes\_all.png

xdg-open outputs/eta\_model/actual\_vs\_predicted.png

nautilus outputs  # or your file manager

```



\### 6.2 Check Data Files



\*\*Windows:\*\*

```cmd

dir data\\processed

dir data\\final

dir outputs\\route\_discovery

dir outputs\\eta\_model

```



\*\*Mac/Linux:\*\*

```bash

ls -lh data/processed/

ls -lh data/final/

ls -lh outputs/route\_discovery/

ls -lh outputs/eta\_model/

```



---



\##  STEP 7: Understanding the Results



\### What You've Just Created:



1\. \*\*Cleaned GPS Data\*\* 

&nbsp;  - 131,197 clean GPS points from 100 taxis

&nbsp;  - 2,181 segmented trips



2\. \*\*Discovered Routes\*\*

&nbsp;  - 20 distinct routes identified

&nbsp;  - Visualizations showing route maps

&nbsp;  - F1 Score: 0.8083 (80.83% accuracy)



3\. \*\*ETA Prediction Model\*\*

&nbsp;  - LightGBM model trained on 1,839 trips

&nbsp;  - Can predict trip duration with 9.04-minute error

&nbsp;  - R² = 0.9513 (explains 95% of variance)



4\. \*\*Ready-to-Deploy Models\*\*

&nbsp;  - `outputs/route\_discovery/route\_discovery\_results.pkl`

&nbsp;  - `outputs/eta\_model/lightgbm\_eta\_model.pkl`



---



\##  STEP 8: Explore the Code



\### Key Files to Review:



\*\*Start here (easiest to hardest):\*\*



1\. \*\*`config/config.yaml`\*\* - All configuration settings

2\. \*\*`preprocessing/utils.py`\*\* - Helper functions (haversine distance, etc.)

3\. \*\*`data\_loading/load\_tdrive.py`\*\* - How data is loaded

4\. \*\*`preprocessing/01\_remove\_outliers.py`\*\* - Data cleaning logic

5\. \*\*`clustering/dbscan\_routes.py`\*\* - Route discovery algorithm

6\. \*\*`prediction/train\_lightgbm.py`\*\* - ETA model training



\*\*Open any file:\*\*



\*\*Windows:\*\*

```cmd

notepad preprocessing\\utils.py

```



\*\*Mac/Linux:\*\*

```bash

nano preprocessing/utils.py

\# or: code preprocessing/utils.py  (if you have VS Code)

```



---



\##  STEP 9: Common Issues \& Solutions



\### Issue 1: "conda: command not found"



\*\*Windows Solution:\*\*

```cmd

\# Add Anaconda to PATH manually

set PATH=%USERPROFILE%\\anaconda3;%USERPROFILE%\\anaconda3\\Scripts;%PATH%

```



\*\*Mac/Linux Solution:\*\*

```bash

\# Add to shell profile

echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc

source ~/.bashrc

```



\### Issue 2: "ModuleNotFoundError: No module named 'XXX'"



\*\*Solution:\*\*

```cmd

\# Reactivate environment

conda activate dnervenv



\# Reinstall package

pip install XXX

```



\### Issue 3: "FileNotFoundError: config/config.yaml not found"



\*\*Solution:\*\*

```cmd

\# Windows: Make sure you're in project directory

cd %USERPROFILE%\\Projects\\d-nerve-ml-models



\# Mac/Linux:

\# cd ~/Projects/d-nerve-ml-models



\# Check if file exists

\# Windows: dir config\\config.yaml

\# Mac/Linux: ls config/config.yaml

```



\### Issue 4: T-Drive data not found



\*\*Solution:\*\*

```cmd

\# Check path in config

\# Windows: notepad config\\config.yaml

\# Mac/Linux: nano config/config.yaml



\# Update this line to YOUR path:

\# Windows: tdrive\_path: "C:/Users/YOUR\_USERNAME/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

\# Mac: tdrive\_path: "/Users/yourname/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

\# Linux: tdrive\_path: "/home/yourname/d-nerve-data/t-drive/release/taxi\_log\_2008\_by\_id"

```



\### Issue 5: Scripts take too long



\*\*Solution:\*\*

\- DBSCAN on 2,181 trips takes ~30 minutes (normal)

\- To test faster, edit `config/config.yaml`:

```yaml

&nbsp; num\_taxis\_sample: 50  # Use 50 instead of 100

```

\- Then re-run from data loading



\### Issue 6: Permission Denied (Mac/Linux)



\*\*Solution:\*\*

```bash

\# Make scripts executable

chmod +x data\_loading/\*.py

chmod +x preprocessing/\*.py

chmod +x clustering/\*.py

chmod +x prediction/\*.py

```



---



\##  STEP 10: Next Steps for Each Team



\### For Backend Team (Group 1):



\*\*Your tasks:\*\*

1\. Load the trained models:

```python

&nbsp;  import pickle

&nbsp;  with open('outputs/eta\_model/lightgbm\_eta\_model.pkl', 'rb') as f:

&nbsp;      model = pickle.load(f)

```



2\. Create FastAPI endpoint `/predict\_eta`

3\. Implement feature extraction from GPS data

4\. Return predicted duration to mobile app



\*\*Files you need:\*\*

\- `outputs/eta\_model/lightgbm\_eta\_model.pkl`

\- `outputs/route\_discovery/route\_discovery\_results.pkl`

\- `prediction/feature\_engineering.py` (for reference)



\### For Mobile Team (Group 3):



\*\*Your tasks:\*\*

1\. Display routes on Google Maps

2\. Call backend `/predict\_eta` API

3\. Show ETA to users

4\. UI for route selection



\*\*Data you'll receive from backend:\*\*

\- Route coordinates (list of lat/lon points)

\- Predicted ETA (minutes)

\- Route metadata (popularity, etc.)



\### For ML Team (Group 2):



\*\*You're done! But available for:\*\*

\- Questions about models

\- Retraining with Cairo data

\- Performance tuning if needed

\- Backend integration support



---



\##  STEP 11: Learning Resources



\### Understanding the Algorithms:



\*\*DBSCAN (Route Discovery):\*\*

\- Paper: https://en.wikipedia.org/wiki/DBSCAN

\- Tutorial: https://scikit-learn.org/stable/modules/clustering.html#dbscan



\*\*LightGBM (ETA Prediction):\*\*

\- Documentation: https://lightgbm.readthedocs.io/

\- Tutorial: https://www.kaggle.com/learn/intro-to-machine-learning



\*\*Haversine Distance:\*\*

\- Explanation: https://en.wikipedia.org/wiki/Haversine\_formula



\### Python \& Data Science:



\- \*\*Pandas:\*\* https://pandas.pydata.org/docs/

\- \*\*NumPy:\*\* https://numpy.org/doc/

\- \*\*Scikit-learn:\*\* https://scikit-learn.org/stable/



---



\##  STEP 12: Getting Help



\### If You're Stuck:



1\. \*\*Check this guide\*\* - Most issues are covered above

2\. \*\*Review error message\*\* - Often tells you exactly what's wrong

3\. \*\*Check GitHub Issues\*\* - Someone may have had same problem

4\. \*\*Ask team lead\*\* - ML team member who set this up

5\. \*\*Create issue on GitHub\*\* - Document problem for team



\### GitHub Repository:

https://github.com/d-nerve-cairo/d-nerve-ml-models



\### Team Communication:

\[Add your team's Slack/Discord/WhatsApp here]



---



\##  Final Checklist



After completing this guide, you should have:



\- \[ ] Anaconda installed with `dnervenv` environment

\- \[ ] Git repository cloned to `~/Projects/d-nerve-ml-models` (or your chosen location)

\- \[ ] T-Drive dataset downloaded and extracted

\- \[ ] Config file updated with YOUR data path

\- \[ ] All preprocessing scripts executed successfully

\- \[ ] DBSCAN route discovery complete (20 routes found)

\- \[ ] LightGBM model trained (MAE = 9.04 min)

\- \[ ] Visualizations generated in `outputs/` folder

\- \[ ] Understanding of what each script does

\- \[ ] Ability to re-run entire pipeline



\*\*Time taken:\*\* ~60 minutes



\*\*Status:\*\*  You're now fully onboarded and ready to contribute!



---



\##  Quick Reference Commands



\### Universal Setup Commands:



\*\*Activate environment:\*\*

```bash

conda activate dnervenv



\# Navigate to project (choose your OS):

\# Windows: cd %USERPROFILE%\\Projects\\d-nerve-ml-models

\# Mac/Linux: cd ~/Projects/d-nerve-ml-models

```



\*\*Re-run entire pipeline (Windows):\*\*

```cmd

python data\_loading\\load\_tdrive.py

python preprocessing\\01\_remove\_outliers.py

python preprocessing\\02\_segment\_trips.py

python clustering\\dbscan\_routes.py

python prediction\\feature\_engineering.py

python prediction\\train\_lightgbm.py

python evaluation\\calculate\_f1.py

```



\*\*Re-run entire pipeline (Mac/Linux):\*\*

```bash

python data\_loading/load\_tdrive.py

python preprocessing/01\_remove\_outliers.py

python preprocessing/02\_segment\_trips.py

python clustering/dbscan\_routes.py

python prediction/feature\_engineering.py

python prediction/train\_lightgbm.py

python evaluation/calculate\_f1.py

```



\*\*Check results:\*\*

```bash

\# Windows: explorer outputs

\# Mac: open outputs

\# Linux: nautilus outputs

```



\*\*Update from GitHub:\*\*

```bash

git pull origin main

```



---



\*\*Last Updated:\*\* December 1, 2025  

\*\*Version:\*\* 1.0  

\*\*Maintained by:\*\* Group 2 - ML Team



\*\*Questions?\*\* Create an issue on GitHub or contact team lead.



---



\*\*🎉 Congratulations! You're ready to work on D-Nerve ML!\*\*

