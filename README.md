# Predicta — Predictive Maintenance ML

An educational project exploring Remaining Useful Life (RUL) prediction for simulated turbofan engines using NASA's C-MAPSS dataset.

## Goal
Estimate how many operating cycles remain before an engine reaches failure using sensor measurements. This is a regression problem.

## Status
**Phase 1: data exploration — in progress.** This starter repository contains a notebook scaffold. Add findings only after running it and inspecting the data.

## Dataset
Official source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

Start with FD001. NASA describes FD001 as 100 training trajectories, 100 test trajectories, one operating condition, and one fault mode. The data are simulated, not real aircraft operational records.

### Download
1. Download `CMAPSSData.zip` from the NASA page.
2. Extract it.
3. Copy `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into `data/raw/`.

Raw data files are excluded from Git by `.gitignore`.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Activate it:
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/01_data_exploration.ipynb` and run cells from top to bottom.

## Week 1 deliverables
- [ ] Load FD001 training data
- [ ] Verify shape, column names, types, missing values and duplicates
- [ ] Count engines and inspect cycle ranges
- [ ] Identify constant or near-constant sensors
- [ ] Plot selected sensor readings for one engine
- [ ] Write findings based on actual outputs
- [ ] Commit and push the notebook

## Important
When we later evaluate models, we will choose the validation strategy deliberately. Individual rows from the same engine are related over time, so a careless random row split can produce misleading results.

## Planned roadmap
1. Data exploration and quality checks
2. RUL target construction
3. Baseline regression model
4. Feature engineering and stronger models
5. Leakage-aware evaluation and error analysis
6. Explainability
7. API and optional dashboard

## Dataset citation
Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository. See the official NASA dataset page above.
