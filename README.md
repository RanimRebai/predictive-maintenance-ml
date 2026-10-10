# Predicta — Predictive Maintenance Using Machine Learning

Predicta is a machine learning project focused on **predictive maintenance and Remaining Useful Life (RUL) estimation** using simulated turbofan engine degradation data. The goal is to investigate sensor behavior, engineer meaningful features, and develop models that estimate how many operating cycles an engine has remaining before the end of its observed useful life.

The project follows an end-to-end machine learning workflow, from exploratory data analysis and target construction to model evaluation and potential deployment.

## Problem Statement

Unexpected equipment failures can lead to costly downtime and maintenance. Predictive maintenance uses operational data to estimate equipment condition and support maintenance decisions before failures occur.

This project investigates the following regression problem:

**Given an engine's sensor measurements and operating settings up to a particular cycle, how accurately can its Remaining Useful Life be estimated?**

The primary prediction target is Remaining Useful Life, measured in operating cycles.

## Dataset

The project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation dataset**, beginning with the FD001 subset.

* **FD001 training set:** 100 simulated engine trajectories
* **Operating conditions:** One operating condition
* **Fault modes:** One simulated fault mode
* **Features:** Three operational settings and 21 sensor measurements
* **Data structure:** Sequential observations grouped by engine

Each row represents an engine observation at a particular operating cycle.

Dataset source: [NASA Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/)

The dataset is not included in this repository. Download the dataset from the source and place the extracted `train_FD001.txt` file in `data/raw/`.

## Technical Stack

* **Language:** Python
* **Data manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine learning:** scikit-learn
* **Development:** Jupyter Notebook, Git, GitHub

Additional tools may be introduced as the modeling and deployment stages develop.

## Project Workflow

### 1. Exploratory Data Analysis

* Inspect dataset dimensions, feature names, and data types.
* Examine missing values, duplicate records, and sensor behavior.
* Analyze engine trajectory lengths and sensor measurements over time.
* Investigate the distribution of the prediction target.

### 2. Target Engineering

Construct RUL labels for the training observations using each engine's final observed cycle and the current cycle. Examine the resulting target distribution and document the assumptions behind this definition.

### 3. Feature Engineering

Investigate sensor trends, operating settings, and temporal information to identify features that may help estimate engine degradation.

### 4. Model Development

Establish a baseline regression model before evaluating more advanced approaches. Compare candidate models using appropriate regression metrics.

### 5. Evaluation

Use an evaluation strategy that accounts for observations belonging to the same engine. Avoid information leakage between training and evaluation data, and assess how well predictions generalize to unseen engines.

### 6. Future Extensions

Potential extensions include model interpretation, experiment tracking, and serving predictions through an API. These will be considered after the core modeling workflow is established.

## Repository Structure

```text
predictive-maintenance-ml/
├── data/
│   └── raw/
│       └── README.md
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

* `data/raw/` — locally stored raw dataset files; excluded from version control.
* `notebooks/` — notebooks for exploratory analysis and experiments.
* `src/` — source code for reusable data processing and modeling components.
* `tests/` — automated tests as the project develops.
* `requirements.txt` — Python dependencies.

## Getting Started

### Prerequisites

* Python 3.10 or later
* Git
* The NASA C-MAPSS dataset

### Installation

Clone the repository:

```bash
git clone https://github.com/RanimRebai/predictive-maintenance-ml.git
cd predictive-maintenance-ml
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Dataset Setup

Download the NASA C-MAPSS dataset and extract `train_FD001.txt` into:

```text
data/raw/train_FD001.txt
```

The raw dataset is intentionally excluded from Git version control.

### Run the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open `notebooks/01_data_exploration.ipynb` and execute the cells in order.

## Evaluation Metrics

Model performance will be assessed using regression metrics appropriate for RUL estimation, such as:

* **Mean Absolute Error (MAE):** average absolute prediction error in cycles.
* **Root Mean Squared Error (RMSE):** prediction error with greater sensitivity to large errors.

Evaluation results will be reported after the models have been trained and tested. No performance figures are claimed at the current stage.

## Current Status

**Phase 1 — Data Exploration**

The initial repository structure and exploratory notebook are in place. Current work focuses on understanding the FD001 data, examining engine trajectories and sensor behavior, and establishing a reliable basis for target construction and subsequent modeling.

The project is under active development. Model training, quantitative evaluation, and deployment are planned stages rather than completed features.

## Reproducibility and Data Handling

* Dependencies are listed in `requirements.txt`.
* Raw dataset files and virtual environments are excluded from version control.
* Modeling decisions and evaluation results will be documented as the project progresses.
* Dataset-derived conclusions will be based on observed results rather than assumed performance.

## License and Attribution

The dataset is provided by NASA. Refer to the [NASA Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository/) for dataset documentation and applicable terms.

A repository license has not yet been specified. A license should be added before the project is redistributed under explicit open-source terms.
