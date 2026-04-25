# Titanic Survival Prediction Project

## Project Overview

This project analyzes the Titanic dataset and builds a feature-rich modeling dataset for survival prediction.
The assignment workflow is implemented in three major parts:

1. Data Cleaning
2. Feature Engineering
3. Feature Selection

The project includes notebook-based exploration and modular Python scripts for reproducible execution.

## Dataset

Files used:

- `data/train.csv`: training data with target column `Survived`
- `data/titanic_test.csv`: test data provided in this repository
- `data/test.csv`: optional alias for compatibility with assignment naming

Target variable:

- `Survived` (`1` = survived, `0` = did not survive)

## Project Structure

```text
titanic_assignment/
├── data/
│   ├── train.csv
│   ├── titanic_test.csv
│   ├── test.csv
│   ├── train_cleaned.csv
│   ├── train_engineered.csv
│   └── selected_features.csv
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
├── requirements.txt
└── README.md
```

## Setup

From project root:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## How To Run

### Option 1: Notebook Workflow

```bash
jupyter notebook
```

Open `notebooks/Titanic_Feature_Engineering.ipynb` and run cells in order.

### Option 2: Script Workflow

```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

## Assignment Coverage

### Part 1: Data Cleaning

Implemented:

- Missing values identified and handled across columns
- `Embarked` imputed with mode
- `Age` imputed with grouped median by `Pclass` and `Sex`
- `CabinMissing` indicator created
- `Deck` extracted from `Cabin` before dropping `Cabin`
- Outlier capping for `Fare` at 99th percentile
- `Sex` consistency normalization (lowercase strip)
- Duplicate row removal
- Output file: `data/train_cleaned.csv`

### Part 2: Feature Engineering

Implemented:

- Derived features: `FamilySize`, `IsAlone`, `Title`, `Deck`, `AgeGroup`, `FarePerPerson`
- One-hot encoding for nominal features: `Sex`, `Embarked`, `Title`, `Deck`, `AgeGroup`
- Ordinal handling: `Pclass` retained as integer order
- Interaction features: `Pclass_Fare`, `Age_Title`
- Transformations: `LogFare`, `LogAge`
- Scaling for distance-based models using `StandardScaler`
- Visualizations in notebook to justify transformations
- Output file: `data/train_engineered.csv`

### Part 3: Feature Selection

Implemented:

- Correlation analysis and redundant-feature pruning
- Random Forest feature importance ranking
- Recursive Feature Elimination (RFE)
- Final selected feature list output
- Output file: `data/selected_features.csv`

## Key Findings

- Family context (`FamilySize`, `IsAlone`) provides strong survival signals.
- Socioeconomic proxies (`Pclass`, `Fare`, `Title`) are informative predictors.
- Log transforms reduce skew in continuous features and stabilize model behavior.
- Tree-based importance and RFE produce overlapping high-value feature sets.

## Notes

- This repository includes both notebook and script implementations as requested.
- `test.csv` is included as a compatibility alias for the assignment structure.
>>>>>>> 3df296d (Aligned the project structure in order to complete the assignment)
