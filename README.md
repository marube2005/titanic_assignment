# Titanic Survival Prediction Project

## Project Overview

This project analyzes the **Titanic passenger dataset** to understand the factors that influenced survival during the disaster and to build machine learning models that predict passenger survival.

The workflow follows a standard data science pipeline:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Feature Selection
5. Model Training and Evaluation

The goal is to transform raw passenger data into meaningful features and identify which characteristics most strongly influence survival.

---

## Dataset

The dataset contains information about passengers aboard the Titanic, including:

* Passenger class (`Pclass`)
* Name
* Sex
* Age
* Number of siblings/spouses (`SibSp`)
* Number of parents/children (`Parch`)
* Ticket number
* Fare
* Cabin
* Port of embarkation (`Embarked`)
* Survival status (`Survived`)

Target variable:

* **Survived**

  * `1` → Passenger survived
  * `0` → Passenger did not survive

---

## Project Structure

```
Titanic-Project/
│
├── data/
│   └── train.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_selection.ipynb
│   └── 04_model_training.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/titanic-assignment.git
cd titanic-assignment
```

Install required packages:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

---

## How to Run the Project

### Option 1: Run the Notebooks (Recommended)

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run notebooks in order:

1. `01_data_cleaning.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_feature_selection.ipynb`
4. `04_model_training.ipynb`

Each notebook corresponds to one stage of the data science pipeline.

---

### Option 2: Run Python Scripts

You can also run the scripts individually:

```bash
python scripts/data_cleaning.py
python scripts/feature_engineering.py
python scripts/feature_selection.py
```

These scripts perform preprocessing and prepare the dataset for modeling.

---

## Feature Engineering Highlights

Several new features were created to improve predictive performance:

* **FamilySize** – total family members onboard
* **IsAlone** – indicates passengers traveling alone
* **Fare_per_person** – normalized fare within families
* **Title extraction** – titles extracted from passenger names
* **Deck extraction** – derived from cabin information
* **AgeGroup** – categorized age ranges
* **Interaction features** such as `Age_Title` and `Pclass_Fare`

Categorical variables were converted to numeric form using **one-hot encoding**.

---

## Feature Selection Methods

Three methods were used to identify the most informative predictors:

1. **Correlation analysis** to detect redundant variables
2. **Random Forest feature importance** to rank predictive features
3. **Recursive Feature Elimination (RFE)** to select an optimal subset of features

This process reduces noise and improves model generalization.

---

## Key Observations

Several factors strongly influenced survival:

### Gender

Female passengers had a significantly higher survival rate compared to males.

### Passenger Class

Passengers in **1st class** had a higher chance of survival than those in lower classes.

### Age

Children had higher survival rates, reflecting evacuation priorities.

### Family Size

Passengers traveling with **small families** tended to survive more often than those traveling alone or in very large groups.

### Fare

Higher fares (often associated with higher class cabins) were correlated with better survival outcomes.

---

## Tools and Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Seaborn
* Matplotlib
* Jupyter Notebook

---

## Future Improvements

Possible extensions of this project include:

* Hyperparameter tuning
* Ensemble models (XGBoost, Gradient Boosting)
* Cross-validation pipelines
* Advanced feature engineering (ticket groups, family survival signals)

---

## Author

Data Science / Machine Learning project focused on applying practical data preprocessing, feature engineering, and model selection techniques to a real-world dataset.
