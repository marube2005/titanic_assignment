"""Part 2: Feature engineering for Titanic assignment.

Usage:
	python scripts/feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def engineer_features(input_path: Path, output_path: Path) -> pd.DataFrame:
	"""Build engineered features and save engineered dataset."""
	df = pd.read_csv(input_path)

	# Derived features.
	df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
	df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
	df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)

	# Title extraction and rare-title grouping.
	df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\\.", expand=False)
	rare_titles = [
		"Lady",
		"Countess",
		"Capt",
		"Col",
		"Don",
		"Dr",
		"Major",
		"Rev",
		"Sir",
		"Jonkheer",
		"Dona",
	]
	df["Title"] = df["Title"].replace(rare_titles, "Rare")
	df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

	# Ensure Deck exists (it should come from cleaning step).
	if "Deck" not in df.columns:
		df["Deck"] = "U"
	df["Deck"] = df["Deck"].fillna("U")

	# Age groups.
	bins = [0, 12, 19, 59, np.inf]
	labels = ["Child", "Teen", "Adult", "Senior"]
	df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)

	# Interaction features.
	df["Pclass_Fare"] = df["Pclass"] * df["Fare"]
	df["Age_Title"] = df["Age"] * (df["Title"] == "Mr").astype(int)

	# Transform skewed columns.
	df["LogFare"] = np.log1p(df["Fare"])
	df["LogAge"] = np.log1p(df["Age"])

	# One-hot encode nominal features; retain Pclass as ordinal numeric.
	columns_to_encode = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]
	df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

	# Scale numeric columns for distance-based models.
	scale_cols = ["Age", "Fare", "LogFare", "LogAge", "FarePerPerson"]
	scaler = StandardScaler()
	df[scale_cols] = scaler.fit_transform(df[scale_cols])

	# Drop identifiers / non-informative text fields.
	drop_cols = ["Name", "Ticket", "PassengerId"]
	existing_drop_cols = [col for col in drop_cols if col in df.columns]
	df = df.drop(columns=existing_drop_cols)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	return df


def main() -> None:
	input_path = DATA_DIR / "train_cleaned.csv"
	output_path = DATA_DIR / "train_engineered.csv"

	engineered = engineer_features(input_path=input_path, output_path=output_path)
	print(f"Saved engineered data: {output_path}")
	print(f"Shape: {engineered.shape}")


if __name__ == "__main__":
	main()
