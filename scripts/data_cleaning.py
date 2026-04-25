"""Part 1: Data cleaning for Titanic assignment.

Usage:
	python scripts/data_cleaning.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
	"""Load raw Titanic data, clean it, and save cleaned output."""
	df = pd.read_csv(input_path)

	# Standardize Sex values for consistency.
	df["Sex"] = df["Sex"].astype(str).str.strip().str.lower()

	# Impute missing categorical and numeric values.
	df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
	df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
		lambda x: x.fillna(x.median())
	)

	# Extract deck information before dropping Cabin.
	df["Deck"] = df["Cabin"].str[0].fillna("U")
	df["CabinMissing"] = df["Cabin"].isnull().astype(int)
	df = df.drop(columns=["Cabin"])

	# Cap extreme fare outliers.
	fare_cap = df["Fare"].quantile(0.99)
	df["Fare"] = np.where(df["Fare"] > fare_cap, fare_cap, df["Fare"])

	# Remove duplicates if present.
	df = df.drop_duplicates().reset_index(drop=True)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_path, index=False)
	return df


def main() -> None:
	input_path = DATA_DIR / "train.csv"
	output_path = DATA_DIR / "train_cleaned.csv"

	cleaned = clean_data(input_path=input_path, output_path=output_path)
	print(f"Saved cleaned data: {output_path}")
	print(f"Shape: {cleaned.shape}")
	print("Missing values after cleaning:")
	print(cleaned.isnull().sum())


if __name__ == "__main__":
	main()
