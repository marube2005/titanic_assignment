"""Part 3: Feature selection for Titanic assignment.

Usage:
	python scripts/feature_selection.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def select_features(
	input_path: Path,
	output_path: Path,
	corr_threshold: float = 0.9,
	top_k: int = 15,
	rfe_k: int = 10,
) -> list[str]:
	"""Run feature selection and persist chosen features."""
	df = pd.read_csv(input_path)

	X = df.drop(columns=["Survived"])
	y = df["Survived"]

	# Correlation pruning on numeric columns only.
	numeric_X = X.select_dtypes(include="number")
	corr = numeric_X.corr().abs()
	upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
	to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
	X = X.drop(columns=[c for c in to_drop if c in X.columns])

	# Random Forest importance.
	rf = RandomForestClassifier(n_estimators=300, random_state=42)
	rf.fit(X, y)
	importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
	top_features = importances.index[:top_k].tolist()

	# RFE for compact subset.
	rfe = RFE(estimator=RandomForestClassifier(n_estimators=200, random_state=42), n_features_to_select=rfe_k)
	rfe.fit(X, y)
	rfe_features = X.columns[rfe.support_].tolist()

	# Final: intersection for robustness; fallback to top importance.
	final_features = [f for f in rfe_features if f in top_features]
	if len(final_features) < 8:
		final_features = top_features[:10]

	out_df = pd.DataFrame(
		{
			"feature": final_features,
			"rf_importance": [importances[f] for f in final_features],
		}
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(output_path, index=False)
	return final_features


def main() -> None:
	input_path = DATA_DIR / "train_engineered.csv"
	output_path = DATA_DIR / "selected_features.csv"

	selected = select_features(input_path=input_path, output_path=output_path)
	print(f"Saved selected features: {output_path}")
	print("Selected features:")
	print(selected)


if __name__ == "__main__":
	main()
