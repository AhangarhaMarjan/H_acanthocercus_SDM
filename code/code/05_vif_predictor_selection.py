"""
Script: 05_vif_predictor_selection.py

Purpose:
Apply iterative variance inflation factor (VIF) screening to candidate
environmental predictors and save the final retained variable set.

Notes:
- Edit INPUT_FILE, OUTPUT_FOLDER, and CANDIDATE_FEATURES before running.
- This script removes the predictor with the highest VIF until all remaining
  variables have VIF <= 10, unless only one predictor remains.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


INPUT_FILE = Path("path/to/predictor_values.xlsx")
OUTPUT_FOLDER = Path("path/to/vif_outputs")
VIF_THRESHOLD = 10.0

# Edit this list to match the predictor set being screened.
CANDIDATE_FEATURES = [
    "bio3", "bio4", "bio8", "bio12", "clay", "slope",
]


def calculate_vif(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate VIF for all columns in the supplied data frame."""
    x_const = add_constant(data, has_constant="add")
    values = []

    for index, feature in enumerate(data.columns, start=1):
        values.append({
            "Feature": feature,
            "VIF": variance_inflation_factor(x_const.values, index),
        })

    return pd.DataFrame(values).sort_values("VIF", ascending=False)


def iterative_vif_selection(data: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove high-VIF variables iteratively and return retained/removed summaries."""
    remaining = data.copy()
    removed_records = []

    while remaining.shape[1] > 1:
        vif_table = calculate_vif(remaining)
        max_vif = float(vif_table.iloc[0]["VIF"])
        feature_to_remove = str(vif_table.iloc[0]["Feature"])

        if max_vif <= threshold:
            break

        removed_records.append({
            "Removed_feature": feature_to_remove,
            "VIF_at_removal": max_vif,
        })
        remaining = remaining.drop(columns=[feature_to_remove])

    final_vif = calculate_vif(remaining)
    removed_df = pd.DataFrame(removed_records)
    return final_vif, removed_df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_FILE)

    missing = [feature for feature in CANDIDATE_FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"These candidate features were not found in the input file: {missing}")

    x = df[CANDIDATE_FEATURES].dropna()
    final_vif, removed_df = iterative_vif_selection(x, VIF_THRESHOLD)

    final_vif.to_excel(OUTPUT_FOLDER / "final_vif_table.xlsx", index=False)
    final_vif.to_csv(OUTPUT_FOLDER / "final_vif_table.csv", index=False)
    removed_df.to_excel(OUTPUT_FOLDER / "vif_removed_variables.xlsx", index=False)
    removed_df.to_csv(OUTPUT_FOLDER / "vif_removed_variables.csv", index=False)

    corr_matrix = x[final_vif["Feature"].tolist()].corr(method="spearman")
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Spearman correlation heatmap of VIF-retained predictors")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "vif_retained_predictor_heatmap.png", dpi=300)
    plt.close()

    print("Final VIF table:")
    print(final_vif.to_string(index=False))
    print(f"Outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
