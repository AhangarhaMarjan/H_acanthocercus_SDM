"""
Script: 04_spearman_correlation_filtering.py

Purpose:
Calculate Spearman rank correlations among candidate environmental predictors
and list highly correlated variable pairs using |r| > 0.8.

Notes:
- Edit INPUT_FILE and OUTPUT_FOLDER before running.
- The script reports correlated pairs. Final removal decisions should also
  consider ecological relevance and model behavior.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_FILE = Path("path/to/predictor_values.xlsx")
OUTPUT_FOLDER = Path("path/to/spearman_outputs")
CORRELATION_THRESHOLD = 0.80


def find_highly_correlated_pairs(correlation_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return variable pairs with absolute Spearman correlation above the threshold."""
    records = []
    columns = correlation_matrix.columns.tolist()

    for i, var1 in enumerate(columns):
        for var2 in columns[i + 1:]:
            rho = correlation_matrix.loc[var1, var2]
            if abs(rho) > threshold:
                records.append({
                    "Variable_1": var1,
                    "Variable_2": var2,
                    "Spearman_rho": rho,
                    "Abs_rho": abs(rho),
                })

    return pd.DataFrame(records).sort_values("Abs_rho", ascending=False)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_FILE)
    environmental_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(environmental_columns) < 2:
        raise ValueError("At least two numeric predictor columns are required.")

    df_env = df[environmental_columns]
    spearman_corr = df_env.corr(method="spearman")

    spearman_corr.to_excel(OUTPUT_FOLDER / "spearman_correlation_matrix.xlsx")
    spearman_corr.to_csv(OUTPUT_FOLDER / "spearman_correlation_matrix.csv")

    high_corr_pairs = find_highly_correlated_pairs(spearman_corr, CORRELATION_THRESHOLD)
    high_corr_pairs.to_excel(OUTPUT_FOLDER / "highly_correlated_pairs.xlsx", index=False)
    high_corr_pairs.to_csv(OUTPUT_FOLDER / "highly_correlated_pairs.csv", index=False)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        spearman_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Spearman rank correlation matrix (|r| > {CORRELATION_THRESHOLD})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "spearman_correlation_heatmap.png", dpi=300)
    plt.close()

    print(f"Number of numeric predictors: {len(environmental_columns)}")
    print(f"Number of highly correlated pairs: {len(high_corr_pairs)}")
    print(f"Outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
