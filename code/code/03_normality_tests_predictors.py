"""
Script: 03_normality_tests_predictors.py

Purpose:
Run Kolmogorov-Smirnov and Shapiro-Wilk normality tests for numeric
environmental predictors and save diagnostic plots.

Notes:
- Edit INPUT_FILE and OUTPUT_FOLDER before running.
- Results were used to justify the use of non-parametric Spearman correlation
  during predictor screening.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


INPUT_FILE = Path("path/to/predictor_values.xlsx")
OUTPUT_FOLDER = Path("path/to/normality_test_outputs")
MIN_SAMPLE_SIZE = 30


def test_variable_normality(series: pd.Series) -> dict:
    """Return K-S and Shapiro-Wilk normality results for one variable."""
    data = series.dropna()
    mean_value = data.mean()
    std_value = data.std()

    if std_value == 0 or np.isnan(std_value):
        return {
            "K-S Statistic": np.nan,
            "K-S P-Value": np.nan,
            "K-S Result": "Constant or invalid",
            "Shapiro Statistic": np.nan,
            "Shapiro P-Value": np.nan,
            "Shapiro Result": "Constant or invalid",
        }

    ks_statistic, ks_p_value = stats.kstest(data, "norm", args=(mean_value, std_value))
    shapiro_statistic, shapiro_p_value = stats.shapiro(data)

    return {
        "K-S Statistic": ks_statistic,
        "K-S P-Value": ks_p_value,
        "K-S Result": "Normal" if ks_p_value > 0.05 else "Not normal",
        "Shapiro Statistic": shapiro_statistic,
        "Shapiro P-Value": shapiro_p_value,
        "Shapiro Result": "Normal" if shapiro_p_value > 0.05 else "Not normal",
    }


def save_diagnostic_plot(data: pd.Series, variable_name: str, output_folder: Path) -> None:
    """Save a histogram and Q-Q plot for one predictor."""
    clean_data = data.dropna()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(clean_data, kde=True, bins=30)
    plt.title(f"Histogram of {variable_name}")
    plt.xlabel("Value")

    plt.subplot(1, 2, 2)
    stats.probplot(clean_data, dist="norm", plot=plt)
    plt.title(f"Q-Q plot of {variable_name}")

    plt.tight_layout()
    plt.savefig(output_folder / f"{variable_name}_normality_plot.png", dpi=300)
    plt.close()


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(INPUT_FILE)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for column in numeric_columns:
        data = df[column].dropna()
        if len(data) < MIN_SAMPLE_SIZE:
            continue

        row = {"Variable": column, "Sample size": len(data)}
        row.update(test_variable_normality(df[column]))
        results.append(row)
        save_diagnostic_plot(df[column], column, OUTPUT_FOLDER)
        print(f"Processed normality tests for: {column}")

    results_df = pd.DataFrame(results)
    results_df.to_excel(OUTPUT_FOLDER / "normality_test_results.xlsx", index=False)
    results_df.to_csv(OUTPUT_FOLDER / "normality_test_results.csv", index=False)

    print(f"Normality-test results saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
