"""
Script: 11_spatial_agreement_and_transferability_metrics.py

Purpose:
Calculate spatial agreement and temporal-transferability metrics from the four
cells of a binary confusion matrix.

Notes:
- Edit TP, FN, FP, TN, and OUTPUT_FOLDER before running.
- In this manuscript, the contemporary present-day model was treated as the
  reference and the historical model transferred to present conditions was the
  prediction being evaluated.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUTPUT_FOLDER = Path("path/to/spatial_agreement_outputs")

# Confusion-matrix values from the spatial overlay of the two binary maps.
TP = 3082
FN = 937
FP = 656
TN = 85163


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else np.nan


def calculate_metrics(tp: int, fn: int, fp: int, tn: int) -> pd.DataFrame:
    """Calculate common binary agreement metrics."""
    total = tp + fn + fp + tn

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    overall_accuracy = safe_divide(tp + tn, total)
    f1_score = safe_divide(2 * precision * sensitivity, precision + sensitivity)
    jaccard = safe_divide(tp, tp + fp + fn)
    dice = safe_divide(2 * tp, 2 * tp + fp + fn)
    tss = sensitivity + specificity - 1

    observed_agreement = overall_accuracy
    expected_agreement = safe_divide((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total ** 2)
    kappa = safe_divide(observed_agreement - expected_agreement, 1 - expected_agreement)

    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_divide(tp * tn - fp * fn, mcc_denominator)

    records = [
        ("Overall accuracy", "OA", overall_accuracy, "0 to 1"),
        ("Sensitivity / recall", "Sensitivity", sensitivity, "0 to 1"),
        ("Specificity", "Specificity", specificity, "0 to 1"),
        ("Precision", "Precision", precision, "0 to 1"),
        ("F1-score", "F1", f1_score, "0 to 1"),
        ("Jaccard index", "Jaccard", jaccard, "0 to 1"),
        ("Dice coefficient", "Dice", dice, "0 to 1"),
        ("True skill statistic", "TSS", tss, "-1 to +1"),
        ("Cohen's kappa", "Kappa", kappa, "-1 to +1"),
        ("Matthews correlation coefficient", "MCC", mcc, "-1 to +1"),
    ]

    df = pd.DataFrame(records, columns=["Metric", "Abbreviation", "Value", "Range"])
    df["Value"] = df["Value"].round(4)
    return df


def save_confusion_matrix_plot(tp: int, fn: int, fp: int, tn: int, output_path: Path) -> None:
    """Save a heatmap of the spatial confusion matrix."""
    matrix = np.array([[tp, fn], [fp, tn]])

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=True, square=True,
                annot_kws={"size": 13, "weight": "bold"})
    plt.title("Spatial confusion matrix", fontsize=14, pad=15, weight="bold")
    plt.xlabel("Prediction: historical model transferred to present", fontsize=11)
    plt.ylabel("Reference: contemporary present-day model", fontsize=11)
    plt.xticks([0.5, 1.5], ["Suitable (1)", "Unsuitable (0)"], fontsize=10)
    plt.yticks([0.5, 1.5], ["Suitable (1)", "Unsuitable (0)"], fontsize=10, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(TP, FN, FP, TN)
    metrics.to_excel(OUTPUT_FOLDER / "spatial_agreement_metrics.xlsx", index=False)
    metrics.to_csv(OUTPUT_FOLDER / "spatial_agreement_metrics.csv", index=False)
    save_confusion_matrix_plot(TP, FN, FP, TN, OUTPUT_FOLDER / "spatial_confusion_matrix.png")

    print(metrics.to_string(index=False))
    print(f"Spatial agreement outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
