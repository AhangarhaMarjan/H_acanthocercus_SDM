"""
Script: 12_run_maxent_tuning_and_final_models.py

Purpose:
Run MaxEnt tuning for the dual-baseline species distribution modelling workflow.
The script tests 48 candidate settings from six feature-class combinations and
eight regularization multipliers, using 15 random subsampling replicates.

Notes:
- Edit MAXENT_JAR, SAMPLES_FILE, ENVIRONMENTAL_LAYERS, and OUTPUT_BASE before running.
- The script assumes MaxEnt 3.4.4 is available as maxent.jar.
- Final model interpretation should also consider omission rates, gain, model
  complexity, and ecological interpretability, not test AUC alone.
"""

from pathlib import Path
import subprocess
import time
from datetime import timedelta

import pandas as pd


MAXENT_JAR = Path("path/to/maxent.jar")
SAMPLES_FILE = Path("path/to/occurrence_samples.csv")
ENVIRONMENTAL_LAYERS = Path("path/to/ascii_predictor_layers")
OUTPUT_BASE = Path("path/to/maxent_tuning_outputs")

JAVA_MEMORY = "8g"
THREADS = 4
REPLICATES = 15
RANDOM_TEST_PERCENT = 25
OUTPUT_FORMAT = "cloglog"
MAXIMUM_ITERATIONS = 5000

FEATURE_CONFIGS = {
    "L": {"linear": True, "quadratic": False, "hinge": False, "product": False, "threshold": False},
    "LQ": {"linear": True, "quadratic": True, "hinge": False, "product": False, "threshold": False},
    "LQH": {"linear": True, "quadratic": True, "hinge": True, "product": False, "threshold": False},
    "LQHP": {"linear": True, "quadratic": True, "hinge": True, "product": True, "threshold": False},
    "LQHT": {"linear": True, "quadratic": True, "hinge": True, "product": False, "threshold": True},
    "LQHPT": {"linear": True, "quadratic": True, "hinge": True, "product": True, "threshold": True},
}

REGULARIZATION_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def bool_arg(name: str, value: bool) -> str:
    return f"{name}={'true' if value else 'false'}"


def build_maxent_command(feature_name: str, rm: float, output_dir: Path) -> list[str]:
    """Build one MaxEnt command-line call."""
    features = FEATURE_CONFIGS[feature_name]

    command = [
        "java",
        f"-mx{JAVA_MEMORY}",
        "-jar", str(MAXENT_JAR),
        f"samplesfile={SAMPLES_FILE}",
        f"environmentallayers={ENVIRONMENTAL_LAYERS}",
        f"outputdirectory={output_dir}",
        f"betamultiplier={rm}",
        f"replicates={REPLICATES}",
        "replicatetype=subsample",
        f"randomtestpoints={RANDOM_TEST_PERCENT}",
        f"outputformat={OUTPUT_FORMAT}",
        bool_arg("linear", features["linear"]),
        bool_arg("quadratic", features["quadratic"]),
        bool_arg("hinge", features["hinge"]),
        bool_arg("product", features["product"]),
        bool_arg("threshold", features["threshold"]),
        f"maximumiterations={MAXIMUM_ITERATIONS}",
        f"threads={THREADS}",
        "autorun",
        "plots=true",
        "jackknife=true",
        "responsecurves=true",
        "visible=false",
        "askoverwrite=false",
    ]
    return command


def read_average_results(model_dir: Path, feature_name: str, rm: float) -> dict | None:
    """Extract selected summary fields from maxentResults.csv."""
    csv_path = model_dir / "maxentResults.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    first_column = df.columns[0]
    average_rows = df[df[first_column].astype(str).str.contains("average", case=False, na=False)]
    if average_rows.empty:
        average_rows = df.tail(1)

    row = average_rows.iloc[0]
    return {
        "Setting_Name": model_dir.name,
        "Feature_Class": feature_name,
        "Regularization_Multiplier": rm,
        "Training_AUC": row.get("Training AUC", None),
        "Test_AUC": row.get("Test AUC", None),
        "AUC_SD": row.get("AUC Standard Deviation", None),
        "Test_Gain": row.get("Test gain", None),
        "Omission_10pct": row.get("10 percentile training presence test omission", None),
        "Threshold_10pct_cloglog": row.get("10 percentile training presence Cloglog threshold", None),
    }


def run_tuning() -> None:
    """Run all candidate MaxEnt settings and save a model-selection summary."""
    for required_path in [MAXENT_JAR, SAMPLES_FILE, ENVIRONMENTAL_LAYERS]:
        if not required_path.exists():
            raise FileNotFoundError(f"Required path not found: {required_path}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    combinations = [(feature_name, rm) for feature_name in FEATURE_CONFIGS for rm in REGULARIZATION_MULTIPLIERS]
    total_tasks = len(combinations)
    summary_records = []
    start_time = time.time()

    print(f"Running {total_tasks} MaxEnt candidate settings.")

    for index, (feature_name, rm) in enumerate(combinations, start=1):
        model_name = f"FC_{feature_name}_RM_{rm}"
        model_dir = OUTPUT_BASE / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        command = build_maxent_command(feature_name, rm, model_dir)
        run_log = model_dir / "maxent_run_log.txt"

        result = subprocess.run(command, capture_output=True, text=True)
        run_log.write_text(
            "COMMAND:\n" + " ".join(command) + "\n\nSTDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr,
            encoding="utf-8",
        )

        if result.returncode != 0:
            print(f"MaxEnt returned a non-zero status for {model_name}. Check maxent_run_log.txt.")

        average_results = read_average_results(model_dir, feature_name, rm)
        if average_results is not None:
            summary_records.append(average_results)

        elapsed = time.time() - start_time
        average_time = elapsed / index
        eta = average_time * (total_tasks - index)
        print(f"{index}/{total_tasks} finished: {model_name}; ETA {timedelta(seconds=int(eta))}")

    if summary_records:
        summary = pd.DataFrame(summary_records)
        summary = summary.sort_values(by=["Test_AUC"], ascending=False)
        summary.to_excel(OUTPUT_BASE / "model_selection_summary.xlsx", index=False)
        summary.to_csv(OUTPUT_BASE / "model_selection_summary.csv", index=False)
        print(f"Model-selection summary saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    run_tuning()
