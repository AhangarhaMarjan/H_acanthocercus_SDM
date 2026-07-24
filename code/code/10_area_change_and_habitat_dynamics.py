"""
Script: 10_area_change_and_habitat_dynamics.py

Purpose:
Calculate suitable-habitat area and pairwise habitat-dynamics classes from
binary MaxEnt suitability maps.

Notes:
- Edit PAIRWISE_COMPARISONS before running.
- Binary rasters should use 1 = suitable and 0 = unsuitable. NoData values are
  ignored in area calculations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


OUTPUT_FOLDER = Path("path/to/habitat_dynamics_outputs")
NODATA_VALUES = {-9999, 255}

# Edit these pairs before running.
PAIRWISE_COMPARISONS = [
    {
        "name": "present_to_ssp245",
        "baseline": Path("path/to/binary_present.tif"),
        "target": Path("path/to/binary_ssp245.tif"),
    },
]

# Optional three-period trajectory examples. Leave empty if not needed.
TRAJECTORY_COMPARISONS = [
    # {
    #     "name": "past_present_ssp245",
    #     "past": Path("path/to/binary_past.tif"),
    #     "present": Path("path/to/binary_present.tif"),
    #     "future": Path("path/to/binary_ssp245.tif"),
    # },
]

FOUR_CLASS_NAMES = {
    0: "Stable unsuitable",
    1: "Loss",
    2: "Gain",
    3: "Stable suitable",
}

TRAJECTORY_NAMES = {
    0: "Stable unsuitable",
    1: "Past-only suitable",
    2: "Transient present-only suitable",
    3: "Suitable in past and present, lost in future",
    4: "Future-only suitable",
    5: "Loss followed by future return",
    6: "Newly suitable from present onward",
    7: "Stable suitable across all periods",
}


def read_binary_raster(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a binary suitability raster and return data, valid mask, and metadata."""
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata

    invalid = np.zeros(data.shape, dtype=bool)
    if nodata is not None:
        invalid |= data == nodata
    for nodata_value in NODATA_VALUES:
        invalid |= data == nodata_value

    valid = ~invalid
    return data, valid, meta


def area_grid_km2(meta: dict) -> np.ndarray:
    """Approximate cell area in km² for each row of a geographic raster."""
    transform = meta["transform"]
    height = meta["height"]
    width = meta["width"]

    pixel_width_deg = abs(transform.a)
    pixel_height_deg = abs(transform.e)
    rows = np.arange(height)
    latitudes = transform.f + (rows + 0.5) * transform.e

    row_area = (pixel_width_deg * 111.32) * (pixel_height_deg * 111.32 * np.cos(np.radians(latitudes)))
    return np.repeat(row_area[:, np.newaxis], width, axis=1)


def summarize_area(class_map: np.ndarray, valid_mask: np.ndarray, area_map: np.ndarray, class_names: dict) -> pd.DataFrame:
    """Summarize area by raster class."""
    records = []
    total_area = float(area_map[valid_mask].sum())

    for code, label in class_names.items():
        mask = valid_mask & (class_map == code)
        area = float(area_map[mask].sum())
        records.append({
            "Code": code,
            "Class": label,
            "Area_km2": area,
            "Area_percent": (area / total_area * 100) if total_area > 0 else np.nan,
        })

    return pd.DataFrame(records)


def pairwise_habitat_dynamics(name: str, baseline_path: Path, target_path: Path, output_folder: Path) -> pd.DataFrame:
    """Calculate stable unsuitable, loss, gain, and stable suitable classes."""
    baseline, valid_baseline, meta = read_binary_raster(baseline_path)
    target, valid_target, _ = read_binary_raster(target_path)

    valid = valid_baseline & valid_target
    class_map = np.full(baseline.shape, -9999, dtype="int16")
    class_map[valid & (baseline == 0) & (target == 0)] = 0
    class_map[valid & (baseline == 1) & (target == 0)] = 1
    class_map[valid & (baseline == 0) & (target == 1)] = 2
    class_map[valid & (baseline == 1) & (target == 1)] = 3

    area_map = area_grid_km2(meta)
    summary = summarize_area(class_map, valid, area_map, FOUR_CLASS_NAMES)
    summary.insert(0, "Comparison", name)

    meta.update(driver="GTiff", dtype="int16", nodata=-9999, count=1)
    with rasterio.open(output_folder / f"{name}_four_class_dynamics.tif", "w", **meta) as dst:
        dst.write(class_map, 1)

    return summary


def trajectory_analysis(name: str, past_path: Path, present_path: Path, future_path: Path, output_folder: Path) -> pd.DataFrame:
    """Calculate 0-7 trajectory classes from past, present, and future binary maps."""
    past, valid_past, meta = read_binary_raster(past_path)
    present, valid_present, _ = read_binary_raster(present_path)
    future, valid_future, _ = read_binary_raster(future_path)

    valid = valid_past & valid_present & valid_future
    trajectory = np.full(past.shape, -9999, dtype="int16")
    trajectory[valid] = past[valid] + 2 * present[valid] + 4 * future[valid]

    area_map = area_grid_km2(meta)
    summary = summarize_area(trajectory, valid, area_map, TRAJECTORY_NAMES)
    summary.insert(0, "Trajectory", name)

    meta.update(driver="GTiff", dtype="int16", nodata=-9999, count=1)
    with rasterio.open(output_folder / f"{name}_eight_class_trajectory.tif", "w", **meta) as dst:
        dst.write(trajectory, 1)

    return summary


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    pairwise_results = []
    for comparison in PAIRWISE_COMPARISONS:
        summary = pairwise_habitat_dynamics(
            comparison["name"],
            comparison["baseline"],
            comparison["target"],
            OUTPUT_FOLDER,
        )
        pairwise_results.append(summary)

    if pairwise_results:
        pairwise_df = pd.concat(pairwise_results, ignore_index=True)
        pairwise_df.to_csv(OUTPUT_FOLDER / "four_class_habitat_dynamics_summary.csv", index=False)
        pairwise_df.to_excel(OUTPUT_FOLDER / "four_class_habitat_dynamics_summary.xlsx", index=False)

    trajectory_results = []
    for comparison in TRAJECTORY_COMPARISONS:
        summary = trajectory_analysis(
            comparison["name"],
            comparison["past"],
            comparison["present"],
            comparison["future"],
            OUTPUT_FOLDER,
        )
        trajectory_results.append(summary)

    if trajectory_results:
        trajectory_df = pd.concat(trajectory_results, ignore_index=True)
        trajectory_df.to_csv(OUTPUT_FOLDER / "eight_class_trajectory_summary.csv", index=False)
        trajectory_df.to_excel(OUTPUT_FOLDER / "eight_class_trajectory_summary.xlsx", index=False)

    print(f"Habitat-dynamics outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
