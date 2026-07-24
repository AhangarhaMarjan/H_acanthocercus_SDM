"""
Script: 09_calculate_mess_and_mod.py

Purpose:
Calculate Multivariate Environmental Similarity Surface (MESS) and Most
Dissimilar Variable (MoD) rasters for temporal or future projections.

Notes:
- Edit REFERENCE_FOLDER, TARGET_FOLDER, OUTPUT_FOLDER, and SCENARIO_NAME before running.
- Reference and target folders must contain aligned ASCII rasters with matching
  file names for the retained predictors.
"""

from pathlib import Path

import csv
import numpy as np
import rasterio


REFERENCE_FOLDER = Path("path/to/reference_predictor_layers")
TARGET_FOLDER = Path("path/to/target_predictor_layers")
OUTPUT_FOLDER = Path("path/to/mess_mod_outputs")
SCENARIO_NAME = "scenario_name"
OUTPUT_NODATA = -9999


def load_matching_stacks(reference_folder: Path, target_folder: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load matching reference and target predictor rasters."""
    reference_files = sorted(reference_folder.glob("*.asc"))
    reference_arrays = []
    target_arrays = []
    variable_names = []
    output_meta = None

    for reference_path in reference_files:
        target_path = target_folder / reference_path.name
        if not target_path.exists():
            print(f"Missing target file for {reference_path.name}; skipped.")
            continue

        with rasterio.open(reference_path) as ref_src, rasterio.open(target_path) as target_src:
            ref_data = ref_src.read(1).astype("float32")
            target_data = target_src.read(1).astype("float32")

            if ref_src.nodata is not None:
                ref_data[ref_data == ref_src.nodata] = np.nan
            if target_src.nodata is not None:
                target_data[target_data == target_src.nodata] = np.nan

            reference_arrays.append(ref_data)
            target_arrays.append(target_data)
            variable_names.append(reference_path.stem)

            if output_meta is None:
                output_meta = ref_src.meta.copy()
                output_meta.update(driver="GTiff", dtype="float32", nodata=OUTPUT_NODATA, count=1)

    if not reference_arrays:
        raise ValueError("No matching reference-target predictor files were found.")

    return np.stack(reference_arrays), np.stack(target_arrays), variable_names, output_meta


def variable_similarity(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate MESS similarity for one environmental predictor."""
    valid_reference = reference[~np.isnan(reference)]
    if valid_reference.size == 0:
        return np.full_like(target, np.nan, dtype="float32")

    ref_min = float(np.min(valid_reference))
    ref_max = float(np.max(valid_reference))
    ref_range = ref_max - ref_min

    similarity = np.full_like(target, np.nan, dtype="float32")
    valid_target = ~np.isnan(target)

    if ref_range == 0:
        similarity[valid_target & (target == ref_min)] = 100
        similarity[valid_target & (target != ref_min)] = -100
        return similarity

    low_mask = valid_target & (target < ref_min)
    high_mask = valid_target & (target > ref_max)
    mid_mask = valid_target & (target >= ref_min) & (target <= ref_max)

    similarity[low_mask] = ((target[low_mask] - ref_min) / ref_range) * 100
    similarity[high_mask] = ((ref_max - target[high_mask]) / ref_range) * 100

    reference_sorted = np.sort(valid_reference)
    positions = np.searchsorted(reference_sorted, target[mid_mask], side="left")
    percentiles = (positions / len(reference_sorted)) * 100
    similarity[mid_mask] = np.where(percentiles <= 50, 2 * percentiles, 2 * (100 - percentiles))

    return similarity


def calculate_mess_mod(reference_stack: np.ndarray, target_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MESS raster, MoD raster, and per-variable similarity stack."""
    similarity_stack = np.stack([
        variable_similarity(reference_stack[index], target_stack[index])
        for index in range(reference_stack.shape[0])
    ])

    valid_mask = ~np.all(np.isnan(similarity_stack), axis=0)
    mess_map = np.full(similarity_stack.shape[1:], OUTPUT_NODATA, dtype="float32")
    mod_map = np.full(similarity_stack.shape[1:], OUTPUT_NODATA, dtype="float32")

    mess_map[valid_mask] = np.nanmin(similarity_stack[:, valid_mask], axis=0)
    mod_map[valid_mask] = np.nanargmin(similarity_stack[:, valid_mask], axis=0) + 1

    return mess_map, mod_map, similarity_stack


def write_raster(path: Path, array: np.ndarray, meta: dict) -> None:
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype("float32"), 1)


def write_mod_legend(path: Path, variable_names: list[str]) -> None:
    """Save the numeric MoD code and predictor-name correspondence."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["MoD_code", "Predictor"])
        for index, name in enumerate(variable_names, start=1):
            writer.writerow([index, name])


def main() -> None:
    if not REFERENCE_FOLDER.exists():
        raise FileNotFoundError(f"Reference folder not found: {REFERENCE_FOLDER}")
    if not TARGET_FOLDER.exists():
        raise FileNotFoundError(f"Target folder not found: {TARGET_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    reference_stack, target_stack, variable_names, meta = load_matching_stacks(REFERENCE_FOLDER, TARGET_FOLDER)
    mess_map, mod_map, _ = calculate_mess_mod(reference_stack, target_stack)

    write_raster(OUTPUT_FOLDER / f"mess_{SCENARIO_NAME}.tif", mess_map, meta)
    write_raster(OUTPUT_FOLDER / f"mod_{SCENARIO_NAME}.tif", mod_map, meta)
    write_mod_legend(OUTPUT_FOLDER / f"mod_legend_{SCENARIO_NAME}.csv", variable_names)

    print(f"MESS and MoD outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
