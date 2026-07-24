"""
Script: 07_future_maxent_ensemble_mean.py

Purpose:
Create ensemble mean, ensemble standard deviation, and binary suitability maps
from MaxEnt projections produced for multiple future GCMs.

Notes:
- Edit INPUT_FOLDER, OUTPUT_FOLDER, THRESHOLD, and SCENARIO_NAME before running.
- Input files are expected to be continuous MaxEnt outputs for the same scenario
  and harmonized to the same grid.
"""

from pathlib import Path

import numpy as np
import rasterio


INPUT_FOLDER = Path("path/to/future_maxent_predictions")
OUTPUT_FOLDER = Path("path/to/future_ensemble_outputs")
THRESHOLD = 0.3427
SCENARIO_NAME = "SSP5_8_5"
OUTPUT_NODATA = -9999


def read_prediction_stack(input_folder: Path) -> tuple[np.ndarray, dict]:
    """Read all continuous prediction rasters and return a stacked array and metadata."""
    files = sorted(
        path for path in input_folder.glob("*.asc")
        if "clamping" not in path.name.lower()
    )

    if not files:
        raise FileNotFoundError(f"No prediction ASCII files were found in: {input_folder}")

    arrays = []
    output_meta = None

    for path in files:
        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            arrays.append(data)

            if output_meta is None:
                output_meta = src.meta.copy()
                output_meta.update(driver="GTiff", dtype="float32", nodata=OUTPUT_NODATA, count=1)

    return np.stack(arrays), output_meta


def write_raster(path: Path, array: np.ndarray, meta: dict) -> None:
    """Write one raster output."""
    output = array.astype("float32")
    output[np.isnan(output)] = OUTPUT_NODATA

    with rasterio.open(path, "w", **meta) as dst:
        dst.write(output, 1)


def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    stack, meta = read_prediction_stack(INPUT_FOLDER)

    ensemble_mean = np.nanmean(stack, axis=0)
    ensemble_std = np.nanstd(stack, axis=0)
    binary_map = np.where(ensemble_mean >= THRESHOLD, 1.0, 0.0)
    binary_map[np.isnan(ensemble_mean)] = np.nan

    write_raster(OUTPUT_FOLDER / f"mean_{SCENARIO_NAME}.tif", ensemble_mean, meta)
    write_raster(OUTPUT_FOLDER / f"std_{SCENARIO_NAME}.tif", ensemble_std, meta)
    write_raster(OUTPUT_FOLDER / f"binary_{SCENARIO_NAME}.tif", binary_map, meta)

    print(f"Future ensemble outputs saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
