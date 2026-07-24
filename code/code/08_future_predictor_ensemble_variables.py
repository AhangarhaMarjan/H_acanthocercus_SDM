"""
Script: 08_future_predictor_ensemble_variables.py

Purpose:
Average future bioclimatic predictor rasters across multiple GCM folders to
produce ensemble predictor layers for MaxEnt projection and MESS/MoD analysis.

Notes:
- Edit BASE_FOLDER, OUTPUT_FOLDER, and VARIABLES before running.
- Predictor rasters must be aligned to the same grid before averaging.
"""

from pathlib import Path

import numpy as np
import rasterio


BASE_FOLDER = Path("path/to/future_gcm_predictor_folders")
OUTPUT_FOLDER = Path("path/to/future_predictor_ensemble_layers")
VARIABLES = ["bio3", "bio4", "bio8", "bio12"]
OUTPUT_NODATA = -9999


def find_variable_files(base_folder: Path, variable_name: str) -> list[Path]:
    """Find raster files for one predictor across GCM folders."""
    candidates = []
    for path in base_folder.rglob("*.asc"):
        stem = path.stem.lower()
        if stem == variable_name.lower() or stem.endswith(f"_{variable_name.lower()}"):
            candidates.append(path)
    return sorted(candidates)


def average_variable(files: list[Path]) -> tuple[np.ndarray, dict]:
    """Average a set of aligned rasters for one predictor variable."""
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
                output_meta.update(dtype="float32", nodata=OUTPUT_NODATA, count=1)

    mean_array = np.nanmean(np.stack(arrays), axis=0)
    return mean_array, output_meta


def main() -> None:
    if not BASE_FOLDER.exists():
        raise FileNotFoundError(f"Base folder not found: {BASE_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    for variable in VARIABLES:
        files = find_variable_files(BASE_FOLDER, variable)
        if not files:
            print(f"No files found for variable: {variable}")
            continue

        mean_array, meta = average_variable(files)
        mean_array[np.isnan(mean_array)] = OUTPUT_NODATA
        output_path = OUTPUT_FOLDER / f"{variable}.asc"

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mean_array.astype("float32"), 1)

        print(f"Created ensemble predictor for {variable} from {len(files)} files.")

    print(f"Future predictor ensemble layers saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
