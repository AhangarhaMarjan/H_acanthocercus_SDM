"""
Script: 06_threshold_and_create_binary_maps.py

Purpose:
Convert continuous MaxEnt cloglog suitability outputs to binary suitable /
unsuitable maps using the selected 10th percentile training presence threshold.

Notes:
- Edit INPUT_RASTER, OUTPUT_RASTER, and THRESHOLD before running.
- The output uses 1 = suitable, 0 = unsuitable, and 255 = NoData.
"""

from pathlib import Path

import numpy as np
import rasterio


INPUT_RASTER = Path("path/to/maxent_prediction.asc")
OUTPUT_RASTER = Path("path/to/binary_prediction.tif")
THRESHOLD = 0.3427
OUTPUT_NODATA = 255


def create_binary_map(input_raster: Path, output_raster: Path, threshold: float) -> None:
    """Create a binary suitability raster from a continuous suitability raster."""
    with rasterio.open(input_raster) as src:
        data = src.read(1).astype("float32")
        meta = src.meta.copy()
        nodata = src.nodata

        nodata_mask = np.zeros(data.shape, dtype=bool)
        if nodata is not None:
            nodata_mask |= data == nodata
        nodata_mask |= np.isnan(data)

        binary_data = np.where(data >= threshold, 1, 0).astype("uint8")
        binary_data[nodata_mask] = OUTPUT_NODATA

        meta.update(driver="GTiff", dtype="uint8", nodata=OUTPUT_NODATA, count=1)
        output_raster.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_raster, "w", **meta) as dst:
            dst.write(binary_data, 1)


def main() -> None:
    if not INPUT_RASTER.exists():
        raise FileNotFoundError(f"Input raster not found: {INPUT_RASTER}")

    create_binary_map(INPUT_RASTER, OUTPUT_RASTER, THRESHOLD)
    print(f"Binary map saved to: {OUTPUT_RASTER}")


if __name__ == "__main__":
    main()
