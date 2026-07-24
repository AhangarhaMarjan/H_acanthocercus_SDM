"""
Script: 02_export_rasters_to_ascii_for_maxent.py

Purpose:
Export GeoTIFF predictor rasters to ASCII format for MaxEnt input.

Notes:
- Edit INPUT_FOLDER and OUTPUT_FOLDER before running.
- The script assumes north-up rasters with square pixels, which is the usual
  format for harmonized environmental layers used in MaxEnt.
"""

from pathlib import Path

import numpy as np
import rasterio


# Edit these paths before running.
INPUT_FOLDER = Path("path/to/geotiff_predictors")
OUTPUT_FOLDER = Path("path/to/ascii_predictors")
DEFAULT_NODATA = -9999


def write_ascii_grid(tif_path: Path, asc_path: Path) -> None:
    """Convert one GeoTIFF raster to an ESRI ASCII grid."""
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA

        with asc_path.open("w", encoding="utf-8") as file:
            file.write(f"ncols         {src.width}\n")
            file.write(f"nrows         {src.height}\n")
            file.write(f"xllcorner     {src.bounds.left}\n")
            file.write(f"yllcorner     {src.bounds.bottom}\n")
            file.write(f"cellsize      {src.res[0]}\n")
            file.write(f"NODATA_value  {nodata}\n")
            np.savetxt(file, data, fmt="%g")


def main() -> None:
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_FOLDER}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(INPUT_FOLDER.glob("*.tif"))

    if not tif_files:
        print(f"No TIFF files were found in: {INPUT_FOLDER}")
        return

    for tif_path in tif_files:
        asc_path = OUTPUT_FOLDER / f"{tif_path.stem.lower()}.asc"
        try:
            write_ascii_grid(tif_path, asc_path)
            print(f"Converted: {tif_path.name} -> {asc_path.name}")
        except Exception as exc:
            print(f"Failed to convert {tif_path.name}: {exc}")

    print("Raster-to-ASCII conversion finished.")


if __name__ == "__main__":
    main()
