"""
Script: 01_standardize_ascii_headers_and_nodata.py

Purpose:
Standardize ASCII raster headers and NoData values before using the layers in
MaxEnt. This script was used after raster alignment and masking in GIS software.

Notes:
- Edit ASCII_FOLDER and STANDARD_HEADER before running.
- This script does not resample rasters; it only rewrites the ASCII header and
  replaces extreme NoData-like values with -9999.
"""

from pathlib import Path

import numpy as np


# Edit this path before running.
ASCII_FOLDER = Path("path/to/aligned_ascii_layers")

# Header of the final harmonized grid used in the MaxEnt workflow.
STANDARD_HEADER = (
    "ncols         463\n"
    "nrows         354\n"
    "xllcorner     44.0416666666667\n"
    "yllcorner     25.0416666666667\n"
    "cellsize      0.0416666666666667\n"
    "NODATA_value  -9999"
)


def standardize_ascii_file(file_path: Path) -> None:
    """Rewrite one ASCII raster using the final header and a consistent NoData value."""
    data = np.loadtxt(file_path, skiprows=6)

    # Some exported rasters store NoData as very large or very small floating values.
    data[data < -1e30] = -9999
    data[data > 1e30] = -9999

    with file_path.open("w", encoding="utf-8") as file:
        file.write(STANDARD_HEADER + "\n")
        np.savetxt(file, data, fmt="%g")


def main() -> None:
    if not ASCII_FOLDER.exists():
        raise FileNotFoundError(f"Input folder not found: {ASCII_FOLDER}")

    ascii_files = sorted(ASCII_FOLDER.glob("*.asc"))
    if not ascii_files:
        print(f"No ASCII files were found in: {ASCII_FOLDER}")
        return

    for file_path in ascii_files:
        try:
            standardize_ascii_file(file_path)
            print(f"Standardized: {file_path.name}")
        except Exception as exc:
            print(f"Skipped {file_path.name}: {exc}")

    print("ASCII header and NoData standardization finished.")


if __name__ == "__main__":
    main()
