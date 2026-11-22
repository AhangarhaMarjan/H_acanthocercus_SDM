"""
Reference grid of the project and helpers to read/write data on it.

The reference grid is the native TerraClimate / WorldClim 2.5-arc-min lattice
(1/24 degree, origin at -180 W, 90 N), cropped to the processing bbox in config.
Because every climate product shares this lattice (verified in step 01), cropping is
done by integer row/column indices: no interpolation is ever applied to climate data.
"""
from dataclasses import dataclass

import numpy as np

CELL = 1.0 / 24.0


@dataclass
class RefGrid:
    west: float
    north: float
    ncols: int
    nrows: int
    col_off: int          # column offset in the global 8640-column lattice
    row_off: int          # row offset in the global 4320-row lattice
    lons: np.ndarray      # cell-centre longitudes (west -> east)
    lats: np.ndarray      # cell-centre latitudes (north -> south)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.west, self.north, CELL, CELL)

    @property
    def shape(self):
        return (self.nrows, self.ncols)


def ref_grid(cfg: dict) -> RefGrid:
    """Build the reference grid from config 'grid.bbox' (snapped to the 1/24 lattice)."""
    bb = cfg["grid"]["bbox"]
    col0 = int(round((bb["west"] + 180.0) / CELL))
    col1 = int(round((bb["east"] + 180.0) / CELL))
    row0 = int(round((90.0 - bb["north"]) / CELL))
    row1 = int(round((90.0 - bb["south"]) / CELL))
    west = -180.0 + col0 * CELL
    north = 90.0 - row0 * CELL
    ncols, nrows = col1 - col0, row1 - row0
    lons = west + (np.arange(ncols) + 0.5) * CELL
    lats = north - (np.arange(nrows) + 0.5) * CELL
    return RefGrid(west, north, ncols, nrows, col0, row0, lons, lats)


def global_window(src, grid: RefGrid):
    """Rasterio Window of the reference grid inside a global 1/24-degree raster.

    Raises if the raster is not on the global lattice (origin -180/90, cell 1/24).
    """
    from rasterio.windows import Window
    t = src.transform
    if not (abs(t.a - CELL) < 1e-9 and abs(-t.e - CELL) < 1e-9
            and abs(t.c + 180.0) < 1e-6 and abs(t.f - 90.0) < 1e-6):
        raise ValueError(f"{src.name}: not on the global 1/24-degree lattice ({t})")
    return Window(grid.col_off, grid.row_off, grid.ncols, grid.nrows)


def read_global_tif(path, grid: RefGrid, band=None) -> np.ndarray:
    """Read a global 1/24-degree GeoTIFF cropped to the grid as float32 with NaN for nodata.

    Returns (rows, cols) if band is given or the file has 1 band, else (bands, rows, cols).
    """
    import rasterio
    with rasterio.open(path) as src:
        win = global_window(src, grid)
        if band is not None:
            a = src.read(band, window=win, masked=True)
        elif src.count == 1:
            a = src.read(1, window=win, masked=True)
        else:
            a = src.read(window=win, masked=True)
    out = a.astype("float32").filled(np.nan)
    # some products store nodata as huge floats without declaring it
    out[~np.isfinite(out) | (np.abs(out) > 1e20)] = np.nan
    return out


def to_dataarray(data: np.ndarray, grid: RefGrid, name: str, dim0: str = None,
                 coord0=None, units: str = None):
    """Wrap a (rows, cols) or (n, rows, cols) array on the reference grid as an xarray DataArray."""
    import xarray as xr
    coords = {"lat": grid.lats, "lon": grid.lons}
    if data.ndim == 2:
        da = xr.DataArray(data, dims=("lat", "lon"), coords=coords, name=name)
    else:
        coords = {dim0: coord0, **coords}
        da = xr.DataArray(data, dims=(dim0, "lat", "lon"), coords=coords, name=name)
    if units:
        da.attrs["units"] = units
    return da


def nc_encoding(ds) -> dict:
    """Compressed float32 encoding with -9999 fill value for every data variable."""
    return {v: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": -9999.0}
            for v in ds.data_vars}


def write_geotiff(path, data: np.ndarray, grid: RefGrid, nodata: float = -9999.0):
    """Write a (rows, cols) or (bands, rows, cols) float array (NaN = nodata) on the grid."""
    import rasterio
    arr = data if data.ndim == 3 else data[None, ...]
    arr = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    profile = dict(driver="GTiff", height=grid.nrows, width=grid.ncols, count=arr.shape[0],
                   dtype="float32", crs="EPSG:4326", transform=grid.transform,
                   nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def iran_mask(cfg: dict, grid: RefGrid, static_root, boundary_name: str) -> np.ndarray:
    """Boolean mask (True = inside Iran) from the boundary polygon, cell-centre rule.

    Used here only for QA summaries; the analysis mask is built in step 08.
    """
    import geopandas as gpd
    from rasterio.features import geometry_mask
    from pathlib import Path
    hits = list(Path(static_root).rglob(boundary_name))
    if len(hits) != 1:
        raise FileNotFoundError(f"boundary file '{boundary_name}' found {len(hits)} times")
    g = gpd.read_file(hits[0]).to_crs("EPSG:4326")
    return geometry_mask(g.geometry, out_shape=grid.shape, transform=grid.transform,
                         invert=True, all_touched=False)


def cell_area_km2(grid: RefGrid) -> np.ndarray:
    """Geodesic (spherical, authalic radius) area of every grid cell in km2, shape (rows, cols)."""
    r = 6371.0072                                    # authalic Earth radius (km)
    lat_top = grid.lats + CELL / 2
    lat_bot = grid.lats - CELL / 2
    band = (r ** 2) * np.deg2rad(CELL) * (np.sin(np.deg2rad(lat_top)) - np.sin(np.deg2rad(lat_bot)))
    return np.repeat(band[:, None], grid.ncols, axis=1)
