"""
Step 07 - Static predictors (terrain and soil) on the reference grid
====================================================================

Purpose
    Bring the static candidate predictors onto the 1/24-degree reference grid:
      terrain (Copernicus DEM GLO-30, 30 m): elevation, slope, northness, eastness, VRM
      soil (SoilGrids 2.0, 0-5 cm, ~250 m):  sand, clay, soc, phh2o, bdod, cfvo

Method (decisions D07, D08, D14, D42)
    - Terrain derivatives were computed at 30 m (ArcGIS Pro, geodesic) before aggregation, because
      computing them after aggregation underestimates slope (D07); aspect is used as northness/eastness
      (D08).
    - VRM, the Vector Ruggedness Measure (Sappington et al. 2007), is computed here at 30 m with a 3 x 3
      window from slope, northness and eastness (surface normals), then aggregated. VRM measures the
      variability of surface orientation and is designed to be largely independent of slope (D42).
    - Aggregation: area-weighted average (GDAL 'average') of all valid 30 m / 250 m cells within each
      1/24-degree cell, processed in row blocks to limit memory.
    - SoilGrids: value 0 marks no data (identical 15.3 % in all layers); cells with bdod = 0 are set to
      no data in every soil layer before aggregation (D14). Mapped units are converted to conventional
      units: sand, clay, cfvo % ; soc g/kg ; phh2o pH ; bdod g/cm3.

Checks
    Value ranges, share of Iran cells with data, Spearman correlations among static predictors
    (in particular VRM vs slope), values at occurrence records.

Inputs   static folder (config paths.raw.static)
Outputs  data_processed/07_static_layers/{variable}.tif and static_predictors.nc
QA       results/qa/07_static_layers/  07_report.txt, 07_run_info.json, 07_*.csv, 07_fig_*.png
Manuscript results/manuscript/tables/T07_static_at_occurrences.csv, notes/07_static_layers.md
Run      python scripts/07_static_layers.py      (terrain part reads ~60 GB: 30-90 minutes)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, raw_dir, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import CELL, ref_grid, to_dataarray, nc_encoding, write_geotiff  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402
from hemiscorpius.terrain import vrm as compute_vrm  # noqa: E402

STEP = "07_static_layers"
TERRAIN_KEYS = {"elevation": ("elev", "dem"), "slope": ("slope",), "northness": ("north",),
                "eastness": ("east",)}
SOIL = {  # name: (divisor to conventional unit, unit)
    "sand": (10.0, "%"), "clay": (10.0, "%"), "cfvo": (10.0, "%"),
    "soc": (10.0, "g/kg"), "phh2o": (10.0, "pH"), "bdod": (100.0, "g/cm3")}
UNITS = {"elevation": "m", "slope": "degrees", "northness": "-1..1", "eastness": "-1..1", "vrm": "0..1",
         **{k: v[1] for k, v in SOIL.items()}}
RANGES = {"elevation": (-100, 6000), "slope": (0, 90), "northness": (-1, 1), "eastness": (-1, 1),
          "vrm": (0, 1), "sand": (0, 100), "clay": (0, 100), "cfvo": (0, 100), "soc": (0, 300),
          "phh2o": (3, 11), "bdod": (0.2, 2.2)}
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def find_files(static, keys):
    tifs = [p for p in Path(static).rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
    out = {}
    for name, kws in keys.items():
        hits = [f for f in tifs if any(k in f.name.lower() for k in kws)]
        if len(hits) != 1:
            raise FileNotFoundError(f"{name}: expected 1 file, found {[h.name for h in hits]}")
        out[name] = hits[0]
    return out


def clean(a):
    a = a.astype("float32")
    a[~np.isfinite(a) | (np.abs(a) > 1e20)] = np.nan
    return a


# --------------------------------------------------------------------------- terrain
def aggregate_terrain(files, grid, block_rows, window):
    """Single pass over the 30 m rasters in row blocks: aggregate the four layers and VRM."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import Window
    halo = window // 2
    names = list(TERRAIN_KEYS) + ["vrm"]
    out = {n: np.full(grid.shape, np.nan, dtype="float32") for n in names}
    srcs = {n: rasterio.open(f) for n, f in files.items()}
    try:
        ref = srcs["slope"]
        for n, s in srcs.items():
            if s.transform != ref.transform or s.shape != ref.shape:
                raise ValueError(f"{n} is not on the same 30 m grid as slope")
        t0 = time.time()
        for r0 in range(0, grid.nrows, block_rows):
            r1 = min(r0 + block_rows, grid.nrows)
            north_b, south_b = grid.north - r0 * CELL, grid.north - r1 * CELL
            east = grid.west + grid.ncols * CELL
            col_l, row_t = ~ref.transform * (grid.west, north_b)     # inverse affine -> (col, row)
            col_r, row_b = ~ref.transform * (east, south_b)
            col0, row0 = int(np.floor(col_l)) - halo - 1, int(np.floor(row_t)) - halo - 1
            col1, row1 = int(np.ceil(col_r)) + halo + 1, int(np.ceil(row_b)) + halo + 1
            win = Window(col0, row0, col1 - col0, row1 - row0)
            data = {n: clean(s.read(1, window=win, boundless=True, fill_value=np.nan)) for n, s in srcs.items()}
            v = compute_vrm(data["slope"], data["northness"], data["eastness"], window)
            data["vrm"] = v
            src_t = ref.window_transform(win)
            dst_t = from_origin(grid.west, north_b, CELL, CELL)
            for n in names:
                dst = np.full((r1 - r0, grid.ncols), np.nan, dtype="float32")
                reproject(data[n], dst, src_transform=src_t, src_crs="EPSG:4326", src_nodata=np.nan,
                          dst_transform=dst_t, dst_crs="EPSG:4326", dst_nodata=np.nan,
                          resampling=Resampling.average)
                out[n][r0:r1] = dst
            if (r0 // block_rows) % 12 == 0:
                done = r1 / grid.nrows
                print(f"    terrain: {done:5.1%} done ({(time.time() - t0) / 60:.1f} min)")
    finally:
        for s in srcs.values():
            s.close()
    return out


# --------------------------------------------------------------------------- soil
def aggregate_soil(static, grid):
    import rasterio
    from rasterio.warp import reproject, Resampling
    tifs = [p for p in Path(static).rglob("*") if p.suffix.lower() in (".tif", ".tiff")]
    files = {}
    for name in SOIL:
        hits = [f for f in tifs if f.name.lower().startswith(name)]
        if len(hits) != 1:
            raise FileNotFoundError(f"soil {name}: expected 1 file, found {[h.name for h in hits]}")
        files[name] = hits[0]
    with rasterio.open(files["bdod"]) as s:
        nodata_mask = s.read(1) == 0
        transform, shape = s.transform, s.shape
    out = {}
    for name, (div, _) in SOIL.items():
        with rasterio.open(files[name]) as s:
            if s.transform != transform or s.shape != shape:
                raise ValueError(f"soil {name} is not on the same grid as bdod")
            a = s.read(1).astype("float32") / div
        a[nodata_mask] = np.nan
        dst = np.full(grid.shape, np.nan, dtype="float32")
        reproject(a, dst, src_transform=transform, src_crs="EPSG:4326", src_nodata=np.nan,
                  dst_transform=grid.transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.average)
        out[name] = dst
    return out, float(nodata_mask.mean())


# --------------------------------------------------------------------------- QA and outputs
def qa_layers(layers, mask, occ, qa, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    section("3. Value ranges and coverage inside Iran")
    rows = []
    for n, a in layers.items():
        v = a[mask]
        fin = v[np.isfinite(v)]
        lo, hi = RANGES[n]
        share = fin.size / v.size
        rows.append(dict(variable=n, unit=UNITS[n], min=float(fin.min()), mean=float(fin.mean()),
                         max=float(fin.max()), share_iran_cells_with_data=share))
        good = fin.min() >= lo - 1e-6 and fin.max() <= hi + 1e-6
        note("OK" if good else "FAIL", f"{n}: {fin.min():.3f} to {fin.max():.3f} {UNITS[n]} (expected {lo}..{hi})")
        note("OK" if share > 0.995 else "WARN", f"{n}: data in {share:.2%} of Iran cells")
    pd.DataFrame(rows).to_csv(qa / "07_value_summary.csv", index=False)

    section("4. Spearman correlations among static predictors (inside Iran)")
    names = list(layers)
    stack = np.stack([layers[n][mask] for n in names], axis=1)
    keep = np.isfinite(stack).all(axis=1)
    rho = spearmanr(stack[keep]).correlation
    cor = pd.DataFrame(rho, index=names, columns=names)
    cor.to_csv(qa / "07_spearman_static.csv")
    print("   " + cor.round(2).to_string().replace("\n", "\n   "))
    r_vs = float(cor.loc["vrm", "slope"])
    note("OK" if abs(r_vs) < 0.8 else "WARN",
         f"Spearman r(VRM, slope) = {r_vs:.2f} (|r| < 0.8 means VRM adds information beyond slope)")

    r, c = occ["row"].values, occ["col"].values
    tab = occ[["lon", "lat"]].copy()
    tab.insert(0, "record", np.arange(1, len(occ) + 1))
    for n in names:
        tab[f"{n}_{UNITS[n]}"] = np.round(layers[n][r, c], 3)
    tab.to_csv(qa / "07_static_at_occurrences.csv", index=False)
    tab.to_csv(tables / "T07_static_at_occurrences.csv", index=False)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax, n in zip(axes.ravel(), names):
        im = ax.imshow(np.where(mask, layers[n], np.nan), cmap="viridis")
        ax.scatter(c, r, s=4, c="red")
        ax.set_title(f"{n} ({UNITS[n]})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(qa / "07_fig_static_predictors.png", dpi=170)
    plt.close(fig)
    return pd.DataFrame(rows), cor


def save_layers(layers, out, grid):
    import xarray as xr
    for n, a in layers.items():
        write_geotiff(out / f"{n}.tif", a, grid)
    ds = xr.Dataset({n: to_dataarray(a, grid, n, units=UNITS[n]) for n, a in layers.items()})
    ds.attrs.update(terrain="Copernicus DEM GLO-30 derivatives (30 m) averaged to 1/24 degree; VRM 3x3 at 30 m",
                    soil="SoilGrids 2.0 0-5 cm (~250 m) averaged to 1/24 degree; 0 = no data (bdod mask)")
    ds.to_netcdf(out / "static_predictors.nc", encoding=nc_encoding(ds))


def notes_07(cfg, summ, cor, soil_nodata_share, window):
    n = ManuscriptNotes(cfg, STEP, "Static predictors on the reference grid")
    n.number("vrm_window_cells", window, "cells", "VRM window at 30 m", "D42")
    n.number("spearman_vrm_slope", round(float(cor.loc["vrm", "slope"]), 3), "",
             "Spearman correlation between VRM and slope inside Iran", "D42")
    n.number("soilgrids_nodata_share", round(soil_nodata_share, 4), "fraction",
             "share of SoilGrids pixels with value 0 treated as no data", "D14")
    for _, row in summ.iterrows():
        n.number(f"{row.variable}_iran_mean", round(row["mean"], 3), row.unit, f"{row.variable}: mean inside Iran", "")
    n.text("Methods (draft)", f"""
Terrain predictors were derived from the Copernicus DEM GLO-30. Slope and aspect were computed at the native
30 m resolution with the geodesic method (ArcGIS Pro) and aspect was decomposed into northness (cosine) and
eastness (sine). Terrain ruggedness was quantified with the Vector Ruggedness Measure (VRM; Sappington et al.
2007), computed at 30 m in a {window} x {window} window from the surface-normal vectors; VRM measures the
dispersion of surface orientation. In these data it was strongly correlated with slope (Spearman r = {cor.loc['vrm', 'slope']:.2f}
inside Iran), so independence from slope must not be assumed. It was considered because H. acanthocercus is a lapidicolous species. Soil predictors
(sand, clay, soil organic carbon, pH, bulk density and coarse-fragment content, 0-5 cm) were taken from
SoilGrids 2.0; cells without soil data (value 0) were masked. All static layers were aggregated to the
1/24-degree climate grid by area-weighted averaging.""")
    n.save()


def main():
    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_run_info(qa / "07_run_info.json", CONFIG_FILE)
    window = int(cfg["predictors"].get("vrm_window", 3))
    block_rows = int(cfg["predictors"].get("terrain_block_rows", 2))
    static = raw_dir(cfg, "static")

    with Report(qa / "07_report.txt"):
        print(f"Step {STEP}")
        import rasterio
        with rasterio.open(proc / "02_crop_and_units" / "reference_grid.tif") as r_:
            mask = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)

        section("1. Terrain (30 m -> 1/24 degree) incl. VRM")
        try:
            files = find_files(static, TERRAIN_KEYS)
            for k, f in files.items():
                print(f"  {k}: {f.name}")
            terrain = aggregate_terrain(files, grid, block_rows, window)
        except Exception as e:
            note("FAIL", f"terrain failed: {type(e).__name__}: {e}")
            terrain = {}

        section("2. Soil (SoilGrids ~250 m -> 1/24 degree)")
        try:
            soil, soil_nd = aggregate_soil(static, grid)
            print(f"  SoilGrids no-data share (bdod = 0): {soil_nd:.1%}")
        except Exception as e:
            note("FAIL", f"soil failed: {type(e).__name__}: {e}")
            soil, soil_nd = {}, np.nan

        layers = {**terrain, **soil}
        if len(layers) == 11:
            save_layers(layers, out, grid)
            print(f"  saved 11 layers to data_processed/{STEP}/")
            try:
                summ, cor = qa_layers(layers, mask, occ, qa, tables)
                notes_07(cfg, summ, cor, soil_nd, window)
            except Exception as e:
                note("FAIL", f"QA failed: {type(e).__name__}: {e}")
        else:
            note("FAIL", f"only {len(layers)} of 11 layers produced")

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails:
            print(f"   - FAIL: {m}")
        for m in warns:
            print(f"   - WARN: {m}")
        if not fails:
            print("\n  Step 07 completed. Review results/qa/07_static_layers/.")


if __name__ == "__main__":
    main()
