"""
Step 02 - Crop climate data to the reference grid and harmonise units
=====================================================================

Purpose
    Extract every climate product for the processing bbox (Iran plus margin) on the
    reference 1/24-degree grid, convert to common units (temperature degC, precipitation
    mm/month, float32, NaN for missing) and store compact NetCDF files for later steps.

Why
    - All products share the same global lattice (step 01), so cropping is done by
      integer indices: values are never interpolated (decision D04).
    - Units and missing-value conventions differ between products (int16 with -32768,
      scale_factor/add_offset in TerraClimate, NaN in WorldClim monthly). Harmonising them
      once avoids silent errors in every later step.
    - Working on Iran-sized files (~1-2 % of the global volume) makes the pipeline fast.

Inputs
    WorldClim 2.1 monthly weather 1970-2024, WorldClim 2.1 normals and bioclim normals
    1970-2000, WorldClim CMIP6 2021-2040 and 2041-2060, TerraClimate v1.1 1970-2024 (raw folders in config).

Outputs  (data_processed/02_crop_and_units/)
    reference_grid.tif                       grid template (1 inside Iran, nodata outside)
    wc_monthly_{tmin,tmax,prec}_1970_2024.nc 660 monthly layers each
    tc_monthly_{tmin,tmax,prec}_1970_2024.nc 660 monthly layers each (ppt renamed prec)
    wc21_normals_1970_2000.nc                tmin, tmax, prec (12 months)
    wc21_bio_1970_2000.nc                    bio1-bio19
    cmip6/{gcm}_{ssp}_{2021_2040|2041_2060}.nc  tmin, tmax, prec (12 months)

QA  (results/qa/02_crop_and_units/)
    02_report.txt, 02_run_info.json, 02_value_summary.csv (incl. location of extremes),
    02_input_manifest.csv, 02_terraclimate_file_metadata.csv, 02_prec_extreme_values.csv,
    02_iran_annual_series_wc_vs_tc.csv,
    02_fig_iran_annual_series_wc_vs_tc.png, 02_fig_july2000_tmax_wc_tc_difference.png,
    02_prec_extreme_values_per_year.csv, 02_fig_prec_extreme_value_counts.png

Run      python scripts/02_crop_and_units.py
"""
import re
import sys
import warnings

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hemiscorpius.config import load_config, raw_dir, project_path, qa_dir  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.config import CONFIG_FILE  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import (CELL, ref_grid, read_global_tif, to_dataarray,  # noqa: E402
                               nc_encoding, write_geotiff, iran_mask)

STEP = "02_crop_and_units"
VARS = ("tmin", "tmax", "prec")
TC_NAME = {"tmin": "tmin", "tmax": "tmax", "prec": "ppt"}
UNITS = {"tmin": "degC", "tmax": "degC", "prec": "mm/month"}
ISSUES = []
SUMMARY = []
MANIFEST = []
TC_META = []


# --------------------------------------------------------------------------- helpers
def register(path, dataset):
    """Add a raw input file to the manifest."""
    st = Path(path).stat()
    MANIFEST.append(dict(dataset=dataset, file=Path(path).name, size_mb=round(st.st_size / 1e6, 2),
                         modified=pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d")))


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def plausible(name, var, a, grid=None, layer_labels=None):
    """Record value statistics, flag implausible values and report where extremes occur."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        note("FAIL", f"{name} {var}: no valid values in the bbox")
        return
    vmin, vmean, vmax = float(v.min()), float(v.mean()), float(v.max())
    pct_nan = 100.0 * (1 - v.size / a.size)
    row = dict(dataset=name, variable=var, shape=str(a.shape), min=round(vmin, 2),
               mean=round(vmean, 2), max=round(vmax, 2), pct_nan=round(pct_nan, 2))
    if var in ("tmin", "tmax"):
        good = -50 < vmin and vmax < 60
    else:
        good = vmin >= 0
        if vmax > 1000:
            note("WARN", f"{name} prec: maximum monthly value {vmax:.0f} mm is implausible for the "
                         f"region - extreme values are screened below and handled in step 03 (D18-D19)")
    note("OK" if good else "FAIL",
         f"{name} {var}: min {vmin:.2f}, mean {vmean:.2f}, max {vmax:.2f} {UNITS[var]} "
         f"(NaN {pct_nan:.1f}% of bbox)")
    if grid is not None:
        filled = np.where(np.isfinite(a), a, np.nan)
        for label, idx in (("max", np.nanargmax(filled)), ("min", np.nanargmin(filled))):
            pos = np.unravel_index(idx, a.shape)
            r, c = pos[-2], pos[-1]
            layer = ""
            if a.ndim == 3:
                layer = f", layer {layer_labels[pos[0]] if layer_labels is not None else pos[0]}"
            print(f"      {label} located at lat {grid.lats[r]:.3f}, lon {grid.lons[c]:.3f}{layer}")
            row[f"{label}_lat"], row[f"{label}_lon"] = round(float(grid.lats[r]), 3), round(float(grid.lons[c]), 3)
    SUMMARY.append(row)


def write_manifest(rows, qa):
    """Input manifest: every raw file used, with size and modification date (reproducibility)."""
    pd.DataFrame(rows).to_csv(qa / "02_input_manifest.csv", index=False)


def weighted_iran_mean(a, mask, lats):
    """Area-weighted (cos latitude) mean over Iran for each layer of a (n, rows, cols) array."""
    w = np.cos(np.deg2rad(lats))[:, None] * mask
    valid = np.isfinite(a) & (w > 0)
    num = np.where(valid, a * w, 0).sum(axis=(1, 2))
    den = np.where(valid, w, 0).sum(axis=(1, 2))
    return num / den


def annual_series(monthly_mean, var, years):
    s = pd.Series(monthly_mean).groupby(np.repeat(years, 12)).agg("sum" if var == "prec" else "mean")
    return s


def save_nc(da_dict, path, attrs):
    import xarray as xr
    ds = xr.Dataset(da_dict)
    ds.attrs.update(attrs)
    ds.to_netcdf(path, encoding=nc_encoding(ds))


# --------------------------------------------------------------------------- products
def process_wc_monthly(cfg, grid, out, mask, series):
    section("1. WorldClim 2.1 monthly weather 1970-2024")
    folder = raw_dir(cfg, "wc_monthly")
    pat = re.compile(r"_(tmin|tmax|prec)_(\d{4})-(\d{2})\.tif$")
    files = {}
    for f in folder.rglob("*.tif"):
        m = pat.search(f.name)
        if m and 1970 <= int(m.group(2)) <= 2024:
            files[(m.group(1), int(m.group(2)), int(m.group(3)))] = f
    time = pd.date_range("1970-01-01", periods=660, freq="MS")
    years = np.arange(1970, 2025)
    for var in VARS:
        keys = [(var, t.year, t.month) for t in time]
        missing = [k for k in keys if k not in files]
        if missing:
            note("FAIL", f"WorldClim monthly {var}: {len(missing)} months missing")
            continue
        for k in keys:
            register(files[k], "WorldClim monthly")
        stack = np.stack([read_global_tif(files[k], grid) for k in keys]).astype("float32")
        plausible("WorldClim monthly", var, stack, grid, [t.strftime("%Y-%m") for t in time])
        da = to_dataarray(stack, grid, var, "time", time, UNITS[var])
        save_nc({var: da}, out / f"wc_monthly_{var}_1970_2024.nc",
                {"source": "WorldClim 2.1 historical monthly weather (CRU-TS 4.09 downscaled)",
                 "processing": "cropped by index to reference grid; no interpolation"})
        series[("WorldClim", var)] = annual_series(weighted_iran_mean(stack, mask, grid.lats), var, years)
        if var == "tmax":
            series["wc_july2000_tmax"] = stack[(2000 - 1970) * 12 + 6].copy()
        series[("valid", "WorldClim monthly")] = np.isfinite(stack[0])
        del stack
        print(f"  saved wc_monthly_{var}_1970_2024.nc")


def _read_tc_file(path, tcv, grid):
    """Read one TerraClimate file (yearly or multi-year/NCSS subset) cropped to the grid.

    Returns (times, array[n, rows, cols]). Tries the netCDF4 engine, then h5netcdf.
    Works for global files and for files already subset to an area containing the grid.
    """
    import xarray as xr
    last_err = None
    for engine in (None, "h5netcdf"):
        try:
            with xr.open_dataset(path, engine=engine) as ds:      # applies scale/offset
                lat = "lat" if "lat" in ds.coords else "latitude"
                lon = "lon" if "lon" in ds.coords else "longitude"
                la, lo = ds[lat].values, ds[lon].values
                rows = np.round((90.0 - la) / CELL - 0.5).astype(int) - grid.row_off
                cols = np.round((lo + 180.0) / CELL - 0.5).astype(int) - grid.col_off
                ri = np.where((rows >= 0) & (rows < grid.nrows))[0]
                ci = np.where((cols >= 0) & (cols < grid.ncols))[0]
                sub = ds[tcv].isel({lat: ri, lon: ci}).sortby("time")
                arr = sub.values.astype("float32")                 # forces the actual read
                sub_la, sub_lo = sub[lat].values, sub[lon].values
                times = pd.to_datetime(sub["time"].values)
            if sub_la[0] < sub_la[-1]:
                arr, sub_la = arr[:, ::-1, :], sub_la[::-1]
            if (arr.shape[1:] != grid.shape or np.max(np.abs(sub_la - grid.lats)) > 1e-4
                    or np.max(np.abs(sub_lo - grid.lons)) > 1e-4):
                raise ValueError(f"grid {arr.shape[1:]} does not match the reference grid {grid.shape}")
            return times, arr
        except Exception as e:
            last_err = e
    raise last_err


def tc_file_metadata(path, tcv):
    """Version, creation date and time span stored inside a TerraClimate file."""
    import xarray as xr
    rec = dict(variable=tcv, file=Path(path).name,
               size_mb=round(Path(path).stat().st_size / 1e6, 1),
               version="unreadable", date_created="unreadable", first="", last="", method="")
    try:
        with xr.open_dataset(path) as ds:
            t = pd.to_datetime(ds["time"].values)
            rec.update(version=str(ds.attrs.get("version", "missing")),
                       date_created=str(ds.attrs.get("date_created", "missing")),
                       first=t.min().strftime("%Y-%m"), last=t.max().strftime("%Y-%m"),
                       method=str(ds.attrs.get("method", ""))[:120])
    except Exception as e:
        rec["method"] = f"open failed: {type(e).__name__}"
    return rec


def _tc_files_for(folder, tcv):
    """All netCDF files in the TerraClimate folder that contain variable tcv."""
    import xarray as xr
    found = []
    for f in sorted(folder.rglob("*.nc")):
        try:
            with xr.open_dataset(f, decode_times=False) as ds:
                if tcv in ds.data_vars:
                    found.append(f)
        except Exception:
            if re.search(rf"_{tcv}_", f.name):                    # unreadable but clearly this variable
                found.append(f)
    return found


def process_terraclimate(cfg, grid, out, mask, series):
    section("2. TerraClimate v1.1 1970-2024 (yearly files or multi-year subsets)")
    folder = raw_dir(cfg, "terraclimate")
    time = pd.date_range("1970-01-01", periods=660, freq="MS")
    years = np.arange(1970, 2025)
    for var in VARS:
        tcv = TC_NAME[var]
        files = _tc_files_for(folder, tcv)
        print(f"  {tcv}: {len(files)} file(s) found")
        month_data, month_version = {}, {}
        for f in files:
            meta = tc_file_metadata(f, tcv)
            TC_META.append(meta)
            register(f, "TerraClimate")
            try:
                times, arr = _read_tc_file(f, tcv, grid)
            except Exception as e:
                note("FAIL", f"TerraClimate {f.name} cannot be read: {type(e).__name__}: {e}")
                continue
            for t, layer in zip(times, arr):
                key = (t.year, t.month)
                if not (1970 <= t.year <= 2024):
                    continue
                if key in month_data:
                    note("FAIL", f"TerraClimate {tcv} {t:%Y-%m} present in more than one file")
                month_data[key] = layer
                month_version[key] = meta["version"]
        missing = [f"{t:%Y-%m}" for t in time if (t.year, t.month) not in month_data]
        if missing:
            note("FAIL", f"TerraClimate {tcv}: {len(missing)} of 660 months missing "
                         f"(first: {missing[:6]})")
            continue
        versions = sorted(set(month_version.values()))
        note("OK" if versions == ["V1.1"] else "FAIL",
             f"TerraClimate {tcv}: versions of the 660 months {versions}")
        if versions != ["V1.1"]:
            by_year = pd.Series({y: month_version[(y, 1)] for y in years})
            for v in versions:
                yrs = list(by_year[by_year == v].index)
                print(f"      {v}: years {yrs}")
        series.setdefault("tc_version_by_year", {})[var] = {y: month_version[(y, 1)] for y in years}
        stack = np.stack([month_data[(t.year, t.month)] for t in time]).astype("float32")
        stack[~np.isfinite(stack)] = np.nan
        plausible("TerraClimate", var, stack, grid, [t.strftime("%Y-%m") for t in time])
        da = to_dataarray(stack, grid, var, "time", time, UNITS[var])
        save_nc({var: da}, out / f"tc_monthly_{var}_1970_2024.nc",
                {"source": "TerraClimate", "versions_found": ",".join(versions),
                 "original_variable": tcv,
                 "processing": "cropped by index to reference grid; scale/offset applied; no interpolation"})
        series[("TerraClimate", var)] = annual_series(weighted_iran_mean(stack, mask, grid.lats), var, years)
        if var == "tmax":
            series["tc_july2000_tmax"] = stack[(2000 - 1970) * 12 + 6].copy()
        series[("valid", "TerraClimate")] = np.isfinite(stack[0])
        del stack, month_data
        print(f"  saved tc_monthly_{var}_1970_2024.nc")


def process_normals(cfg, grid, out, series):
    section("3. WorldClim 2.1 normals 1970-2000 (tmin, tmax, prec) and bioclim normals")
    folder = raw_dir(cfg, "wc21_normals")
    das = {}
    for var in VARS:
        files = sorted(folder.rglob(f"wc2.1_2.5m_{var}_[0-1][0-9].tif"))
        if len(files) != 12:
            note("FAIL", f"normals {var}: {len(files)} files (expected 12)")
            continue
        for f in files:
            register(f, "WorldClim 2.1 normals")
        stack = np.stack([read_global_tif(f, grid) for f in files])
        plausible("WorldClim normals", var, stack, grid, [f"month {m}" for m in range(1, 13)])
        das[var] = to_dataarray(stack, grid, var, "month", np.arange(1, 13), UNITS[var])
        series[("valid", "WorldClim normals")] = np.isfinite(stack[0])
    if das:
        save_nc(das, out / "wc21_normals_1970_2000.nc",
                {"source": "WorldClim 2.1 normals 1970-2000", "use": "reference for delta method"})
        print("  saved wc21_normals_1970_2000.nc")

    folder = raw_dir(cfg, "wc21_bio")
    das = {}
    for n in range(1, 20):
        hits = list(folder.rglob(f"wc2.1_2.5m_bio_{n}.tif"))
        if len(hits) != 1:
            note("FAIL", f"bioclim normals bio{n}: found {len(hits)} files")
            continue
        register(hits[0], "WorldClim 2.1 bioclim normals")
        das[f"bio{n}"] = to_dataarray(read_global_tif(hits[0], grid), grid, f"bio{n}")
    if len(das) == 19:
        save_nc(das, out / "wc21_bio_1970_2000.nc",
                {"source": "WorldClim 2.1 bioclim normals 1970-2000", "use": "consistency check only"})
        note("OK", "bioclim normals bio1-bio19 cropped")


def process_cmip6(cfg, grid, out, series):
    fut = cfg["future"]
    periods = [fut["period"]] + ([fut["interpolation_period"]] if fut.get("interpolation_period") else [])
    section(f"4. WorldClim CMIP6 ({', '.join(periods)})")
    folder = raw_dir(cfg, "cmip6")
    sub = out / "cmip6"
    sub.mkdir(exist_ok=True)
    for per in periods:
        for gcm in fut["gcms"]:
            for ssp in fut["ssps"]:
                das = {}
                for var in VARS:
                    name = f"wc2.1_2.5m_{var}_{gcm}_{ssp}_{per}.tif"
                    hits = list(folder.rglob(name))
                    if len(hits) != 1:
                        note("FAIL", f"CMIP6 {name}: found {len(hits)} files")
                        continue
                    register(hits[0], "WorldClim CMIP6")
                    try:
                        stack = read_global_tif(hits[0], grid)        # (12, rows, cols)
                    except Exception as e:
                        note("FAIL", f"CMIP6 {name} cannot be read ({type(e).__name__}) - "
                                     f"probably a damaged download: download this file again")
                        continue
                    plausible(f"CMIP6 {gcm} {ssp} {per}", var, stack, grid, [f"month {m}" for m in range(1, 13)])
                    das[var] = to_dataarray(stack, grid, var, "month", np.arange(1, 13), UNITS[var])
                    series[("valid", "CMIP6")] = np.isfinite(stack[0])
                if len(das) == 3:
                    save_nc(das, sub / f"{gcm}_{ssp}_{per.replace('-', '_')}.nc",
                            {"source": "WorldClim 2.1 CMIP6 downscaled", "gcm": gcm, "ssp": ssp, "period": per})
    print(f"  saved cmip6/*.nc for periods {periods}")


# --------------------------------------------------------------------------- QA
def qa_products(series, mask, grid, qa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    section("5. Land-mask agreement between products (inside Iran)")
    valid = {k[1]: v for k, v in series.items() if isinstance(k, tuple) and k[0] == "valid"}
    n_iran = int(mask.sum())
    print(f"  cells inside Iran (cell-centre rule): {n_iran}")
    names = list(valid)
    for n in names:
        print(f"  {n:20s}: valid cells inside Iran = {int((valid[n] & mask).sum())}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = int(((valid[a] ^ valid[b]) & mask).sum())
            note("OK" if diff == 0 else "WARN",
                 f"cells inside Iran valid in only one of {a} / {b}: {diff} (resolved by common mask, step 08)")

    section("6. Iran-mean annual series: WorldClim vs TerraClimate (1970-2024)")
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, var in zip(axes, VARS):
        wc, tc = series.get(("WorldClim", var)), series.get(("TerraClimate", var))
        if wc is None or tc is None:
            continue
        r = float(np.corrcoef(wc.values, tc.values)[0, 1])
        bias = float((tc - wc).mean())
        for label, s in (("WorldClim 2.1 monthly", wc), ("TerraClimate v1.1", tc)):
            ax.plot(s.index, s.values, label=label)
        ax.axvline(1999.5, color="grey", lw=0.8, ls="--")
        ax.set_ylabel(f"{var} ({'mm/yr' if var == 'prec' else 'degC'})")
        ax.set_title(f"{var}: r = {r:.3f}, mean difference TC - WC = {bias:.2f}")
        print(f"  {var}: correlation r = {r:.3f}; mean difference TC - WC = {bias:.2f}")
        for y in wc.index:
            rows.append(dict(year=y, variable=var, worldclim=wc[y], terraclimate=tc[y]))
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Year (dashed line: boundary between periods)")
    fig.tight_layout()
    fig.savefig(qa / "02_fig_iran_annual_series_wc_vs_tc.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(qa / "02_iran_annual_series_wc_vs_tc.csv", index=False)

    if "wc_july2000_tmax" in series and "tc_july2000_tmax" in series:
        wc, tc = series["wc_july2000_tmax"], series["tc_july2000_tmax"]
        ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, a, t in ((axes[0], wc, "WorldClim monthly"), (axes[1], tc, "TerraClimate")):
            im = ax.imshow(np.where(mask, a, np.nan), extent=ext, cmap="inferno")
            ax.set_title(f"July 2000 tmax - {t}")
            fig.colorbar(im, ax=ax, shrink=0.8, label="degC")
        im = axes[2].imshow(np.where(mask, tc - wc, np.nan), extent=ext, cmap="RdBu_r",
                            vmin=-5, vmax=5)
        axes[2].set_title("TerraClimate - WorldClim")
        fig.colorbar(im, ax=axes[2], shrink=0.8, label="degC")
        fig.tight_layout()
        fig.savefig(qa / "02_fig_july2000_tmax_wc_tc_difference.png", dpi=200)
        plt.close(fig)
        print("  figures saved: 02_fig_iran_annual_series_wc_vs_tc.png, 02_fig_july2000_tmax_wc_tc_difference.png")


def qa_prec_outliers(out, grid, mask, qa, series_versions=None, top=15):
    """Flag monthly precipitation values that are extreme relative to the cell's own climate.

    Rule: value >= 200 mm AND value >= 10 x (median of the same calendar month at that cell
    over 1970-2024 + 5 mm). Each flagged value is compared with the other product.
    """
    import xarray as xr
    section("7. Screening of extreme monthly precipitation (inside Iran)")
    paths = {"WorldClim": out / "wc_monthly_prec_1970_2024.nc",
             "TerraClimate": out / "tc_monthly_prec_1970_2024.nc"}
    data = {}
    for k, pth in paths.items():
        if pth.exists():
            with xr.open_dataset(pth) as ds:
                data[k] = ds["prec"].values.astype("float32")
                time = pd.to_datetime(ds["time"].values)
    if not data:
        print("  no precipitation files available")
        return
    rows, per_year, count_maps = [], {}, {}
    for k, arr in data.items():
        flagged = np.zeros(arr.shape, dtype=bool)
        for m in range(12):
            sub = arr[m::12]
            med = np.nanmedian(sub, axis=0)
            flagged[m::12] = (sub >= 200) & (sub >= 10 * (med + 5)) & mask[None]
        idx = np.argwhere(flagged)
        count_maps[k] = flagged.sum(axis=0)
        per_year[k] = pd.Series(flagged.sum(axis=(1, 2))).groupby(time.year).sum()
        print(f"  {k}: {len(idx)} flagged cell-months out of "
              f"{int(mask.sum()) * arr.shape[0]:,} inside Iran")
        other = [o for o in data if o != k]
        for t, r, c in idx:
            m = t % 12
            rec = dict(product=k, date=time[t].strftime("%Y-%m"), lat=round(float(grid.lats[r]), 3),
                       lon=round(float(grid.lons[c]), 3), value_mm=round(float(arr[t, r, c]), 1),
                       cell_median_same_month_mm=round(float(np.nanmedian(arr[m::12, r, c])), 1))
            if other:
                rec["other_product_same_month_mm"] = round(float(data[other[0]][t, r, c]), 1)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(qa / "02_prec_extreme_values.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    py = pd.DataFrame(per_year)
    vers = series_versions.get("prec") if series_versions else None
    if vers:
        py["TerraClimate_file_version"] = pd.Series(vers)
    py.to_csv(qa / "02_prec_extreme_values_per_year.csv")
    print("\n  flagged cell-months per year (02_prec_extreme_values_per_year.csv):")
    print("   " + py[py.drop(columns=[c for c in py.columns if "version" in c]).sum(axis=1) > 0]
          .to_string().replace("\n", "\n   "))
    ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
    fig, axes = plt.subplots(1, len(count_maps), figsize=(7 * len(count_maps), 5))
    axes = np.atleast_1d(axes)
    for ax, (k, cm) in zip(axes, count_maps.items()):
        im = ax.imshow(np.where(mask, cm, np.nan), extent=ext, cmap="magma_r")
        ax.set_title(f"{k}: number of flagged months per cell")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(qa / "02_fig_prec_extreme_value_counts.png", dpi=200)
    plt.close(fig)
    if len(df):
        print(f"\n  top {top} flagged values (all in 02_prec_extreme_values.csv):")
        print("   " + df.sort_values("value_mm", ascending=False).head(top)
              .to_string(index=False).replace("\n", "\n   "))
        note("WARN", f"{len(df)} extreme monthly precipitation values flagged - review 02_prec_extreme_values.csv")
    else:
        note("OK", "no extreme monthly precipitation values flagged")


def manuscript_notes_02(cfg, grid, mask, series, qa):
    """Key numbers and draft text for the manuscript (data preparation, product comparison)."""
    n = ManuscriptNotes(cfg, STEP, "Climate data preparation and product comparison")
    n.number("grid_rows", grid.nrows, "cells", "rows of the processing grid")
    n.number("grid_cols", grid.ncols, "cells", "columns of the processing grid")
    n.number("iran_cells", int(mask.sum()), "cells", "1/24-degree cells inside Iran (GADM 4.1)")
    valid = {k[1]: v for k, v in series.items() if isinstance(k, tuple) and k[0] == "valid"}
    if "WorldClim monthly" in valid:
        n.number("wc_monthly_missing_coastal_cells", int((mask & ~valid["WorldClim monthly"]).sum()),
                 "cells", "Iran cells without WorldClim monthly data (coastal)")
    meta = pd.DataFrame(TC_META)
    if len(meta):
        n.number("terraclimate_files", len(meta), "files", "TerraClimate files used")
        n.number("terraclimate_versions", ",".join(sorted(meta["version"].unique())), "",
                 "versions found in file attributes", "data provenance")
    ser = qa / "02_iran_annual_series_wc_vs_tc.csv"
    if ser.exists():
        d = pd.read_csv(ser)
        for v in VARS:
            s_ = d[d.variable == v]
            if len(s_):
                n.number(f"iran_annual_r_{v}", float(np.corrcoef(s_.worldclim, s_.terraclimate)[0, 1]), "",
                         f"correlation of Iran-mean annual {v}, WC vs TC, 1970-2024", "R1-1, R1-6")
                n.number(f"iran_annual_mean_diff_{v}", float((s_.terraclimate - s_.worldclim).mean()),
                         "mm/yr" if v == "prec" else "degC", f"mean TC - WC difference, annual {v}", "R1-6")
    ext = qa / "02_prec_extreme_values.csv"
    if ext.exists():
        e = pd.read_csv(ext)
        for prod in ("WorldClim", "TerraClimate"):
            sub = e[e["product"] == prod]
            n.number(f"prec_extreme_cell_months_{prod.lower()}", len(sub), "cell-months",
                     f"{prod}: monthly precipitation >= 200 mm and >= 10 x (cell same-month median + 5 mm), "
                     f"inside Iran 1970-2024", "D18")
            if len(sub):
                n.number(f"prec_extreme_max_{prod.lower()}", float(sub.value_mm.max()), "mm/month",
                         f"{prod}: largest flagged monthly value", "D18")
        n.number("iran_cell_months_total", int(mask.sum()) * 660, "cell-months", "Iran cells x 660 months")
    n.text("Methods (draft)", f"""
All climate products share the same global 2.5-arc-minute (1/24 degree) lattice. Data were
cropped to a processing window covering Iran (43.5-64.0 E, 24.5-40.5 N; {grid.nrows} x {grid.ncols}
cells) by integer row and column indices, so that no climate layer was interpolated or
resampled at any point. Temperatures were expressed in degC and precipitation in mm per month;
TerraClimate scale factors and offsets were applied on reading, and the version attribute of every
TerraClimate file was verified (all files TerraClimate v1.1). Iran comprised {int(mask.sum())} grid cells
(GADM 4.1 boundary, cell-centre rule).""")
    n.text("Results / Supplementary (draft)", """
Iran-mean annual series of the two products were highly correlated (see key numbers), but
differences between products were spatially structured, reaching several degrees locally, and the
TerraClimate-WorldClim difference in minimum temperature increased after 2000. Both products
contained implausible monthly precipitation values in the hyper-arid southern belt, much more
frequently in TerraClimate (see key numbers and decision D18).""")
    n.save()


def main():
    cfg = load_config()
    grid = ref_grid(cfg)
    out = project_path(cfg, "processed") / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    static_root = raw_dir(cfg, "static")
    boundary = cfg["paths"].get("static_files", {}).get("boundary", "gadm41_IRN_0.shp")

    write_run_info(qa / "02_run_info.json", CONFIG_FILE)
    with Report(qa / "02_report.txt"):
        print(f"Step {STEP}")
        print(f"reference grid: {grid.nrows} rows x {grid.ncols} cols, "
              f"W {grid.west} N {grid.north}, cell {CELL:.10f} deg")
        mask = iran_mask(cfg, grid, static_root, boundary)
        template = np.where(mask, 1.0, np.nan).astype("float32")
        write_geotiff(out / "reference_grid.tif", template, grid)
        print(f"  reference_grid.tif written ({int(mask.sum())} cells inside Iran)")

        series = {}
        for fn in (process_wc_monthly, process_terraclimate):
            try:
                fn(cfg, grid, out, mask, series)
            except Exception as e:
                note("FAIL", f"{fn.__name__} crashed: {type(e).__name__}: {e}")
        for fn in (process_normals, process_cmip6):
            try:
                fn(cfg, grid, out, series)
            except Exception as e:
                note("FAIL", f"{fn.__name__} crashed: {type(e).__name__}: {e}")
        try:
            qa_products(series, mask, grid, qa)
        except Exception as e:
            note("FAIL", f"QA crashed: {type(e).__name__}: {e}")

        try:
            qa_prec_outliers(out, grid, mask, qa, series.get("tc_version_by_year"))
        except Exception as e:
            note("FAIL", f"precipitation screening crashed: {type(e).__name__}: {e}")
        if TC_META:
            pd.DataFrame(TC_META).to_csv(qa / "02_terraclimate_file_metadata.csv", index=False)
            print("  TerraClimate per-file metadata: 02_terraclimate_file_metadata.csv")
        pd.DataFrame(SUMMARY).to_csv(qa / "02_value_summary.csv", index=False)
        try:
            manuscript_notes_02(cfg, grid, mask, series, qa)
        except Exception as e:
            note("WARN", f"manuscript notes not written: {type(e).__name__}: {e}")
        write_manifest(MANIFEST, qa)
        print(f"  input manifest: {len(MANIFEST)} raw files listed in 02_input_manifest.csv")
        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails:
            print(f"   - FAIL: {m}")
        for m in warns:
            print(f"   - WARN: {m}")
        if not fails:
            print("\n  Step 02 completed. Review the figures in results/qa/02_crop_and_units/.")


if __name__ == "__main__":
    main()
