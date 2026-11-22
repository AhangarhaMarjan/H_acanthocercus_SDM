"""
Step 01 - Check raw data
========================

Purpose
    Verify every raw input before any processing: completeness, naming, temporal
    coverage, CRS, resolution, units, value ranges and grid alignment.

Why
    Missing months, wrong units, mixed TerraClimate versions or shifted grids would
    propagate silently to all later steps. This step makes the input state explicit,
    auditable and reproducible.

Inputs   raw-data folders listed in config/config.yaml (paths.raw)
Output   results/qa/01_check_raw_data/01_report.txt, 01_run_info.json
Run      python scripts/01_check_raw_data.py   (from the project root)

Nothing is written outside results/qa/. Raw data are opened read-only.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hemiscorpius.config import load_config, raw_dir, qa_dir  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.config import CONFIG_FILE  # noqa: E402

STEP = "01_check_raw_data"
CELL = 1 / 24
ISSUES = []          # (level, message) collected for the final summary
OFFSETS = {}         # dataset -> (x_offset, y_offset) from the 1/24-degree lattice


# --------------------------------------------------------------------------- helpers
def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def files_in(folder, exts):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in exts)


def lattice_offset(x):
    off = (x + 180.0) % CELL
    return min(off, CELL - off)


def tif_meta(path, label=None, record_offset=True):
    import rasterio
    with rasterio.open(path) as r:
        b, t = r.bounds, r.transform
        print(f"    file:   {path.name}")
        print(f"    crs:    {r.crs}")
        print(f"    size:   {r.height} rows x {r.width} cols x {r.count} band(s)")
        print(f"    res:    {t.a:.10f} x {-t.e:.10f}")
        print(f"    dtype:  {r.dtypes[0]}   nodata: {r.nodata}")
        print(f"    bounds: W {b.left:.5f}  S {b.bottom:.5f}  E {b.right:.5f}  N {b.top:.5f}")
        if r.crs is not None and r.crs.is_geographic:
            off = (lattice_offset(b.left), lattice_offset(b.top))
            print(f"    offset from global 1/24-degree lattice: x {off[0]:.2e}, y {off[1]:.2e}")
            if record_offset and label:
                OFFSETS[label] = off
        return r.crs, (t.a, -t.e), r.count


def sampled_stats(path, max_side=2000):
    """Min/mean/max from a decimated read (fast, suitable for very large rasters)."""
    import rasterio
    with rasterio.open(path) as r:
        f = max(1, int(np.ceil(max(r.height, r.width) / max_side)))
        a = r.read(1, out_shape=(max(1, r.height // f), max(1, r.width // f)),
                   masked=True).astype("float64")
    a = a.compressed()
    if a.size == 0:
        return None
    return float(a.min()), float(a.mean()), float(a.max())


def find_one(files, keywords):
    hits = [f for f in files if any(k in f.name.lower() for k in keywords)]
    return hits


def check_folder_exists(cfg, key):
    p = raw_dir(cfg, key)
    if not p.exists():
        note("FAIL", f"{key}: folder not found -> {p}")
        return None
    zips = files_in(p, {".zip"})
    if zips:
        note("WARN", f"{key}: {len(zips)} .zip file(s) still present; only extracted files are checked")
    return p


# --------------------------------------------------------------------------- sections
def check_wc_monthly(cfg):
    section("1. WorldClim 2.1 historical monthly weather (CRU-TS 4.09), 1970-2024")
    p = check_folder_exists(cfg, "wc_monthly")
    if p is None:
        return
    files = files_in(p, {".tif"})
    pat = re.compile(r"(tmin|tmax|prec).*?(\d{4})-(\d{2})")
    counts = defaultdict(lambda: defaultdict(int))
    for f in files:
        m = pat.search(f.name)
        if m:
            counts[m.group(1)][int(m.group(2))] += 1
    print(f"  tif files found: {len(files)} (expected 1980 = 3 variables x 660 months)")
    print(f"  example names: {[f.name for f in files[:2]]}")
    for v in ("tmin", "tmax", "prec"):
        incomplete = [y for y in range(1970, 2025) if counts[v].get(y, 0) != 12]
        n = sum(counts[v][y] for y in range(1970, 2025))
        if not incomplete:
            note("OK", f"{v}: 660/660 months for 1970-2024")
        else:
            note("FAIL", f"{v}: {n}/660 months; incomplete years: {incomplete}")
    if files:
        tif_meta(files[0], "WorldClim monthly")


def check_wc21_normals(cfg):
    section("2. WorldClim 2.1 normals 1970-2000 (tmin, tmax, prec) - reference for delta method")
    p = check_folder_exists(cfg, "wc21_normals")
    if p is None:
        return
    files = files_in(p, {".tif"})
    for v in ("tmin", "tmax", "prec"):
        hits = [f for f in files if f"_{v}_" in f.name.lower()]
        months = sorted({int(m.group(1)) for f in hits
                         for m in [re.search(r"_(\d{2})\.tif$", f.name)] if m})
        if months == list(range(1, 13)):
            note("OK", f"{v}: 12 monthly normals")
        else:
            note("FAIL", f"{v}: months found {months} (expected 1-12)")
    others = [f.name for f in files if not any(f"_{v}_" in f.name.lower() for v in ("tmin", "tmax", "prec"))]
    if others:
        print(f"  other tif files in this folder (not used): {others[:10]}")
    if files:
        tif_meta(files[0], "WorldClim normals")


def check_wc21_bio(cfg):
    section("3. WorldClim 2.1 bioclim normals 1970-2000 - consistency check only")
    p = check_folder_exists(cfg, "wc21_bio")
    if p is None:
        return
    files = files_in(p, {".tif"})
    nums = sorted({int(m.group(1)) for f in files for m in [re.search(r"bio_?(\d{1,2})\.tif$", f.name)] if m})
    if nums == list(range(1, 20)):
        note("OK", "bio1-bio19 present")
    else:
        note("WARN", f"bio numbers found: {nums} (expected 1-19)")
    if files:
        tif_meta(files[0], "WorldClim bio normals")


def check_cmip6(cfg):
    fut = cfg["future"]
    periods = [fut["period"]] + ([fut["interpolation_period"]] if fut.get("interpolation_period") else [])
    section(f"4. WorldClim CMIP6 ({', '.join(periods)}; 5 GCMs x 2 SSPs x tmin/tmax/prec)")
    p = check_folder_exists(cfg, "cmip6")
    if p is None:
        return
    files = files_in(p, {".tif"})
    pat = re.compile(r"(tmin|tmax|prec|bioc)_(.+?)_(ssp\d{3})_(\d{4}-\d{4})")
    index = {}
    for f in files:
        m = pat.search(f.name)
        if m:
            index[(m.group(1), m.group(2), m.group(3), m.group(4))] = f
    print(f"  tif files found: {len(files)} (required: {30 * len(periods)})")
    import rasterio
    for per in periods:
        missing = [f"{v}_{g}_{s}" for g in fut["gcms"] for s in fut["ssps"] for v in ("tmin", "tmax", "prec")
                   if (v, g, s, per) not in index]
        note("FAIL" if missing else "OK",
             f"CMIP6 {per}: {'missing ' + str(missing) if missing else 'all 30 required files present'}")
        bad, unreadable = [], []
        bb = cfg["grid"]["bbox"]
        from rasterio.windows import from_bounds
        for key, f in index.items():
            if key[3] == per and key[0] in ("tmin", "tmax", "prec"):
                try:
                    with rasterio.open(f) as r:
                        if r.count != 12:
                            bad.append(f.name)
                        # full read of the Iran window of all bands: detects damaged downloads
                        win = from_bounds(bb["west"], bb["south"], bb["east"], bb["north"], r.transform)
                        r.read(window=win.round_offsets().round_lengths())
                except Exception as e:
                    unreadable.append(f"{f.name} ({type(e).__name__})")
        note("FAIL" if bad else "OK", f"CMIP6 {per}: files without 12 bands: {bad or 'none'}")
        note("FAIL" if unreadable else "OK",
             f"CMIP6 {per}: unreadable (damaged) files: {unreadable or 'none'}")
    if files:
        tif_meta(files[0], "CMIP6")


def check_terraclimate(cfg):
    import xarray as xr
    section("5. TerraClimate v1.1 (tmax, tmin, ppt)")
    p = check_folder_exists(cfg, "terraclimate")
    if p is None:
        return
    files = files_in(p, {".nc", ".nc4"})
    print(f"  netCDF files found: {len(files)}")
    print(f"  example names: {[f.name for f in files[:3]]}")
    coverage = defaultdict(set)
    attrs_done = False
    vars_done = set()
    for f in files:
        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            note("FAIL", f"cannot open {f.name}: {e}")
            continue
        dvars = [v for v in ("tmax", "tmin", "ppt") if v in ds.data_vars]
        if "time" in ds.coords:
            yrs = set(ds["time"].dt.year.values.tolist())
            for v in dvars:
                coverage[v] |= yrs
        lat = "lat" if "lat" in ds.coords else "latitude"
        lon = "lon" if "lon" in ds.coords else "longitude"
        if not attrs_done and dvars:
            attrs_done = True
            print(f"\n  global attributes (from {f.name}):")
            for k in ("method", "version", "source", "date_created"):
                print(f"    {k}: {str(ds.attrs.get(k))[:160]}")
            attrs_text = " ".join(str(v) for v in ds.attrs.values()).lower()
            if "era5" in attrs_text and "1.1" in attrs_text:
                note("OK", "attributes indicate TerraClimate v1.1 (ERA5-based)")
            else:
                note("WARN", "version not identified from attributes - check manually (must be v1.1)")
            lats, lons = ds[lat].values, ds[lon].values
            dlat, dlon = abs(lats[1] - lats[0]), abs(lons[1] - lons[0])
            print(f"    grid: {len(lats)} x {len(lons)}, res {dlat:.10f} x {dlon:.10f}")
            OFFSETS["TerraClimate"] = (lattice_offset(lons.min() - dlon / 2),
                                       lattice_offset(lats.max() + dlat / 2))
        for v in dvars:
            if v in vars_done:
                continue
            vars_done.add(v)
            lats = ds[lat].values
            lat_sl = slice(40, 25) if lats[0] > lats[-1] else slice(25, 40)
            print(f"\n  [{v}] value check on {f.name}")
            print(f"    units: {ds[v].attrs.get('units')} | scale_factor: "
                  f"{ds[v].encoding.get('scale_factor')} | add_offset: {ds[v].encoding.get('add_offset')}")
            sub = ds[v].sel({lat: lat_sl, lon: slice(44, 64)})
            for month in (0, 6):            # January and July
                vals = sub.isel(time=month).values
                print(f"    month {month + 1:02d} over Iran: min {np.nanmin(vals):.2f}, "
                      f"mean {np.nanmean(vals):.2f}, max {np.nanmax(vals):.2f}")
            allv = sub.values
            vmin, vmax = np.nanmin(allv), np.nanmax(allv)
            if v in ("tmax", "tmin"):
                ok_range = -40 < vmin and vmax < 60
                note("OK" if ok_range else "FAIL",
                     f"{v}: values in plausible degC range ({vmin:.1f} to {vmax:.1f})")
            else:
                note("OK" if vmin >= 0 else "FAIL", f"ppt: no negative values (min {vmin:.1f} mm)")
        ds.close()
    print("\n  year coverage:")
    for v in ("tmax", "tmin", "ppt"):
        yrs = sorted(coverage[v])
        if not yrs:
            note("FAIL", f"{v}: not found")
            continue
        gaps = [y for y in range(yrs[0], yrs[-1] + 1) if y not in coverage[v]]
        print(f"    {v}: {yrs[0]}-{yrs[-1]} ({len(yrs)} years), gaps: {gaps or 'none'}")
    hist = all(set(range(1970, 2000)) <= coverage[v] for v in ("tmax", "tmin", "ppt"))
    cont = all(set(range(2000, 2025)) <= coverage[v] for v in ("tmax", "tmin", "ppt"))
    note("OK" if cont else "FAIL", f"TerraClimate 2000-2024 complete: {cont}")
    note("OK" if hist else "WARN", f"TerraClimate 1970-1999 complete: {hist}")
    print(f"\n  => study design: {'FULL 2x2 (4 baselines)' if hist and cont else '3 baselines (TC_1970_1999 unavailable)'}")


def check_static(cfg):
    section("6. Static data: terrain (30 m), soil (250 m), vectors, occurrences")
    p = check_folder_exists(cfg, "static")
    if p is None:
        return
    tifs = files_in(p, {".tif", ".tiff"})

    print("\n  6a. Terrain layers (expected 30 m, from ArcGIS Pro)")
    ranges = {"elevation": (-200, 6000), "slope": (0, 90), "northness": (-1, 1), "eastness": (-1, 1)}
    keys = {"elevation": ["elev", "dem"], "slope": ["slope"],
            "northness": ["north"], "eastness": ["east"]}
    for name, kws in keys.items():
        hits = find_one(tifs, kws)
        if len(hits) != 1:
            note("FAIL", f"{name}: expected 1 file, found {len(hits)} {[h.name for h in hits]}")
            continue
        print(f"\n  [{name}]")
        tif_meta(hits[0], record_offset=False)
        st = sampled_stats(hits[0])
        if st:
            lo, hi = ranges[name]
            print(f"    sampled values: min {st[0]:.3f}, mean {st[1]:.3f}, max {st[2]:.3f}")
            if st[0] < lo - 1e-6 or st[2] > hi + 1e-6:
                note("FAIL", f"{name}: values outside expected range [{lo}, {hi}]")
            else:
                note("OK", f"{name}: values within expected range")

    print("\n  6b. Soil layers (SoilGrids 0-5 cm, 250 m)")
    for name in cfg["predictors"]["candidates"]["soil"]:
        hits = [f for f in tifs if f.name.lower().startswith(name)]
        if len(hits) != 1:
            note("FAIL", f"soil {name}: expected 1 file, found {len(hits)} {[h.name for h in hits]}")
            continue
        print(f"\n  [{name}]")
        tif_meta(hits[0], record_offset=False)
        st = sampled_stats(hits[0])
        if st:
            print(f"    sampled values (SoilGrids mapped units): min {st[0]:.2f}, mean {st[1]:.2f}, max {st[2]:.2f}")
            note("OK", f"soil {name}: readable")
        import rasterio
        with rasterio.open(hits[0]) as r:
            f = max(1, int(np.ceil(max(r.height, r.width) / 2000)))
            a = r.read(1, out_shape=(r.height // f, r.width // f))
        zero_share = float((a == 0).mean()) * 100
        print(f"    share of pixels equal to 0: {zero_share:.1f}% (nodata defined: {r.nodata})")
        if r.nodata is None and zero_share > 0:
            note("WARN", f"soil {name}: no nodata value defined; 0 is probably 'no data' (sea/outside) - handled in step 07 (static layers)")

    print("\n  6c. Vector layers")
    try:
        import geopandas as gpd
        for f in files_in(p, {".shp", ".gpkg"}):
            g = gpd.read_file(f)
            cols = list(g.columns)
            kind = ("ecoregions" if "ECO_NAME" in cols else
                    "naturalearth countries" if any(c in cols for c in ("ADMIN", "SOVEREIGNT")) else
                    "boundary (assumed)")
            b = g.total_bounds
            print(f"\n  [{kind}] {f.name}")
            print(f"    crs: {g.crs} | features: {len(g)} | bounds W {b[0]:.3f} S {b[1]:.3f} E {b[2]:.3f} N {b[3]:.3f}")
            if g.crs is None:
                note("FAIL", f"{f.name}: no CRS defined")
            elif not g.crs.is_geographic:
                note("WARN", f"{f.name}: projected CRS (will be reprojected to EPSG:4326)")
            else:
                note("OK", f"{f.name}: geographic CRS")
    except Exception as e:
        note("FAIL", f"vector check failed: {e}")

    print("\n  6d. Occurrence records")
    try:
        import pandas as pd
        tables = files_in(p, {".csv", ".xlsx"})
        if not tables:
            note("FAIL", "no occurrence CSV/XLSX found in static folder")
        for f in tables:
            df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
            print(f"\n  [{f.name}] {len(df)} rows | columns: {list(df.columns)}")
            print("   " + df.head(3).to_string().replace("\n", "\n   "))
            low = {c.lower().strip(): c for c in df.columns}
            lon = next((low[c] for c in ("lon", "long", "longitude", "x", "decimallongitude") if c in low), None)
            lat = next((low[c] for c in ("lat", "latitude", "y", "decimallatitude") if c in low), None)
            if lon is None or lat is None:
                note("FAIL", f"{f.name}: longitude/latitude columns not recognised")
                continue
            xy = df[[lon, lat]].apply(pd.to_numeric, errors="coerce")
            bad = xy.isna().any(axis=1)
            n_nan = int(bad.sum())
            note("OK" if n_nan == 0 else "FAIL", f"rows with missing/non-numeric coordinates: {n_nan}")
            if n_nan:
                print("    problematic rows (row number = line in file minus 1):")
                print("   " + df[bad].to_string().replace("\n", "\n   "))
            xy = xy[~bad]
            print(f"    valid records used below: {len(xy)}")
            bb = cfg["grid"]["bbox"]
            inside = xy[lon].between(bb["west"], bb["east"]) & xy[lat].between(bb["south"], bb["north"])
            note("OK" if inside.all() else "FAIL", f"records inside processing bbox: {int(inside.sum())}/{len(xy)}")
            n_dup = int(xy.duplicated().sum())
            note("OK" if n_dup == 0 else "WARN", f"exact duplicate coordinates: {n_dup}")
            cells = (np.floor((xy[lon] + 180) / CELL).astype("Int64").astype(str) + "_" +
                     np.floor((90 - xy[lat]) / CELL).astype("Int64").astype(str))
            n_cell = int(cells.nunique())
            print(f"    records: {len(xy)} | unique 1/24-degree cells: {n_cell}")
            note("OK", f"{len(xy)} records fall in {n_cell} distinct grid cells")
    except Exception as e:
        note("FAIL", f"occurrence check failed: {e}")


def check_alignment():
    section("7. Grid alignment of climate products (offset from the global 1/24-degree lattice)")
    tol = 1e-6
    for name, (ox, oy) in OFFSETS.items():
        if ox < tol and oy < tol:
            note("OK", f"{name}: aligned (x {ox:.1e}, y {oy:.1e})")
        else:
            note("WARN", f"{name}: offset x {ox:.2e}, y {oy:.2e} deg - alignment needed")


def main():
    cfg = load_config()
    out = qa_dir(cfg, STEP) / "01_report.txt"
    write_run_info(out.parent / "01_run_info.json", CONFIG_FILE)
    with Report(out):
        print(f"Step {STEP}")
        print(f"raw-data root: {cfg['paths']['raw']['root']}")
        for fn in (check_wc_monthly, check_wc21_normals, check_wc21_bio,
                   check_cmip6, check_terraclimate, check_static):
            try:
                fn(cfg)
            except Exception as e:
                note("FAIL", f"{fn.__name__} crashed: {type(e).__name__}: {e}")
        check_alignment()
        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails:
            print(f"   - FAIL: {m}")
        for m in warns:
            print(f"   - WARN: {m}")
        if not fails:
            print("\n  No blocking problems. Review warnings, then proceed to step 02.")


if __name__ == "__main__":
    main()
