"""
Step 06 - Independent precipitation check with CHIRPS v3
========================================================

Purpose
    Assess the two climate products against an independent gauge-satellite precipitation product,
    CHIRPS v3 (Funk et al. 2026), over 1981-2024, with emphasis on the species' range. Questions:
      Q1  Are the implausible summer values of D18 corroborated by an independent product?
      Q2  Which product reproduces month-to-month and year-to-year precipitation variability better?
      Q3  Which product's change between 1981-1999 and 2000-2024 agrees better with CHIRPS?
      Q4  Which product's wettest quarter agrees with CHIRPS (relevant for bio8/bio16-bio19, D28)?

Why / framing
    Neither product is ground truth and CHIRPS is not either. CHIRPS uses a different method
    (thermal-infrared cold-cloud duration + station data, own climatology CHPclim2) and is independent
    of the temporal anomaly sources of WorldClim (CRU-TS) and TerraClimate (ERA5), although some
    stations may be shared. Results are reported as consistency with an independent product.
    CHIRPS v3 station data are corrected for gauge undercatch, so absolute totals are expected to be
    higher; comparisons therefore rely on anomalies, correlations and RELATIVE changes, which are not
    affected by a constant multiplicative bias. CHIRPS is used only for this check, never for modelling.

Regridding
    CHIRPS (0.05 deg) is the only dataset that does not share the 1/24-degree lattice. It is regridded
    to the reference grid with GDAL 'average' resampling (area-weighted for partial overlaps). This does
    not affect decision D04 (no resampling of modelling data).

Inputs   raw CHIRPS v3 monthly GeoTIFFs 1981-01 ... 2024-12 (config paths.raw.chirps)
         data_processed/02_crop_and_units/{wc,tc}_monthly_prec_1970_2024.nc
Outputs  data_processed/06_independent_precip_check/chirps_v3_monthly_prec_1981_2024.nc
         data_processed/06_independent_precip_check/maps/*.tif   (correlation and change maps)
QA       results/qa/06_independent_precip_check/  06_report.txt, 06_run_info.json, 06_*.csv, 06_fig_*.png
Manuscript results/manuscript/tables/T06_*.csv, results/manuscript/notes/06_independent_precip_check.md
Run      python scripts/06_independent_precip_check.py
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, raw_dir, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import CELL, ref_grid, to_dataarray, nc_encoding, write_geotiff  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402
from hemiscorpius.precip import flag_extremes, clean_precip, wettest_quarter_start  # noqa: E402

STEP = "06_independent_precip_check"
START, END = 1981, 2024
N = (END - START + 1) * 12
P1, P2 = (1981, 1999), (2000, 2024)
DEFAULT_RULE = {"min_mm": 200.0, "factor": 10.0, "offset_mm": 5.0}
CHIRPS_EXTREME = {"min_mm": 100.0, "factor": 5.0, "offset_mm": 5.0}   # relaxed rule for corroboration
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


# --------------------------------------------------------------------------- CHIRPS reading
def find_chirps_files(folder):
    pat = re.compile(r"(\d{4})\.(\d{2})\.tif(\.gz)?$")
    files = {}
    for f in Path(folder).rglob("*"):
        m = pat.search(f.name)
        if m and START <= int(m.group(1)) <= END:
            files[(int(m.group(1)), int(m.group(2)))] = f
    return files


def read_chirps_month(path, grid):
    """Read one CHIRPS GeoTIFF window around the grid and regrid to it (GDAL average)."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.windows import from_bounds
    src_path = f"/vsigzip/{path}" if str(path).endswith(".gz") else str(path)
    margin = 0.2
    west, north = grid.west, grid.north
    east, south = west + grid.ncols * CELL, north - grid.nrows * CELL
    with rasterio.open(src_path) as src:
        win = from_bounds(west - margin, south - margin, east + margin, north + margin, src.transform)
        win = win.round_offsets().round_lengths()
        a = src.read(1, window=win).astype("float32")
        nod = src.nodata
        transform = src.window_transform(win)
        crs = src.crs or "EPSG:4326"
    if nod is not None:
        a[a == nod] = np.nan
    a[(a < 0) | ~np.isfinite(a)] = np.nan
    dst = np.full(grid.shape, np.nan, dtype="float32")
    reproject(a, dst, src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=grid.transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=Resampling.average)
    return dst


def load_product(proc, prefix):
    import xarray as xr
    with xr.open_dataset(proc / "02_crop_and_units" / f"{prefix}_monthly_prec_1970_2024.nc") as ds:
        arr = ds["prec"].values.astype("float32")
    i0 = (START - 1970) * 12
    return arr[i0:i0 + N]


# --------------------------------------------------------------------------- metrics
def anomalies(x):
    out = np.empty_like(x)
    for m in range(12):
        out[m::12] = x[m::12] - np.nanmean(x[m::12], axis=0)
    return out


def corr_map(a, b):
    """Per-cell Pearson correlation of two (n, rows, cols) series (NaN-aware)."""
    ok_ = np.isfinite(a) & np.isfinite(b)
    a = np.where(ok_, a, np.nan)
    b = np.where(ok_, b, np.nan)
    a = a - np.nanmean(a, axis=0)
    b = b - np.nanmean(b, axis=0)
    num = np.nansum(a * b, axis=0)
    den = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[ok_.sum(axis=0) < 24] = np.nan
    return r


def annual_totals(x):
    return x.reshape(-1, 12, *x.shape[1:]).sum(axis=1)


def period_mean(x, period):
    i0, i1 = (period[0] - START) * 12, (period[1] - START + 1) * 12
    return np.nanmean(annual_totals(x[i0:i1]), axis=0)


def climatology(x, period):
    i0, i1 = (period[0] - START) * 12, (period[1] - START + 1) * 12
    sub = x[i0:i1]
    return np.stack([np.nanmean(sub[m::12], axis=0) for m in range(12)])


def summarise(maps, mask, occ_mask, label):
    rows = []
    for name, m in maps.items():
        rows.append(dict(metric=label, dataset=name,
                         iran_median=float(np.nanmedian(m[mask])),
                         occurrence_median=float(np.nanmedian(m[occ_mask])),
                         occurrence_mean=float(np.nanmean(m[occ_mask]))))
    return rows


# --------------------------------------------------------------------------- main analyses
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    (out / "maps").mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_run_info(qa / "06_run_info.json", CONFIG_FILE)
    rule = {**DEFAULT_RULE, **cfg.get("precip_cleaning", {})}

    with Report(qa / "06_report.txt"):
        print(f"Step {STEP}  (CHIRPS v3 vs WorldClim monthly vs TerraClimate, {START}-{END})")
        import rasterio
        with rasterio.open(proc / "02_crop_and_units" / "reference_grid.tif") as r_:
            mask = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)
        r, c = occ["row"].values, occ["col"].values
        occ_mask = np.zeros(mask.shape, dtype=bool)          # occurrence cells and 8 neighbours
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                occ_mask[np.clip(r + dr, 0, grid.nrows - 1), np.clip(c + dc, 0, grid.ncols - 1)] = True
        occ_mask &= mask
        print(f"Iran cells {int(mask.sum())}; occurrence neighbourhood cells {int(occ_mask.sum())}")

        # ---------------------------------------------------------------- 1. CHIRPS
        section("1. CHIRPS v3: completeness, regridding, plausibility")
        cache = out / f"chirps_v3_monthly_prec_{START}_{END}.nc"
        try:
            folder = raw_dir(cfg, "chirps")
        except KeyError:
            note("FAIL", "paths.raw.chirps missing in config.yaml")
            return
        files = find_chirps_files(folder)
        expected = [(y, m) for y in range(START, END + 1) for m in range(1, 13)]
        missing = [f"{y}-{m:02d}" for y, m in expected if (y, m) not in files]
        note("OK" if not missing else "FAIL", f"CHIRPS files found {len(files)}/{N}; missing: {missing[:12] or 'none'}")
        if missing:
            return
        chirps = np.full((N,) + grid.shape, np.nan, dtype="float32")
        bad = []
        for i, key in enumerate(expected):
            try:
                chirps[i] = read_chirps_month(files[key], grid)
            except Exception as e:
                bad.append(f"{files[key].name}: {type(e).__name__}")
            if key[1] == 12 and key[0] % 10 == 0:
                print(f"    read up to {key[0]}")
        note("OK" if not bad else "FAIL", f"unreadable CHIRPS files: {bad or 'none'}")
        if bad:
            return
        v = chirps[np.broadcast_to(mask, chirps.shape)]
        v = v[np.isfinite(v)]
        note("OK" if v.min() >= 0 and v.max() < 2000 else "WARN",
             f"CHIRPS inside Iran: min {v.min():.1f}, mean {v.mean():.1f}, max {v.max():.1f} mm/month")
        coverage = np.isfinite(chirps).all(axis=0) & mask
        print(f"  Iran cells with complete CHIRPS series: {int(coverage.sum())} of {int(mask.sum())}")
        ds = xr.Dataset({"prec": to_dataarray(chirps, grid, "prec", "time",
                                              pd.date_range(f"{START}-01-01", periods=N, freq="MS"), "mm/month")})
        ds.attrs.update(source="CHIRPS v3 monthly (Funk et al. 2026; doi:10.15780/G2JQ0P)",
                        processing="regridded to the 1/24-degree reference grid with GDAL average resampling")
        ds.to_netcdf(cache, encoding=nc_encoding(ds))
        valid = coverage                                         # common evaluation domain
        occ_eval = occ_mask & valid

        # ---------------------------------------------------------------- products
        data = {"CHIRPS": chirps,
                "WC original": load_product(proc, "wc"), "TC original": load_product(proc, "tc")}
        data["WC cleaned"], _ = clean_precip(data["WC original"].copy(), rule)
        data["TC cleaned"], _ = clean_precip(data["TC original"].copy(), rule)
        products = [k for k in data if k != "CHIRPS"]
        for k in data:
            data[k] = np.where(valid[None], data[k], np.nan).astype("float32")

        rows = []
        # ---------------------------------------------------------------- 2. variability (Q2)
        # Q2 - temporal agreement with CHIRPS (monthly anomalies and annual totals)
        anom_c, ann_c = anomalies(data["CHIRPS"]), annual_totals(data["CHIRPS"])
        rmaps, amaps = {}, {}
        for k in products:
            rmaps[k] = corr_map(anomalies(data[k]), anom_c)
            amaps[k] = corr_map(annual_totals(data[k]), ann_c)
        rows += summarise(rmaps, valid, occ_eval, "r monthly anomalies vs CHIRPS")
        rows += summarise(amaps, valid, occ_eval, "r annual totals vs CHIRPS")
        for k in ("WC original", "TC original"):
            write_geotiff(out / "maps" / f"r_monthly_anomaly_vs_chirps_{k.split()[0]}.tif",
                          np.where(valid, rmaps[k], np.nan), grid)

        # relative bias (expected > 0 for CHIRPS v3 because of undercatch correction)
        cm = np.nanmean(annual_totals(data["CHIRPS"]), axis=0)
        bmaps = {k: np.nanmean(annual_totals(data[k]), axis=0) / cm - 1 for k in products}
        rows += summarise(bmaps, valid, occ_eval, "relative bias of mean annual total vs CHIRPS (fraction)")

        # ---------------------------------------------------------------- 3. period change (Q3)
        # Q3 - relative change 2000-2024 vs 1981-1999
        chg = {k: period_mean(data[k], P2) / period_mean(data[k], P1) - 1 for k in data}
        rows += summarise(chg, valid, occ_eval, "relative change 2000-2024 vs 1981-1999 (fraction)")
        dchg = {k: np.abs(chg[k] - chg["CHIRPS"]) for k in products}
        rows += summarise(dchg, valid, occ_eval, "|relative change - CHIRPS relative change| (fraction)")
        for k in ("CHIRPS", "WC original", "TC original"):
            write_geotiff(out / "maps" / f"relative_change_2000_2024_vs_1981_1999_{k.split()[0]}.tif",
                          np.where(valid, chg[k], np.nan), grid)

        # ---------------------------------------------------------------- 4. wettest quarter (Q4)
        # Q4 - agreement of the wettest quarter with CHIRPS (climatologies)
        for per in (P1, P2):
            wq_c = wettest_quarter_start(climatology(data["CHIRPS"], per))
            agree = {k: (wettest_quarter_start(climatology(data[k], per)) == wq_c).astype("float32")
                     for k in products}
            rows += summarise(agree, valid, occ_eval, f"wettest quarter agrees with CHIRPS {per[0]}-{per[1]} (share)")
        # summer (Jun-Sep) share of annual precipitation, 2000-2024
        def jjas_share(x):
            cl = climatology(x, P2)
            return cl[5:9].sum(axis=0) / np.maximum(cl.sum(axis=0), 1e-6)
        sh = {k: jjas_share(data[k]) for k in data}
        rows += summarise(sh, valid, occ_eval, "Jun-Sep share of annual precipitation 2000-2024")
        # the climatology shared by both products (WorldClim 2.1 normals) - is the summer share inherited?
        nprec = xr.open_dataset(proc / "02_crop_and_units" / "wc21_normals_1970_2000.nc")["prec"].values
        nshare = nprec[5:9].sum(axis=0) / np.maximum(nprec.sum(axis=0), 1e-6)
        rows += summarise({"WC2.1 normals 1970-2000": np.where(valid, nshare, np.nan)}, valid, occ_eval,
                          "Jun-Sep share of annual precipitation (WorldClim 2.1 normals)")

        section("2-4. Agreement with CHIRPS: variability (Q2), relative change (Q3), wettest quarter (Q4)")
        df = pd.DataFrame(rows)
        df.to_csv(qa / "06_agreement_with_chirps.csv", index=False)
        df.round(4).to_csv(tables / "T06_agreement_with_chirps.csv", index=False)
        print("   " + df.round(3).to_string(index=False).replace("\n", "\n   "))

        # ---------------------------------------------------------------- 5. extremes (Q1)
        section("5. Q1 - Are the flagged extreme months corroborated by CHIRPS?")
        cflags, cmed = flag_extremes(data["CHIRPS"], **CHIRPS_EXTREME)
        ext_rows, top_rows = [], []
        time = pd.date_range(f"{START}-01-01", periods=N, freq="MS")
        for k in ("WC original", "TC original"):
            pf, _ = flag_extremes(data[k], rule["min_mm"], rule["factor"], rule["offset_mm"])
            pf &= valid[None]
            n = int(pf.sum())
            corro = int((pf & cflags).sum())
            ratio = data[k][pf] / np.maximum(data["CHIRPS"][pf], 1.0)
            ext_rows.append(dict(product=k, flagged_cell_months=n, corroborated_by_chirps=corro,
                                 share_corroborated=corro / n if n else np.nan,
                                 median_ratio_product_to_chirps=float(np.median(ratio)) if n else np.nan,
                                 flagged_in_occurrence_neighbourhood=int((pf & occ_eval[None]).sum())))
            for t, rr, cc in np.argwhere(pf):
                top_rows.append(dict(product=k, date=time[t].strftime("%Y-%m"), lat=round(float(grid.lats[rr]), 3),
                                     lon=round(float(grid.lons[cc]), 3), product_mm=float(data[k][t, rr, cc]),
                                     chirps_mm=float(data["CHIRPS"][t, rr, cc]),
                                     chirps_same_month_median_mm=float(cmed[t % 12, rr, cc]),
                                     chirps_also_extreme=bool(cflags[t, rr, cc])))
        ext = pd.DataFrame(ext_rows)
        ext.to_csv(qa / "06_extremes_vs_chirps.csv", index=False)
        ext.round(4).to_csv(tables / "T06_extremes_vs_chirps.csv", index=False)
        tops = pd.DataFrame(top_rows)
        tops.to_csv(qa / "06_flagged_values_with_chirps.csv", index=False)
        print("   " + ext.round(3).to_string(index=False).replace("\n", "\n   "))
        if len(tops):
            print("\n  largest flagged values with the CHIRPS value of the same month and cell:")
            print("   " + tops.sort_values("product_mm", ascending=False).head(12).round(1)
                  .to_string(index=False).replace("\n", "\n   "))
        for _, row in ext.iterrows():
            print(f"  {row['product']}: {row.share_corroborated:.1%} of flagged values also extreme in CHIRPS; "
                  f"median product/CHIRPS ratio {row.median_ratio_product_to_chirps:.1f}")
        note("OK", "extremes compared with CHIRPS; interpretation per product in the lines above "
                   "(non-corroboration is disagreement between gridded products, not proof of an artefact; "
                   "corroboration does not establish the true event magnitude)")

        # ---------------------------------------------------------------- figures
        ext_ = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        for ax, (k, title, cmap, lim) in zip(axes, (
                ("WC original", "WorldClim vs CHIRPS: r of monthly anomalies", "viridis", (0, 1)),
                ("TC original", "TerraClimate vs CHIRPS: r of monthly anomalies", "viridis", (0, 1)),
                ("diff", "r(TC) - r(WC)", "RdBu_r", (-0.4, 0.4)))):
            m = rmaps["TC original"] - rmaps["WC original"] if k == "diff" else rmaps[k]
            im = ax.imshow(np.where(valid, m, np.nan), extent=ext_, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.scatter(occ["lon"], occ["lat"], s=5, c="red")
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75)
        fig.tight_layout()
        fig.savefig(qa / "06_fig_anomaly_correlation_with_chirps.png", dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))
        for ax, k in zip(axes, ("CHIRPS", "WC original", "TC original")):
            im = ax.imshow(np.where(valid, 100 * chg[k], np.nan), extent=ext_, cmap="BrBG", vmin=-40, vmax=40)
            ax.scatter(occ["lon"], occ["lat"], s=5, c="k")
            ax.set_title(f"{k}: change 2000-2024 vs 1981-1999 (%)", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75, label="%")
        fig.tight_layout()
        fig.savefig(qa / "06_fig_relative_change_vs_chirps.png", dpi=200)
        plt.close(fig)

        years = np.arange(START, END + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        for k in ("CHIRPS", "WC original", "TC original"):
            a = annual_totals(data[k])[:, occ_eval]
            s_ = np.nanmean(a, axis=1)
            ax.plot(years, 100 * (s_ / np.nanmean(s_) - 1), label=k)
        ax.axvline(1999.5, color="grey", ls="--", lw=0.8)
        ax.set_ylabel("annual precipitation anomaly (%)")
        ax.set_title("Occurrence neighbourhood: annual precipitation relative to each dataset's own mean")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(qa / "06_fig_occurrence_region_annual_series.png", dpi=200)
        plt.close(fig)

        # ---------------------------------------------------------------- notes
        try:
            n = ManuscriptNotes(cfg, STEP, "Independent precipitation check with CHIRPS v3")
            n.number("chirps_iran_cells_complete", int(valid.sum()), "cells", "Iran cells with complete CHIRPS series", "QA")
            for _, row in df.iterrows():
                key = (row.metric.split(" (")[0].replace(" ", "_").replace("|", "").replace("-", "_")
                       .replace("vs_", "").lower())
                ds_ = row.dataset.replace(" ", "_")
                if row.metric.startswith("wettest quarter"):
                    # codex_20261112: the mean of binary agreements is the share, not the median.
                    n.number(f"{key}_{ds_}_occ_share", round(row.occurrence_mean, 4), "fraction",
                             f"{row.metric} - {row.dataset}, occurrence neighbourhood agreement share",
                             "gridded-product consistency check")
                    continue
                n.number(f"{key}_{ds_}_occ_median", round(row.occurrence_median, 4), "",
                         f"{row.metric} - {row.dataset}, occurrence neighbourhood median", "independent check")
                n.number(f"{key}_{ds_}_iran_median", round(row.iran_median, 4), "",
                         f"{row.metric} - {row.dataset}, Iran median", "independent check")
            for _, row in ext.iterrows():
                p_ = row["product"].split()[0]
                n.number(f"extremes_{p_}_flagged", int(row.flagged_cell_months), "cell-months",
                         f"{row['product']}: flagged values 1981-2024", "D18")
                n.number(f"extremes_{p_}_share_corroborated", round(float(row.share_corroborated), 4), "fraction",
                         f"{row['product']}: share of flagged values also extreme in CHIRPS", "D18")
            n.text("Methods (draft)", f"""
To assess consistency with an additional gridded product, monthly precipitation was compared with CHIRPS v3
(Funk et al. 2026), a gauge-satellite product based on thermal-infrared cold-cloud duration and station data,
over {START}-{END}. CHIRPS was regridded to the 1/24-degree grid by area-weighted averaging; it was used only
for this comparison. Because CHIRPS v3 includes a gauge-undercatch correction, comparisons used monthly
anomaly correlations, annual-total correlations, relative changes between 1981-1999 and 2000-2024, agreement
of the wettest quarter, and corroboration of flagged extreme months (CHIRPS value >= 100 mm and >= 5 x
(its same-month median + 5 mm)), rather than absolute totals. This is a product-consistency check, not
validation against independent local ground observations; station inputs may overlap between products.""")
            n.text("Results (draft placeholder)", """
Report: which product agrees better with CHIRPS in the occurrence neighbourhood (anomaly r, annual r,
relative change, wettest quarter) and the share of flagged extreme values corroborated by CHIRPS.""")
            n.save()
        except Exception as e:
            note("WARN", f"manuscript notes not written: {type(e).__name__}: {e}")

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails:
            print(f"   - FAIL: {m}")
        for m in warns:
            print(f"   - WARN: {m}")
        if not fails:
            print("\n  Step 06 completed. Review figures and tables in results/qa/06_independent_precip_check/.")


if __name__ == "__main__":
    main()
