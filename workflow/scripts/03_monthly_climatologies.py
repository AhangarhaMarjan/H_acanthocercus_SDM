"""
Step 03 - Monthly climatologies of the four climatic baselines
==============================================================

Purpose
    Build the 12-month climatology (mean of each calendar month over the period) of tmin,
    tmax and precipitation for the four baselines of the 2x2 design:
        WC_1970_1999, WC_2000_2024 (WorldClim 2.1 monthly weather)
        TC_1970_1999, TC_2000_2024 (TerraClimate v1.1)
    Each baseline is produced in two versions (decision D19):
        original - products as published (MAIN analysis)
        cleaned  - implausible monthly precipitation replaced (SENSITIVITY analysis)

Why
    - Bioclimatic variables are computed from 12-month climatologies (decision D05), with the
      same rule for every baseline.
    - Both products contain implausible summer precipitation in the hyper-arid southern belt
      (step 02). The cleaned version quantifies how much this affects the results.

Cleaning rule (identical for both products; config 'precip_cleaning')
    A monthly value is flagged if value >= min_mm AND value >= factor x (median of the same
    calendar month at that cell over 1970-2024 + offset_mm). Flagged values are replaced by
    that same-month median. Temperatures are not modified.

Checks
    1. Consistency: WC_1970_1999 (built here) vs official WorldClim 2.1 normals 1970-2000.
    4. Robustness: 25-year (2000-2024) vs 30-year (1995-2024) recent baseline.
    5. Robustness: number of values flagged by alternative cleaning thresholds.
    2. Impact of cleaning at the occurrence cells, incl. changes of the wettest quarter
       (which drives bio8).
    3. Maps and summary of the period effect, product effect and their interaction.

Inputs   data_processed/02_crop_and_units/*.nc, occurrence records (static folder)
Outputs  data_processed/03_monthly_climatologies/
             {baseline}_{original|cleaned}.nc            12-month tmin, tmax, prec
             annual/{baseline}_{version}_annual_{tmean|prec}.tif
             effects/{tmean|prec}_{effect}_{version}.tif  period_WC, period_TC, interaction,
                                                          product_1970_1999, product_2000_2024, period_mean
             qa_layers/prec_flagged_months_{WC|TC}.tif, wettest_quarter_changed_{baseline}.tif
QA       results/qa/03_monthly_climatologies/  03_report.txt, 03_run_info.json, 03_*.csv, 03_fig_*.png
Manuscript results/manuscript/tables/T03_*.csv, results/manuscript/notes/03_monthly_climatologies.md
Run      python scripts/03_monthly_climatologies.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import CELL, ref_grid, to_dataarray, nc_encoding, write_geotiff  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402
from hemiscorpius.precip import clean_precip, wettest_quarter_start as wettest_quarter  # noqa: E402

STEP = "03_monthly_climatologies"
VARS = ("tmin", "tmax", "prec")
PRODUCTS = {"WC": ("wc_monthly", "WorldClim 2.1 monthly"), "TC": ("tc_monthly", "TerraClimate v1.1")}
VERSIONS = ("original", "cleaned")
ISSUES = []
DEFAULT_CLEANING = {"min_mm": 200.0, "factor": 10.0, "offset_mm": 5.0}


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


# --------------------------------------------------------------------------- core
def load_monthly(src, prefix, var):
    import xarray as xr
    with xr.open_dataset(src / f"{prefix}_{var}_1970_2024.nc") as ds:
        arr = ds[var].values.astype("float32")
        time = pd.to_datetime(ds["time"].values)
    if len(time) != 660:
        raise ValueError(f"{prefix}_{var}: {len(time)} months (expected 660)")
    return arr, time


def climatology(arr, time, start, end):
    """Mean of each calendar month over [start, end]; NaN unless every year is valid."""
    sel = (time.year >= start) & (time.year <= end)
    sub, months = arr[sel], time[sel].month
    nyears = end - start + 1
    out = np.full((12,) + arr.shape[1:], np.nan, dtype="float32")
    for m in range(1, 13):
        x = sub[months == m]
        valid = np.isfinite(x).sum(axis=0)
        mean = np.nanmean(x, axis=0)
        out[m - 1] = np.where(valid == nyears, mean, np.nan)
    return out


def annual(clim):
    """Annual mean temperature and annual precipitation from a climatology dict."""
    tmean = ((clim["tmin"] + clim["tmax"]) / 2).mean(axis=0)
    psum = clim["prec"].sum(axis=0)
    return tmean, psum


def save_baseline(out, name, version, clim, grid, meta):
    import xarray as xr
    das = {v: to_dataarray(clim[v], grid, v, "month", np.arange(1, 13),
                           "degC" if v != "prec" else "mm/month") for v in VARS}
    ds = xr.Dataset(das)
    ds.attrs.update(meta)
    ds.to_netcdf(out / f"{name}_{version}.nc", encoding=nc_encoding(ds))


# --------------------------------------------------------------------------- QA
def consistency_with_normals(src, clims, mask, qa):
    import xarray as xr
    section("4. Consistency check: WC_1970_1999 (built here) vs WorldClim 2.1 normals 1970-2000")
    with xr.open_dataset(src / "wc21_normals_1970_2000.nc") as ds:
        normals = {v: ds[v].values.astype("float32") for v in VARS}
    rows = []
    built = clims[("WC_1970_1999", "original")]
    for v in VARS:
        for m in range(12):
            a, b = built[v][m], normals[v][m]
            ok_ = mask & np.isfinite(a) & np.isfinite(b)
            d = a[ok_] - b[ok_]
            rows.append(dict(variable=v, month=m + 1, mean_diff=float(d.mean()),
                             mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                             r=float(np.corrcoef(a[ok_], b[ok_])[0, 1]),
                             normals_mean=float(b[ok_].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "03_consistency_wc_built_vs_wc21_normals.csv", index=False)
    for v in VARS:
        s = df[df.variable == v]
        unit = "mm" if v == "prec" else "degC"
        print(f"  {v}: mean difference {s.mean_diff.mean():+.2f} {unit}, MAE {s.mae.mean():.2f} {unit}, "
              f"r {s.r.min():.3f}-{s.r.max():.3f} (over 12 months)")
        if v == "prec":
            rel = 100 * s.mae.mean() / max(s.normals_mean.mean(), 1e-6)
            note("OK" if rel < 20 else "WARN", f"prec: mean absolute difference = {rel:.1f}% of normals")
        else:
            note("OK" if s.mae.mean() < 0.75 else "WARN",
                 f"{v}: mean absolute difference {s.mae.mean():.2f} degC (expected small; "
                 f"periods differ by one year and in construction)")
    return df


def cleaning_impact(clims, flags_count, occ, mask, grid, qa, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section("5. Impact of the precipitation cleaning (sensitivity version)")
    rows, wq_maps = [], {}
    r, c = occ["row"].values, occ["col"].values
    for name in ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024"):
        po = clims[(name, "original")]["prec"]
        pc = clims[(name, "cleaned")]["prec"]
        ann_o, ann_c = po.sum(axis=0), pc.sum(axis=0)
        wq_o, wq_c = wettest_quarter(po), wettest_quarter(pc)
        changed = mask & np.isfinite(wq_o) & (wq_o != wq_c)
        wq_maps[name] = changed
        (out / "qa_layers").mkdir(exist_ok=True)
        write_geotiff(out / "qa_layers" / f"wettest_quarter_changed_{name}.tif",
                      np.where(mask, changed.astype("float32"), np.nan), grid)
        d_ann = (ann_o - ann_c)[r, c]
        d_month = np.nanmax(np.abs(po - pc)[:, r, c], axis=0)
        rec = dict(baseline=name,
                   iran_cells_with_changed_climatology=int((mask & (np.abs(ann_o - ann_c) > 0.01)).sum()),
                   iran_cells_wettest_quarter_changed=int(changed.sum()),
                   occ_cells=len(occ),
                   occ_cells_changed=int((d_ann > 0.01).sum()),
                   occ_max_annual_diff_mm=round(float(np.nanmax(d_ann)), 1),
                   occ_mean_annual_diff_mm=round(float(np.nanmean(d_ann)), 2),
                   occ_max_monthly_diff_mm=round(float(np.nanmax(d_month)), 1),
                   occ_wettest_quarter_changed=int(changed[r, c].sum()))
        rows.append(rec)
        print(f"  {name}: occurrence cells changed {rec['occ_cells_changed']}/{len(occ)}; "
              f"max annual difference {rec['occ_max_annual_diff_mm']} mm; "
              f"wettest quarter changed at {rec['occ_wettest_quarter_changed']} occurrence cells "
              f"and {rec['iran_cells_wettest_quarter_changed']} cells in Iran")
    df = pd.DataFrame(rows)
    df.to_csv(qa / "03_precip_cleaning_impact.csv", index=False)
    total_wq = int(df.occ_wettest_quarter_changed.sum())
    note("OK" if total_wq == 0 else "WARN",
         f"wettest quarter changed by cleaning at occurrence cells (sum over baselines): {total_wq} "
         f"- bio8/bio9/bio16-19 affected there; sensitivity analysis required")

    for prod in ("WC", "TC"):
        write_geotiff(out / "qa_layers" / f"prec_flagged_months_{prod}.tif",
                      np.where(mask, flags_count[prod].astype("float32"), np.nan), grid)
    ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.8))
    for k, prod in enumerate(("WC", "TC")):
        im = axes[k].imshow(np.where(mask, flags_count[prod], np.nan), extent=ext, cmap="magma_r")
        axes[k].set_title(f"{prod}: flagged months 1970-2024")
        fig.colorbar(im, ax=axes[k], shrink=0.75)
    for k, name in enumerate(("WC_1970_1999", "TC_1970_1999", "TC_2000_2024")):
        ax = axes[k + 2]
        ax.imshow(np.where(mask, wq_maps[name].astype(float), np.nan), extent=ext, cmap="Reds",
                  vmin=0, vmax=1)
        ax.set_title(f"{name}: wettest quarter changed by cleaning")
    for ax in axes:
        ax.scatter(occ["lon"], occ["lat"], s=6, c="cyan", edgecolors="k", linewidths=0.3)
    fig.tight_layout()
    fig.savefig(qa / "03_fig_precip_cleaning_impact.png", dpi=200)
    plt.close(fig)
    return df


EFFECT_CODES = ["period_WC", "period_TC", "interaction", "product_1970_1999",
                "product_2000_2024", "period_mean"]


def effects(clims, occ, mask, grid, qa, out, version="original"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section(f"6. Period effect, product effect and interaction ({version} version)")
    a = {n: annual(clims[(n, version)]) for n in ("WC_1970_1999", "WC_2000_2024",
                                                   "TC_1970_1999", "TC_2000_2024")}
    r, c = occ["row"].values, occ["col"].values
    rows = []
    for k, (label, unit) in enumerate((("annual mean temperature", "degC"),
                                        ("annual precipitation", "mm"))):
        wc70, wc00 = a["WC_1970_1999"][k], a["WC_2000_2024"][k]
        tc70, tc00 = a["TC_1970_1999"][k], a["TC_2000_2024"][k]
        maps = {"Period effect - WorldClim (2000-24 minus 1970-99)": wc00 - wc70,
                "Period effect - TerraClimate (2000-24 minus 1970-99)": tc00 - tc70,
                "Interaction (TC period effect minus WC period effect)": (tc00 - tc70) - (wc00 - wc70),
                "Product effect 1970-99 (TC minus WC)": tc70 - wc70,
                "Product effect 2000-24 (TC minus WC)": tc00 - wc00,
                "Mean period effect (average of both products)": ((wc00 - wc70) + (tc00 - tc70)) / 2}
        w = np.cos(np.deg2rad(grid.lats))[:, None] * mask
        reference = {"period_WC": wc70, "period_TC": tc70, "interaction": None,
                     "product_1970_1999": wc70, "product_2000_2024": wc00, "period_mean": (wc70 + tc70) / 2}
        qcode = "tmean" if unit == "degC" else "prec"
        (out / "effects").mkdir(exist_ok=True)
        for code, (title, m) in zip(EFFECT_CODES, maps.items()):
            write_geotiff(out / "effects" / f"{qcode}_{code}_{version}.tif", np.where(mask, m, np.nan), grid)
            v = np.isfinite(m) & (w > 0)
            iran = float((m * w)[v].sum() / w[v].sum())
            at_occ = m[r, c]
            rel = np.nan
            if unit == "mm" and reference[code] is not None:
                ref_ = reference[code]
                rel = 100.0 * iran / float((ref_ * w)[v].sum() / w[v].sum())
            rows.append(dict(version=version, quantity=label, unit=unit, effect_code=code, effect=title,
                             iran_mean=round(iran, 3),
                             iran_mean_percent_of_reference=round(rel, 2) if np.isfinite(rel) else "",
                             occurrence_mean=round(float(np.nanmean(at_occ)), 3),
                             occurrence_sd=round(float(np.nanstd(at_occ)), 3),
                             occurrence_min=round(float(np.nanmin(at_occ)), 3),
                             occurrence_max=round(float(np.nanmax(at_occ)), 3)))
            print(f"  {label:24s} | {title:55s} | Iran {iran:+8.2f} | occurrences "
                  f"{np.nanmean(at_occ):+8.2f} {unit}")
        lim = 2.0 if unit == "degC" else 100.0
        ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        for ax, (title, m) in zip(axes.ravel(), maps.items()):
            im = ax.imshow(np.where(mask, m, np.nan), extent=ext, cmap="RdBu_r" if unit == "degC" else "BrBG",
                           vmin=-lim, vmax=lim)
            ax.scatter(occ["lon"], occ["lat"], s=6, c="k")
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75, label=unit)
        fig.suptitle(f"{label} - {version} version")
        fig.tight_layout()
        tag = "tmean" if unit == "degC" else "prec"
        fig.savefig(qa / f"03_fig_effects_annual_{tag}_{version}.png", dpi=200)
        plt.close(fig)
    pd.DataFrame(rows).to_csv(qa / f"03_effects_summary_{version}.csv", index=False)
    return rows


ALT_RULES = [(150, 8), (200, 10), (250, 12), (300, 15)]    # (min_mm, factor); (200, 10) = main rule


def cleaning_rule_sensitivity(prec, mask, occ, prod, offset=5.0):
    """How many values each alternative flagging rule would replace (robustness of D19)."""
    r, c = occ["row"].values, occ["col"].values
    occ_mask = np.zeros(mask.shape, dtype=bool)
    occ_mask[r, c] = True
    rows = []
    med = [np.nanmedian(prec[m::12], axis=0) for m in range(12)]
    for min_mm, factor in ALT_RULES:
        n_iran = n_occ = 0
        for m in range(12):
            f = (prec[m::12] >= min_mm) & (prec[m::12] >= factor * (med[m] + offset))
            n_iran += int((f & mask[None]).sum())
            n_occ += int((f & occ_mask[None]).sum())
        rows.append(dict(product=prod, min_mm=min_mm, factor=factor, main_rule=(min_mm, factor) == (200, 10),
                         flagged_cell_months_iran=n_iran, flagged_cell_months_occurrence_cells=n_occ))
    return rows


def period_length_sensitivity(extra_clims, clims, occ, mask, qa):
    """Compare the 25-year baseline 2000-2024 with a 30-year baseline 1995-2024 (WMO length).

    Shows whether using 25 instead of 30 years matters compared with the product effect."""
    from hemiscorpius.bioclim import biovars as py_biovars
    section("8. Robustness: 25-year (2000-2024) vs 30-year (1995-2024) recent baseline")
    r, c = occ["row"].values, occ["col"].values
    key = ["bio1", "bio3", "bio4", "bio12", "bio14", "bio15", "bio17"]
    bio = {}
    for prod in ("WC", "TC"):
        e = extra_clims[prod]
        bio[(prod, "1995_2024")] = py_biovars(e["prec"], e["tmin"], e["tmax"])
        o = clims[(f"{prod}_2000_2024", "original")]
        bio[(prod, "2000_2024")] = py_biovars(o["prec"], o["tmin"], o["tmax"])
    ref = clims[("WC_1970_1999", "original")]
    ref_bio = py_biovars(ref["prec"], ref["tmin"], ref["tmax"])
    rows = []
    for v in key:
        sd = float(np.nanstd(ref_bio[v][mask]))
        prod_eff = bio[("TC", "2000_2024")][v] - bio[("WC", "2000_2024")][v]
        for prod in ("WC", "TC"):
            d = bio[(prod, "2000_2024")][v] - bio[(prod, "1995_2024")][v]
            rows.append(dict(product=prod, variable=v,
                             occ_mean_abs_diff_25_vs_30yr=float(np.nanmean(np.abs(d[r, c]))),
                             std_occ_mean_abs_diff=float(np.nanmean(np.abs(d[r, c]))) / sd if sd else np.nan,
                             std_occ_mean_abs_product_effect=float(np.nanmean(np.abs(prod_eff[r, c]))) / sd if sd else np.nan))
    df = pd.DataFrame(rows)
    df["ratio_period_length_to_product_effect"] = df.std_occ_mean_abs_diff / df.std_occ_mean_abs_product_effect
    df.to_csv(qa / "03_period_length_sensitivity.csv", index=False)
    print("   " + df.round(3).to_string(index=False).replace("\n", "\n   "))
    return df


def annual_outputs(clims, occ, mask, grid, out, qa, manuscript_tables):
    """Annual mean temperature / precipitation rasters per baseline and version, and the
    climate at every occurrence record (candidate supplementary table)."""
    (out / "annual").mkdir(exist_ok=True)
    r, c = occ["row"].values, occ["col"].values
    table = occ[["lon", "lat"]].copy()
    table.insert(0, "record", np.arange(1, len(occ) + 1))
    for (name, ver), clim in clims.items():
        tmean, psum = annual(clim)
        write_geotiff(out / "annual" / f"{name}_{ver}_annual_tmean.tif", np.where(mask, tmean, np.nan), grid)
        write_geotiff(out / "annual" / f"{name}_{ver}_annual_prec.tif", np.where(mask, psum, np.nan), grid)
        table[f"{name}_{ver}_annual_tmean_degC"] = np.round(tmean[r, c], 2)
        table[f"{name}_{ver}_annual_prec_mm"] = np.round(psum[r, c], 1)
    table.to_csv(qa / "03_occurrence_climate_by_baseline.csv", index=False)
    table.to_csv(manuscript_tables / "T03_occurrence_climate_by_baseline.csv", index=False)
    print("  annual rasters: data_processed/03_monthly_climatologies/annual/ | occurrence table: "
          "T03_occurrence_climate_by_baseline.csv")


def manuscript_notes_03(cfg, rule, occ, mask, consistency, impact, eff_rows, flags_count,
                        rule_df=None, plen=None):
    n = ManuscriptNotes(cfg, STEP, "Monthly climatologies of the four baselines")
    if rule_df is not None and len(rule_df):
        for _, row in rule_df.iterrows():
            n.number(f"rule_{row['product']}_{row.min_mm}mm_x{row.factor}_flagged_iran",
                     int(row.flagged_cell_months_iran), "cell-months",
                     f"{row['product']}: values flagged with rule >= {row.min_mm} mm and >= {row.factor} x median",
                     "D19 robustness")
    if plen is not None:
        for _, row in plen.iterrows():
            n.number(f"period_length_ratio_{row['product']}_{row.variable}",
                     round(float(row.ratio_period_length_to_product_effect), 4), "ratio",
                     f"{row['product']} {row.variable}: |2000-2024 minus 1995-2024| relative to |product effect| "
                     f"at occurrences", "baseline length")
    n.number("occurrence_records_on_grid", len(occ), "records", "occurrence records located on the grid")
    for prod in ("WC", "TC"):
        n.number(f"prec_flagged_cell_months_{prod}", int((flags_count[prod] * mask).sum()), "cell-months",
                 f"{prod}: monthly values replaced in the cleaned version (inside Iran)", "D19")
    if consistency is not None:
        for v in VARS:
            s_ = consistency[consistency.variable == v]
            n.number(f"consistency_mae_{v}", float(s_.mae.mean()), "mm" if v == "prec" else "degC",
                     f"WC_1970_1999 (built) vs WorldClim 2.1 normals: mean absolute difference", "QA")
            n.number(f"consistency_r_min_{v}", float(s_.r.min()), "",
                     "lowest monthly spatial correlation with WorldClim 2.1 normals", "QA")
    if impact is not None:
        for _, row in impact.iterrows():
            b = row["baseline"]
            n.number(f"cleaning_occ_cells_changed_{b}", int(row["occ_cells_changed"]), "records",
                     f"{b}: occurrence records whose climatology changes with cleaning", "D19")
            n.number(f"cleaning_occ_max_annual_diff_{b}", float(row["occ_max_annual_diff_mm"]), "mm",
                     f"{b}: largest change of annual precipitation at an occurrence record", "D19")
            n.number(f"cleaning_occ_wettest_quarter_changed_{b}", int(row["occ_wettest_quarter_changed"]),
                     "records", f"{b}: records whose wettest quarter changes with cleaning (bio8 etc.)", "D19")
    for r_ in eff_rows:
        q = "tmean" if r_["unit"] == "degC" else "prec"
        n.number(f"effect_{q}_{r_['effect_code']}_{r_['version']}_iran", r_["iran_mean"], r_["unit"],
                 f"{r_['effect']} - Iran area-weighted mean ({r_['version']})", "R1-1, R2")
        n.number(f"effect_{q}_{r_['effect_code']}_{r_['version']}_occ", r_["occurrence_mean"], r_["unit"],
                 f"{r_['effect']} - mean over occurrence records ({r_['version']})", "R1-1, R2")
    n.text("Methods (draft)", f"""
For each of the four baselines (WorldClim 2.1 monthly weather and TerraClimate v1.1, 1970-1999 and
2000-2024), a 12-month climatology was computed as the mean of each calendar month over the period
(cells with any missing year set to no data); bioclimatic variables were derived from these
climatologies with the same procedure for all baselines. Because both products contained implausible
summer precipitation in the hyper-arid south, a cleaned version was produced for sensitivity analysis:
monthly values >= {rule['min_mm']:.0f} mm and >= {rule['factor']:.0f} times the cell's same-month median
(1970-2024) plus {rule['offset_mm']:.0f} mm were replaced by that median, using the identical rule for both
products. Main analyses used the products as published; the choice was fixed before any model was fitted.
Period effects (2000-2024 minus 1970-1999, within product), product effects (TerraClimate minus WorldClim,
within period) and their interaction were mapped for annual mean temperature and annual precipitation.""")
    n.text("Results (draft placeholder)", """
Report: (i) consistency of the WorldClim monthly climatology with WorldClim 2.1 normals;
(ii) magnitude of period and product effects for Iran and at occurrence records, and the interaction;
(iii) number of occurrence records affected by the precipitation cleaning and by wettest-quarter changes.
All numbers: key_numbers.csv (step 03_monthly_climatologies).""")
    n.save()


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    grid = ref_grid(cfg)
    src = project_path(cfg, "processed") / "02_crop_and_units"
    out = project_path(cfg, "processed") / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    rule = {**DEFAULT_CLEANING, **cfg.get("precip_cleaning", {})}
    write_run_info(qa / "03_run_info.json", CONFIG_FILE)
    manuscript_tables = project_path(cfg, "results") / "manuscript" / "tables"
    manuscript_tables.mkdir(parents=True, exist_ok=True)

    with Report(qa / "03_report.txt"):
        print(f"Step {STEP}")
        print(f"cleaning rule: value >= {rule['min_mm']} mm AND value >= {rule['factor']} x "
              f"(same-month cell median + {rule['offset_mm']} mm) -> replaced by that median")
        if "precip_cleaning" not in cfg:
            note("WARN", "precip_cleaning not in config.yaml - defaults used (add the block for documentation)")

        import rasterio
        with rasterio.open(src / "reference_grid.tif") as r_:
            mask = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)
        print(f"Iran cells: {int(mask.sum())} | occurrence records on grid: {len(occ)}")

        clims, flags_count = {}, {}
        extra_clims, rule_sensitivity_rows = {}, []
        for prod, (prefix, label) in PRODUCTS.items():
            section(f"{'1' if prod == 'WC' else '2'}. {label}: monthly climatologies")
            names = [n for n in cfg["baselines"] if n.startswith(prod)]
            for n in names:
                for ver in VERSIONS:
                    clims[(n, ver)] = {}
            for v in VARS:                                   # one variable in memory at a time
                arr, time = load_monthly(src, prefix, v)
                versions = {"original": arr}
                # robustness: 30-year recent period 1995-2024 (original data, QA only)
                extra_clims.setdefault(prod, {})[v] = climatology(arr, time, 1995, 2024)
                if v == "prec":
                    rule_sensitivity_rows.extend(cleaning_rule_sensitivity(arr, mask, occ, prod))
                    cleaned, flags = clean_precip(arr, rule)
                    flags_count[prod] = flags.sum(axis=0)
                    print(f"  prec: {int((flags & mask[None]).sum())} flagged cell-months inside Iran "
                          f"replaced in the cleaned version")
                    versions["cleaned"] = cleaned
                    del flags
                else:
                    versions["cleaned"] = arr                # temperatures are not modified
                for n in names:
                    b = cfg["baselines"][n]
                    for ver in VERSIONS:
                        clims[(n, ver)][v] = climatology(versions[ver], time, b["start"], b["end"])
                del arr, versions
            for n in names:
                b = cfg["baselines"][n]
                for ver in VERSIONS:
                    clim = clims[(n, ver)]
                    save_baseline(out, n, ver, clim, grid,
                                  {"baseline": n, "product": label, "period": f"{b['start']}-{b['end']}",
                                   "version": ver,
                                   "precip_cleaning_rule": str(rule) if ver == "cleaned" else "none",
                                   "construction": "mean of each calendar month; NaN unless all years valid"})
                    tmean, psum = annual(clim)
                    v_ = mask & np.isfinite(tmean)
                    print(f"  {n} [{ver:8s}] Iran mean: annual temp {np.nanmean(tmean[v_]):6.2f} degC | "
                          f"annual prec {np.nanmean(psum[v_]):7.1f} mm | valid Iran cells {int(v_.sum())}")

        consistency = impact = None
        eff_rows = []
        try:
            consistency = consistency_with_normals(src, clims, mask, qa)
        except Exception as e:
            note("FAIL", f"consistency check crashed: {type(e).__name__}: {e}")
        try:
            impact = cleaning_impact(clims, flags_count, occ, mask, grid, qa, out)
        except Exception as e:
            note("FAIL", f"cleaning impact crashed: {type(e).__name__}: {e}")
        for ver in VERSIONS:
            try:
                eff_rows += effects(clims, occ, mask, grid, qa, out, ver)
            except Exception as e:
                note("FAIL", f"effects ({ver}) crashed: {type(e).__name__}: {e}")
        if eff_rows:
            pd.DataFrame(eff_rows).to_csv(manuscript_tables / "T03_period_product_effects.csv", index=False)
        rule_df = pd.DataFrame(rule_sensitivity_rows)
        if len(rule_df):
            section("7. Robustness of the cleaning rule (alternative thresholds)")
            rule_df.to_csv(qa / "03_cleaning_rule_sensitivity.csv", index=False)
            print("   " + rule_df.to_string(index=False).replace("\n", "\n   "))
        plen = None
        try:
            plen = period_length_sensitivity(extra_clims, clims, occ, mask, qa)
        except Exception as e:
            note("FAIL", f"period-length sensitivity crashed: {type(e).__name__}: {e}")
        try:
            annual_outputs(clims, occ, mask, grid, out, qa, manuscript_tables)
        except Exception as e:
            note("FAIL", f"annual outputs crashed: {type(e).__name__}: {e}")
        try:
            manuscript_notes_03(cfg, rule, occ, mask, consistency, impact, eff_rows, flags_count, rule_df, plen)
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
            print("\n  Step 03 completed. Review figures and tables in results/qa/03_monthly_climatologies/.")


if __name__ == "__main__":
    main()
