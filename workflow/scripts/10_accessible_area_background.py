"""
Step 10 - Accessible area (M) and background points
===================================================

Purpose
    Define the calibration area M (reviewer 1, comment 2) and draw the background points.

Definitions (decisions D47-D48, fixed before any model was fitted)
    MAIN  B'  RESOLVE ecoregions (2017) containing at least one of the 47 known localities, restricted to
              cells within 'main_restriction_km' (200 km) of the localities (Barve et al. 2011: biotic regions
              as M hypotheses, limited by plausible dispersal). D48: the unrestricted ecoregions extend
              > 1,000 km into cold montane NW/NE Iran because 7 records lie at the margins of two large
              ecoregions; amended before any model was fitted.
    SENS  B   the full occupied ecoregions (original plan, D47).
    SENS  C   cells within 'buffer_km' (100 km) of the localities (occurrence-based buffer).
    SENS  A   the whole domain (Iran; approach of the original manuscript).
    M is defined from all 47 known localities (the known range of the species) and is identical for
    every baseline, version and occurrence set. Distances are great-circle distances to cell centres.

Background
    'n_points' cells drawn at random (fixed seed) inside each M, at most one per cell; all cells are used
    when M has fewer cells. Presence cells are not excluded (MaxEnt default; background describes the
    environments available in M).

Checks
    Records inside each M; area of each M; environmental contrast between calibration records and
    background (standardised mean difference) for key predictors, which explains why a national
    background inflates AUC.

Outputs  data_processed/10_accessible_area_background/
             M_main.tif, M_ecoregion.tif, M_buffer100km.tif, M_national.tif (1 = inside)
             background_{M}.csv  (row, col, lon, lat)
             occupied_ecoregions.csv
QA       results/qa/10_accessible_area_background/  10_report.txt, 10_*.csv, 10_fig_*.png
Manuscript results/manuscript/tables/T10_accessible_areas.csv, notes/10_accessible_area_background.md
Run      python scripts/10_accessible_area_background.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, raw_dir, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import CELL, ref_grid, write_geotiff, cell_area_km2  # noqa: E402
from hemiscorpius.occurrences import great_circle_km_between  # noqa: E402

STEP = "10_accessible_area_background"
KEY_VARS = ["bio1", "bio4", "bio12", "bio15", "elevation", "slope"]
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def min_distance_to_points(grid, rows, cols, lon_pts, lat_pts, chunk=20000):
    """Great-circle distance (km) from each listed cell centre to the nearest point."""
    lon_c, lat_c = grid.lons[cols], grid.lats[rows]
    out = np.empty(len(rows), dtype="float32")
    for i in range(0, len(rows), chunk):
        d = great_circle_km_between(lon_c[i:i + chunk], lat_c[i:i + chunk], lon_pts, lat_pts)
        out[i:i + chunk] = d.min(axis=1)
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import xarray as xr
    import geopandas as gpd
    from rasterio.features import rasterize

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "10_run_info.json", CONFIG_FILE)
    aa, bg = cfg["accessible_area"], cfg["background"]
    seed = int(cfg["project"]["seed"])

    with Report(qa / "10_report.txt"):
        print(f"Step {STEP}")
        with rasterio.open(proc / "08_common_mask" / "common_mask.tif") as r_:
            domain = r_.read(1) == 1
        area = cell_area_km2(grid)
        occ = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_all.csv")
        cal = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_thin10km.csv")
        rr, cc = np.where(domain)
        a_dom = float(area[domain].sum())

        # ------------------------------------------------------------------ B: ecoregions
        section("1. Ecoregions containing the 47 known localities (RESOLVE 2017)")
        hits = list(raw_dir(cfg, "static").rglob(cfg["paths"]["static_files"]["ecoregions"]))
        eco = gpd.read_file(hits[0]).to_crs("EPSG:4326")
        bb = cfg["grid"]["bbox"]
        eco = eco.cx[bb["west"]:bb["east"], bb["south"]:bb["north"]].copy()
        pts = gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ.lon, occ.lat), crs="EPSG:4326")
        j = gpd.sjoin(pts, eco[["ECO_ID", "ECO_NAME", "geometry"]], how="left", predicate="within")
        miss = j[j.ECO_ID.isna()]
        if len(miss):
            near = gpd.sjoin_nearest(pts.loc[miss.index], eco[["ECO_ID", "ECO_NAME", "geometry"]], how="left")
            j.loc[miss.index, ["ECO_ID", "ECO_NAME"]] = near[["ECO_ID", "ECO_NAME"]].values
            note("WARN", f"{len(miss)} locality/ies on an ecoregion edge assigned to the nearest ecoregion")
        j = j[~j.index.duplicated()]
        eco_ids = sorted(j.ECO_ID.dropna().astype(int).unique())
        eco_r = rasterize(((g, int(i)) for g, i in zip(eco.geometry, eco.ECO_ID)), out_shape=grid.shape,
                          transform=grid.transform, fill=0, dtype="int32")
        rows = []
        for eid in eco_ids:
            name = j.loc[j.ECO_ID == eid, "ECO_NAME"].iloc[0]
            cells = domain & (eco_r == eid)
            rows.append(dict(ECO_ID=eid, ECO_NAME=name, records=int((j.ECO_ID == eid).sum()),
                             cells_in_domain=int(cells.sum()), area_km2=float(area[cells].sum()),
                             share_of_domain=float(area[cells].sum() / a_dom)))
        eco_tab = pd.DataFrame(rows)
        eco_tab.to_csv(out / "occupied_ecoregions.csv", index=False)
        print("   " + eco_tab.round(3).to_string(index=False).replace("\n", "\n   "))
        m_eco = domain & np.isin(eco_r, eco_ids)
        share = float(area[m_eco].sum() / a_dom)
        print(f"  B (occupied ecoregions): {int(m_eco.sum())} cells, {area[m_eco].sum():,.0f} km2, "
              f"{share:.1%} of the domain")

        # ------------------------------------------------------------------ distances
        dist = np.full(grid.shape, np.inf, dtype="float32")
        dist[rr, cc] = min_distance_to_points(grid, rr, cc, occ.lon.values, occ.lat.values)
        m_buf = domain & (dist <= float(aa["buffer_km"]))
        restrict_km = float(aa["main_restriction_km"])
        m_main = m_eco & (dist <= restrict_km)
        main_label = f"B' (occupied ecoregions within {restrict_km:g} km of the localities)"
        far = m_eco & (dist > 1000)
        print(f"  cells of B farther than 1000 km from any locality: {int(far.sum())} "
              f"({area[far].sum():,.0f} km2) - reason for the restriction (D48)")
        note("OK", f"main M = {main_label}")
        Ms = {"main": m_main, "ecoregion": m_eco, f"buffer{int(aa['buffer_km'])}km": m_buf, "national": domain}

        section("2. Accessible areas")
        rows = []
        r_o, c_o = occ["row"].values, occ["col"].values
        for name, m in Ms.items():
            write_geotiff(out / f"M_{name}.tif", np.where(m, 1.0, np.nan).astype("float32"), grid)
            inside = int(m[r_o, c_o].sum())
            note("OK" if inside == len(occ) else "FAIL", f"M_{name}: {inside}/{len(occ)} localities inside")
            rows.append(dict(M=name, definition=main_label if name == "main" else name, cells=int(m.sum()),
                             area_km2=float(area[m].sum()), share_of_domain=float(area[m].sum() / a_dom)))
        mt = pd.DataFrame(rows)
        mt.to_csv(qa / "10_accessible_areas.csv", index=False)
        mt.round(3).to_csv(tables / "T10_accessible_areas.csv", index=False)
        print("   " + mt.round(3).to_string(index=False).replace("\n", "\n   "))

        section("3. Background points")
        rng = np.random.default_rng(seed)
        n_bg = int(bg["n_points"])
        bg_sets = {}
        for name, m in Ms.items():
            r_m, c_m = np.where(m)
            k = min(n_bg, len(r_m))
            pick = np.sort(rng.choice(len(r_m), size=k, replace=False))
            b = pd.DataFrame({"row": r_m[pick], "col": c_m[pick], "lon": grid.lons[c_m[pick]],
                              "lat": grid.lats[r_m[pick]]})
            b.to_csv(out / f"background_{name}.csv", index=False)
            bg_sets[name] = b
            print(f"  background_{name}: {k} points ({'all cells' if k == len(r_m) else 'random sample'})")

        section("4. Environmental contrast: calibration records vs background (standardised mean difference)")
        con = []
        for b in ("WC_2000_2024", "TC_2000_2024"):
            with xr.open_dataset(proc / "08_common_mask" / f"{b}_original_candidates.nc") as ds:
                stack = {v: ds[v].values for v in KEY_VARS}
            for name, b_pts in bg_sets.items():
                for v in KEY_VARS:
                    p = stack[v][cal["row"].values, cal["col"].values]
                    q = stack[v][b_pts["row"].values, b_pts["col"].values]
                    con.append(dict(baseline=b, M=name, variable=v,
                                    smd=float((np.nanmean(p) - np.nanmean(q)) / np.nanstd(q))))
        con = pd.DataFrame(con)
        con.to_csv(qa / "10_environmental_contrast.csv", index=False)
        piv = con.groupby(["M", "variable"]).smd.mean().unstack()
        piv["mean_abs"] = piv.abs().mean(axis=1)
        print("   " + piv.round(2).to_string().replace("\n", "\n   "))
        order = piv["mean_abs"].sort_values()
        print("  mean |contrast| from smallest to largest: "
              + ", ".join(f"{k} {v:.2f}" for k, v in order.items()))
        if piv.loc["national", "mean_abs"] > piv.loc["main", "mean_abs"]:
            note("OK", f"contrast is larger for the national M ({piv.loc['national', 'mean_abs']:.2f}) than for the "
                       f"main M ({piv.loc['main', 'mean_abs']:.2f}): a national background makes discrimination easier")

        # ------------------------------------------------------------------ figure
        ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
        for ax, name in zip(axes, ["main", "ecoregion", f"buffer{int(aa['buffer_km'])}km", "national"]):
            ax.imshow(np.where(domain, 0.25, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=1)
            ax.imshow(np.where(Ms[name], 1.0, np.nan), extent=ext, cmap="Blues", vmin=0, vmax=1.4, alpha=0.8)
            ax.scatter(occ.lon, occ.lat, s=8, c="red")
            ax.set_title(f"M {name}: {mt.loc[mt.M == name, 'area_km2'].item():,.0f} km2"
                         + (f"\n{main_label}" if name == "main" else ""), fontsize=10)
        fig.tight_layout()
        fig.savefig(qa / "10_fig_accessible_areas.png", dpi=170)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Accessible area M and background")
        n.number("main_M_definition", main_label, "", "definition of the main accessible area", "R1-2")
        for _, row in mt.iterrows():
            n.number(f"M_{row.M}_area_km2", round(row.area_km2), "km2", f"area of M {row.M}", "R1-2")
            n.number(f"M_{row.M}_share_of_domain", round(row.share_of_domain, 4), "fraction",
                     f"share of the domain covered by M {row.M}", "R1-2")
        n.number("n_occupied_ecoregions", len(eco_ids), "ecoregions", "ecoregions containing localities", "R1-2")
        for M_, val in piv["mean_abs"].items():
            n.number(f"contrast_mean_abs_smd_{M_}", round(float(val), 3), "SD units",
                     f"mean |standardised difference| records vs background, M {M_}", "R1-2")
        names_ = "; ".join(f"{r.ECO_NAME} ({r.records} records)" for r in eco_tab.itertuples())
        n.text("Methods (draft)", f"""
The calibration area M was based on the RESOLVE ecoregions (Dinerstein et al. 2017) containing the known
localities of H. acanthocercus ({names_}), following the proposal to use biotic regions as hypotheses of the
area accessible to a species (Barve et al. 2011). Because seven localities lay at the margins of two large
ecoregions that extend more than 1,000 km into cold montane north-western and north-eastern Iran, M was
restricted, before any model was fitted, to the occupied ecoregions within {restrict_km:g} km of the localities
({mt.loc[mt.M == 'main', 'area_km2'].item():,.0f} km2). As sensitivity analyses, models were also calibrated with the full
occupied ecoregions, with a {aa['buffer_km']}-km occurrence buffer, as requested by the reviewer, and with the whole of
Iran (the original approach). Up to {n_bg} background cells were drawn at random within each M.""")
        n.save()

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails + warns:
            print(f"   - {m}")
        if not fails:
            print("\n  Step 10 completed.")


if __name__ == "__main__":
    main()
