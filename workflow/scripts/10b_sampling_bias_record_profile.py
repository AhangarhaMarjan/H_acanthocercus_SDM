"""
Step 10b - Sampling bias (roads, settlements) and profile of every record
=========================================================================

Purpose
    (1) Test whether the records are closer to roads and settlements than the accessible area as a whole
        (sampling bias towards accessible places), and build a bias-corrected background for a
        sensitivity analysis. (2) Build a complete profile of every record for the supplement and the
        Discussion (province, ecoregion, terrain, soil, climate of every baseline, distance to roads and
        settlements).

Method (decision D56, fixed before any bias-corrected model was fitted)
    Data      OpenStreetMap (Geofabrik Iran extract, data up to 2026-10-28; ODbL). Roads usable by vehicles
              (motorway to track, incl. links, residential, service, unclassified); settlements = city, town,
              village, hamlet, suburb, national capital (point layer).
    Distance  exact point-to-feature distances (vector, UTM 40N) at POINT support for both records and
              random points inside the main M (the first version compared record points with 4-km cell means,
              which inflated the contrast; corrected before any bias-corrected model was used).
    Bias test record distances vs random points in the main M: medians, shares within 0.5/1/5 km,
              one-sided Mann-Whitney U test.
    Correction background drawn WITH replacement from the cells of the main M with probability
              proportional to the cell mean of exp(-d / sigma) over a 4x4 lattice of sub-points (expected sampling
              intensity of the cell), d = exact distance to the nearest road or settlement and sigma = mean record
              distance (exponential MLE; floor 0.1 km for positional precision), i.e.
              the background reproduces the accessibility bias of the records (bias-file approach: Phillips et
              al. 2009; Kramer-Schadt et al. 2013; Fourcade et al. 2014). Distances are NOT used as predictors.

Outputs  data_processed/10_accessible_area_background/background_main_biascorr.csv
         data_processed/10b_sampling_bias_record_profile/distance_road_km.tif, distance_settlement_km.tif,
         bias_weight.tif (grid 1/24 deg), record_profile.csv
QA       results/qa/10b_sampling_bias_record_profile/  10b_report.txt, 10b_*.csv, 10b_fig_*.png
Manuscript results/manuscript/tables/T10b_record_profile.csv, T10b_sampling_bias_test.csv
Run      python scripts/10b_sampling_bias_record_profile.py
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
from hemiscorpius.grid import ref_grid, write_geotiff, CELL  # noqa: E402
from hemiscorpius.occurrences import great_circle_km_between  # noqa: E402

STEP = "10b_sampling_bias_record_profile"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
PROFILE_CLIMATE = ["bio1", "bio3", "bio4", "bio5", "bio6", "bio12", "bio15"]
PROFILE_STATIC = ["elevation", "slope", "northness", "eastness", "vrm", "sand", "clay", "cfvo", "soc", "phh2o", "bdod"]
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rasterio
    import xarray as xr
    from scipy.stats import mannwhitneyu

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "10b_run_info.json", CONFIG_FILE)
    sb = cfg["sampling_bias"]
    seed = int(cfg["project"]["seed"])
    bb = cfg["grid"]["bbox"]

    with Report(qa / "10b_report.txt"):
        print(f"Step {STEP}")
        occ = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_all.csv")
        cal = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_thin10km.csv")
        with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
            m_main = np.isfinite(r_.read(1)) & (r_.read(1) == 1)

        section("1. OpenStreetMap layers")
        osm = raw_dir(cfg, "osm")
        roads = gpd.read_file(next(osm.rglob(sb["roads_file"])), bbox=(bb["west"], bb["south"], bb["east"], bb["north"]))
        places = gpd.read_file(next(osm.rglob(sb["places_file"])), bbox=(bb["west"], bb["south"], bb["east"], bb["north"]))
        print(f"  roads read: {len(roads):,} (classes: {roads.fclass.value_counts().head(12).to_dict()})")
        print(f"  places read: {len(places):,} (classes: {places.fclass.value_counts().to_dict()})")
        roads = roads[roads.fclass.isin(sb["road_classes"])].to_crs("EPSG:4326")
        places = places[places.fclass.isin(sb["place_classes"])].to_crs("EPSG:4326")
        print(f"  used: {len(roads):,} road segments; {len(places):,} settlements")
        note("OK" if len(roads) > 1000 and len(places) > 1000 else "WARN",
             f"OSM features used: roads {len(roads):,}, settlements {len(places):,}")

        section("2. Exact distances (vector, UTM 40N) at the same support for records and accessible area")
        from shapely import STRtree, points as shp_points
        utm = "EPSG:32640"
        r_utm, p_utm = roads.to_crs(utm), places.to_crs(utm)
        t_road, t_set = STRtree(r_utm.geometry.values), STRtree(p_utm.geometry.values)

        def dist_km(lon, lat):
            g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(utm).values
            _, d1 = t_road.query_nearest(g, return_distance=True, all_matches=False)
            _, d2 = t_set.query_nearest(g, return_distance=True, all_matches=False)
            return d1 / 1000.0, d2 / 1000.0

        rec = {}
        rec["road"], rec["settlement"] = dist_km(occ.lon.values, occ.lat.values)
        rec["access"] = np.minimum(rec["road"], rec["settlement"])
        rng = np.random.default_rng(seed + 10)
        mr, mc = np.where(m_main)
        k = int(sb["n_test_points"])
        pick = rng.integers(0, len(mr), k)
        half = CELL / 2
        lon_t = grid.lons[mc[pick]] + rng.uniform(-half, half, k)
        lat_t = grid.lats[mr[pick]] + rng.uniform(-half, half, k)
        m_vals = {}
        m_vals["road"], m_vals["settlement"] = dist_km(lon_t, lat_t)
        m_vals["access"] = np.minimum(m_vals["road"], m_vals["settlement"])
        print(f"  {len(occ)} records and {k:,} random points inside the main M (point support, exact distances)")

        section("3. Are the records closer to roads and settlements than random points in the accessible area?")
        rows = []
        for kk in ("road", "settlement", "access"):
            r_, m_ = rec[kk], m_vals[kk]
            u = mannwhitneyu(r_, m_, alternative="less")
            rows.append(dict(distance=kk, records_median_km=float(np.median(r_)), M_median_km=float(np.median(m_)),
                             records_share_within_0_5km=float(np.mean(r_ <= 0.5)), M_share_within_0_5km=float(np.mean(m_ <= 0.5)),
                             records_share_within_1km=float(np.mean(r_ <= 1)), M_share_within_1km=float(np.mean(m_ <= 1)),
                             records_share_within_5km=float(np.mean(r_ <= 5)), M_share_within_5km=float(np.mean(m_ <= 5)),
                             mann_whitney_p_records_closer=float(u.pvalue)))
        bt = pd.DataFrame(rows)
        bt.to_csv(qa / "10b_sampling_bias_test.csv", index=False)
        bt.round(5).to_csv(tables / "T10b_sampling_bias_test.csv", index=False)
        print("   " + bt.round(3).to_string(index=False).replace("\n", "\n   "))
        biased = bt.loc[bt.distance == "access", "mann_whitney_p_records_closer"].item() < 0.05
        note("OK", "records are significantly closer to roads/settlements than the accessible area (sampling bias)"
             if biased else "no significant accessibility bias of the records (correction reported as sensitivity)")

        section("4. Bias-corrected background (sensitivity variant 'bias_corrected')")
        sigma_mle = float(np.mean(rec["access"]))
        sigma = max(sigma_mle, float(sb["sigma_floor_km"]))
        print(f"  sigma = mean record distance {sigma_mle:.3f} km (floor {sb['sigma_floor_km']} km for positional "
              f"precision of records and OSM) -> {sigma:.3f} km")
        n_sub = int(sb["subpoints_per_side"])
        off = (np.arange(n_sub) + 0.5) / n_sub * CELL - half
        ox, oy = np.meshgrid(off, off)
        lon_s = (grid.lons[mc][:, None] + ox.ravel()[None, :]).ravel()
        lat_s = (grid.lats[mr][:, None] + oy.ravel()[None, :]).ravel()
        d_r, d_s = dist_km(lon_s, lat_s)
        d_acc_sub = np.minimum(d_r, d_s).reshape(len(mr), -1)
        w_cell = np.exp(-d_acc_sub / sigma).mean(axis=1)          # expected sampling intensity of the cell
        dmean_cell = d_acc_sub.mean(axis=1)
        w = np.zeros(grid.shape, "float32"); w[mr, mc] = w_cell
        dgrid = np.full(grid.shape, np.nan, "float32"); dgrid[mr, mc] = dmean_cell
        write_geotiff(out / "bias_weight.tif", np.where(m_main, w, np.nan).astype("float32"), grid)
        write_geotiff(out / "distance_access_mean_km.tif", dgrid, grid)
        p = w_cell / w_cell.sum()
        pick = rng.choice(len(mr), size=int(sb["n_background"]), replace=True, p=p)
        b = pd.DataFrame({"row": mr[pick], "col": mc[pick], "lon": grid.lons[mc[pick]], "lat": grid.lats[mr[pick]]})
        b.to_csv(proc / "10_accessible_area_background" / "background_main_biascorr.csv", index=False)
        eff = 1.0 / np.sum(p ** 2)
        print(f"  {len(b)} background points, {b[['row', 'col']].drop_duplicates().shape[0]} distinct cells; "
              f"effective number of cells {eff:,.0f} of {len(mr):,}")
        bd = dgrid[b.row.values, b.col.values]
        print(f"  mean cell distance to road/settlement: corrected background median {np.median(bd):.2f} km | "
              f"all M cells median {np.median(dmean_cell):.2f} km")
        note("OK" if eff >= 1000 else "WARN", f"effective number of background cells {eff:,.0f}")

        section("5. Profile of every record")
        prov = gpd.read_file(next(raw_dir(cfg, "static").rglob(cfg["paths"]["static_files"]["provinces"]))).to_crs("EPSG:4326")
        eco = gpd.read_file(next(raw_dir(cfg, "static").rglob(cfg["paths"]["static_files"]["ecoregions"]))).to_crs("EPSG:4326")
        eco = eco.cx[bb["west"]:bb["east"], bb["south"]:bb["north"]].copy()
        from shapely import make_valid
        n_bad = int((~eco.is_valid).sum())
        eco["geometry"] = eco.geometry.apply(make_valid)      # 17 of 31 RESOLVE polygons in the box are invalid (D57)
        prov["geometry"] = prov.geometry.apply(make_valid)
        print(f"  ecoregion polygons repaired with make_valid: {n_bad} of {len(eco)}")
        pts = gpd.GeoDataFrame(occ, geometry=gpd.points_from_xy(occ.lon, occ.lat), crs="EPSG:4326")
        name_col = next(c for c in prov.columns if c.upper() == "NAME_1")
        # point-within-polygon first (as in step 10); nearest polygon (UTM) only for points on an edge
        def assign(poly, col):
            j = gpd.sjoin(pts, poly[[col, "geometry"]], how="left", predicate="within").groupby(level=0).first()
            miss = j[col].isna()
            if miss.any():
                jn = gpd.sjoin_nearest(pts[miss].to_crs(utm), poly[[col, "geometry"]].to_crs(utm),
                                       how="left").groupby(level=0).first()
                j.loc[miss, col] = jn[col]
            return j[col].values
        jp_vals, je_vals = assign(prov, name_col), assign(eco, "ECO_NAME")
        prof = occ[["id", "lon", "lat"]].copy()
        prof["in_main_calibration_set"] = prof["id"].isin(cal["id"])
        prof["province"] = jp_vals
        prof["ecoregion"] = je_vals
        with xr.open_dataset(proc / "07_static_layers" / "static_predictors.nc") as ds:
            for v in PROFILE_STATIC:
                prof[v] = np.round(ds[v].values[occ.row.values, occ.col.values], 3)
        for bname in BASELINES:
            with xr.open_dataset(proc / "04_bioclim" / f"{bname}_original_bioclim.nc") as ds:
                for v in PROFILE_CLIMATE:
                    prof[f"{bname}_{v}"] = np.round(ds[v].values[occ.row.values, occ.col.values], 2)
        prof["dist_road_km"] = np.round(rec["road"], 3)
        prof["dist_settlement_km"] = np.round(rec["settlement"], 3)
        sx = np.c_[places.geometry.x.values, places.geometry.y.values]
        dd = great_circle_km_between(occ.lon.values, occ.lat.values, sx[:, 0], sx[:, 1])
        near = np.argmin(dd, axis=1)
        prof["nearest_settlement"] = places.iloc[near]["name"].fillna("").values
        prof["nearest_settlement_class"] = places.iloc[near]["fclass"].values
        prof.to_csv(out / "record_profile.csv", index=False)
        prof.to_csv(tables / "T10b_record_profile.csv", index=False, encoding="utf-8-sig")
        summ = prof[["elevation", "slope", "vrm", "dist_road_km", "dist_settlement_km",
                     "TC_2000_2024_bio6", "TC_2000_2024_bio5", "TC_2000_2024_bio12"]].describe().T[["min", "50%", "max"]]
        summ.to_csv(qa / "10b_record_profile_summary.csv")
        print("   " + summ.round(2).to_string().replace("\n", "\n   "))
        print("\n  records per province: " + prof.province.value_counts().to_dict().__repr__())
        print("  records per ecoregion: " + prof.ecoregion.value_counts().to_dict().__repr__())

        section("6. Check: are the accessible areas affected by invalid ecoregion polygons? (D57)")
        from rasterio.features import rasterize
        occ_names = set(prof.ecoregion)
        occ_eco = eco[eco.ECO_NAME.isin(occ_names)]
        r_valid = rasterize(((g, 1) for g in occ_eco.geometry), out_shape=grid.shape, transform=grid.transform,
                            fill=0, dtype="uint8").astype(bool)
        with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
            m_main_used = r_.read(1) == 1
        only_used = int((m_main_used & ~r_valid).sum())
        print(f"  occupied ecoregions (point-in-polygon, repaired): {sorted(occ_names)}")
        print(f"  cells of the main M outside the repaired occupied ecoregions: {only_used} "
              f"({only_used / max(m_main_used.sum(), 1):.2%} of {int(m_main_used.sum())})")
        note("OK" if only_used / max(m_main_used.sum(), 1) < 0.01 else "WARN",
             f"main M agrees with the repaired ecoregion polygons ({only_used} cells differ)")

        # ------------------------------------------------------------------ figure
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, k in zip(axes, ("road", "settlement", "access")):
            m_ = np.sort(m_vals[k])
            ax.plot(m_, np.linspace(0, 1, len(m_)), color="grey", label="random points in the accessible area")
            r_ = np.sort(rec[k])
            ax.plot(r_, np.linspace(0, 1, len(r_)), color="red", label="records (47)")
            ax.set_xscale("symlog", linthresh=0.1)
            ax.set_xlabel(f"distance to nearest {k} (km)")
            ax.set_ylabel("cumulative share")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(qa / "10b_fig_distance_distributions.png", dpi=170)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Sampling bias and record profile")
        for _, r in bt.iterrows():
            n.number(f"records_median_dist_{r.distance}_km", round(r.records_median_km, 2), "km",
                     f"median record distance to nearest {r.distance}", "sampling bias")
            n.number(f"M_median_dist_{r.distance}_km", round(r.M_median_km, 2), "km",
                     f"median distance in the main M to nearest {r.distance}", "sampling bias")
            n.number(f"mann_whitney_p_{r.distance}", round(r.mann_whitney_p_records_closer, 5), "",
                     f"one-sided test: records closer to {r.distance} than the M", "sampling bias")
        n.number("bias_sigma_km", round(sigma, 3), "km", "scale of the exponential accessibility weight", "D56")
        n.text("Methods (draft)", f"""
Because many records were collected near roads and villages, sampling bias was assessed with OpenStreetMap roads
and settlements (Geofabrik extract, data to 28 October 2026). Distances from records and from {k:,} sampled points in the
calibration area to the nearest road or settlement were compared with a one-sided Mann-Whitney test. For a
sensitivity analysis, background cells were drawn with replacement with probability proportional to the within-cell
mean of exp(-d/sigma) over a 4 x 4 sub-point lattice, where d is
the distance to the nearest road or settlement and sigma ({sigma:.1f} km) the mean distance of the records, so that the
background reproduced the accessibility bias of the records (Phillips et al. 2009; Kramer-Schadt et al. 2013;
Fourcade et al. 2014). Distances were not used as predictors.""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 10b completed.")


if __name__ == "__main__":
    main()
