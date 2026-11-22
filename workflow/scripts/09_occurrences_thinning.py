"""
Step 09 - Occurrence records and spatial thinning
=================================================

Purpose
    Prepare the occurrence sets used for model calibration and reduce spatial clustering
    (reviewer 1, minor comment 3: removing duplicate 4-km cells does not remove broader clustering).

Method (decision D46)
    - All 47 unique field and literature localities are kept in the study data (maps, Table S1).
    - Distance-based thinning with the R package spThin (Aiello-Lammens et al. 2015): the record with the
      most neighbours closer than the thinning distance is removed (ties at random) until no pair is
      closer; the algorithm is repeated 'reps' times with a fixed seed and the largest retained set is
      used. Main distance 10 km (~2.5 grid cells); sensitivity sets 5 km, 20 km and one record per grid
      cell (no distance thinning). The same records are used for every climatic baseline.
    - Verification: the spThin result is checked for the minimum-distance constraint and compared with
      an independent Python implementation of the same algorithm (src/hemiscorpius/occurrences.py).
    - Clustering is quantified with the Clark-Evans nearest-neighbour index R (R < 1 clustered,
      R ~ 1 random) within a fixed reference window (minimum convex polygon of all 47 records),
      before and after thinning.
    - Records removed by the 10-km thinning are kept as a 'consistency set': after modelling we check
      whether they are predicted suitable. They lie within 10 km of calibration records, so they are NOT
      spatially independent and are not used as an independent validation.

Outputs  data_processed/09_occurrences_thinning/
             occurrences_all.csv                 47 records with grid cell
             occurrences_thin10km.csv            MAIN calibration set
             occurrences_thin05km.csv, occurrences_thin20km.csv, occurrences_onepercell.csv
             consistency_set_removed_by_thin10km.csv
QA       results/qa/09_occurrences_thinning/  09_report.txt, 09_*.csv, 09_fig_*.png
Manuscript results/manuscript/tables/T09_occurrence_sets.csv, notes/09_occurrences_thinning.md
Run      python scripts/09_occurrences_thinning.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import ref_grid  # noqa: E402
from hemiscorpius.occurrences import (load_occurrences, cell_indices, great_circle_km,  # noqa: E402
                                      thin_greedy, clark_evans, nearest_neighbour_km)

STEP = "09_occurrences_thinning"
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def mcp_area_km2(lon, lat):
    """Geodesic area of the minimum convex polygon of the points."""
    from shapely.geometry import MultiPoint
    from pyproj import Geod
    hull = MultiPoint(list(zip(lon, lat))).convex_hull
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(hull)
    return abs(area) / 1e6, hull


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    grid = ref_grid(cfg)
    out = project_path(cfg, "processed") / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "09_run_info.json", CONFIG_FILE)
    oc = cfg["occurrences"]
    main_km = float(oc["thinning_km"])
    sens_km = [float(x) for x in oc.get("thinning_sensitivity_km", [5, 20])]
    reps = int(oc.get("thinning_reps", 100))
    seed = int(cfg["project"]["seed"])

    with Report(qa / "09_report.txt"):
        print(f"Step {STEP}: main thinning {main_km:g} km; sensitivity {sens_km}; reps {reps}; seed {seed}")
        occ = cell_indices(load_occurrences(cfg), grid).reset_index(drop=True)
        occ.insert(0, "id", [f"R{i + 1:02d}" for i in range(len(occ))])
        occ["cell"] = occ["row"] * grid.ncols + occ["col"]
        occ["species"] = "Hemiscorpius_acanthocercus"
        occ.to_csv(out / "occurrences_all.csv", index=False)

        section("1. Record chain")
        chain = {"raw records (original manuscript)": oc.get("n_raw_records", np.nan),
                 "unique localities (file)": len(occ),
                 "unique 1/24-degree cells": occ["cell"].nunique()}
        for k, v in chain.items():
            print(f"  {k:40s} {v}")

        section("2. Thinning with spThin (R) and independent verification (Python)")
        inp = out / "spthin_input.csv"
        occ[["id", "lon", "lat", "species"]].to_csv(inp, index=False)
        dists = sorted(set([main_km] + sens_km))
        res = subprocess.run([cfg["paths"]["rscript"], str(ROOT / "scripts" / "09_spthin.R"), str(inp), str(out),
                              str(seed), str(reps)] + [f"{d:g}" for d in dists], capture_output=True, text=True)
        for ln in (res.stdout + res.stderr).splitlines():
            if ln.strip() and "built under" not in ln and "Warning message" not in ln:
                print(f"    R> {ln}")
        if res.returncode != 0:
            note("FAIL", f"spThin failed (exit {res.returncode})")
            return

        sets, rows = {}, []
        key = occ[["lon", "lat"]].round(6).apply(tuple, axis=1)
        for d in dists:
            tag = f"{int(d):02d}"
            t = pd.read_csv(out / f"thin_{tag}km.csv")
            tk = set(t[["lon", "lat"]].round(6).apply(tuple, axis=1))
            idx = np.flatnonzero(key.isin(tk).values)
            sets[f"thin{tag}km"] = idx
            dd = great_circle_km(occ.lon.values[idx], occ.lat.values[idx])
            np.fill_diagonal(dd, np.inf)
            note("OK" if dd.min() >= d - 1e-6 else "FAIL",
                 f"spThin {d:g} km: {len(idx)} records retained; minimum pairwise distance {dd.min():.2f} km")
            py_idx, py_sizes = thin_greedy(occ.lon.values, occ.lat.values, d, reps=reps, seed=seed)
            r_sizes = pd.read_csv(out / f"thin_{tag}km_sizes.csv")["retained"]
            diff = len(py_idx) - len(idx)
            note("OK" if abs(diff) <= 1 else "WARN",
                 f"  independent Python implementation: max {len(py_idx)} (R: {len(idx)}; "
                 f"R range over reps {r_sizes.min()}-{r_sizes.max()})")
            rows.append(dict(set=f"thin{tag}km", distance_km=d, n_records=len(idx),
                             n_python=len(py_idx), min_pairwise_km=float(dd.min())))
        one = occ.drop_duplicates("cell").index.values
        sets["onepercell"] = one
        rows.append(dict(set="onepercell", distance_km=0, n_records=len(one), n_python=np.nan,
                         min_pairwise_km=float(np.min(nearest_neighbour_km(occ.lon.values[one], occ.lat.values[one])))))

        section("3. Clustering (Clark-Evans R) before and after thinning")
        area, hull = mcp_area_km2(occ.lon.values, occ.lat.values)
        print(f"  reference window: minimum convex polygon of all records, {area:,.0f} km2")
        ce_rows = []
        for name, idx in [("all", np.arange(len(occ)))] + list(sets.items()):
            nn = nearest_neighbour_km(occ.lon.values[idx], occ.lat.values[idx])
            r_ce, z = clark_evans(nn, area)
            ce_rows.append(dict(set=name, n=len(idx), mean_nn_km=float(nn.mean()), clark_evans_R=r_ce, z=z))
            print(f"  {name:12s} n={len(idx):2d}  mean NN {nn.mean():6.2f} km  R={r_ce:5.2f}  z={z:6.2f}")
        ce = pd.DataFrame(ce_rows)
        ce.to_csv(qa / "09_clark_evans.csv", index=False)
        r_all = ce.loc[ce.set == "all", "clark_evans_R"].item()
        r_main = ce.loc[ce.set == f"thin{int(main_km):02d}km", "clark_evans_R"].item()
        note("OK" if r_main > r_all else "WARN", f"Clark-Evans R increased from {r_all:.2f} (all) to {r_main:.2f} "
                                                f"(main set): clustering reduced")

        section("4. Saving occurrence sets")
        for name, idx in sets.items():
            occ.iloc[idx].to_csv(out / f"occurrences_{name}.csv", index=False)
        main_idx = sets[f"thin{int(main_km):02d}km"]
        removed = occ.drop(index=main_idx)
        removed.to_csv(out / "consistency_set_removed_by_thin10km.csv", index=False)
        print(f"  main calibration set: {len(main_idx)} records; consistency set: {len(removed)} records")
        summ = pd.DataFrame(rows).merge(ce[["set", "clark_evans_R", "mean_nn_km"]], on="set", how="left")
        summ.to_csv(qa / "09_occurrence_sets.csv", index=False)
        summ.round(3).to_csv(tables / "T09_occurrence_sets.csv", index=False)
        occ.assign(in_main_calibration_set=occ.index.isin(main_idx))[
            ["id", "lon", "lat", "in_main_calibration_set"]].to_csv(tables / "T09_all_records.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        hx, hy = hull.exterior.xy
        for ax, (title, idx) in zip(axes, [("Main set (10 km)", main_idx),
                                           ("One record per grid cell", sets["onepercell"])]):
            ax.plot(hx, hy, color="grey", lw=0.8)
            ax.scatter(occ.lon, occ.lat, s=25, facecolors="none", edgecolors="red", label="all records (47)")
            ax.scatter(occ.lon.values[idx], occ.lat.values[idx], s=12, c="blue", label=f"retained ({len(idx)})")
            ax.set_title(title)
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            ax.legend(fontsize=8)
            ax.set_aspect(1 / np.cos(np.deg2rad(27)))
        fig.tight_layout()
        fig.savefig(qa / "09_fig_thinning.png", dpi=170)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Occurrence records and spatial thinning")
        n.number("records_raw", chain["raw records (original manuscript)"], "records", "raw records", "R1 minor 2")
        n.number("records_unique_localities", len(occ), "records", "unique localities", "R1 minor 2")
        n.number("records_unique_cells", int(occ["cell"].nunique()), "cells", "records in distinct grid cells", "R1 minor 2")
        for _, rr in summ.iterrows():
            n.number(f"n_{rr['set']}", int(rr.n_records), "records", f"records retained in set {rr['set']}", "R1 minor 3")
        for _, rr in ce.iterrows():
            n.number(f"clark_evans_R_{rr['set']}", round(rr.clark_evans_R, 3), "", f"Clark-Evans R, set {rr['set']}",
                     "R1 minor 3")
        n.number("mcp_area_km2", round(area), "km2", "reference window for Clark-Evans (MCP of all records)", "")
        n.text("Methods (draft)", f"""
Of the 48 raw records, 47 unique localities fell in {occ['cell'].nunique()} distinct grid cells. To reduce spatial
clustering beyond the removal of duplicate cells, records were thinned with spThin (Aiello-Lammens et al.
2015) at a minimum distance of {main_km:g} km (~2.5 grid cells), keeping the largest set over {reps} randomised
runs ({len(main_idx)} records); the result was verified against an independent implementation of the algorithm.
Clustering, measured with the Clark-Evans index within the minimum convex polygon of all records, changed
from R = {r_all:.2f} to R = {r_main:.2f}. The same calibration records were used for all climatic baselines.
Sets thinned at {', '.join(f'{k:g}' for k in sens_km)} km and with one record per grid cell were used for sensitivity
analyses. All 47 records are reported (Table S1); the records not used for calibration were used only to
check that they fall in areas predicted suitable, as they are not spatially independent of the calibration set.""")
        n.save()

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails + warns:
            print(f"   - {m}")
        if not fails:
            print("\n  Step 09 completed.")


if __name__ == "__main__":
    main()
