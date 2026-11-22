"""
Step 08 - Common modelling domain and candidate-predictor stacks
================================================================

Purpose
    Define one modelling domain shared by every baseline, version, future and static layer, and build
    one stack of the 30 candidate predictors (bio1-bio19 + 11 static) for each baseline and version.

Why
    MaxEnt silently drops cells with missing values, and comparisons of areas, overlap (Jaccard,
    Schoener's D) and projections are only valid if every model uses exactly the same cells.
    A cell belongs to the domain only if it lies in Iran (GADM 4.1, cell-centre rule) AND has a value
    in every candidate layer of every baseline, version and future set.

Checks
    Cells removed by each source, occurrence records outside the domain (must be 0), geodesic area of
    the domain compared with the area of Iran.

Inputs   02 reference grid; 04 bioclim (8 sets); 05 future bioclim (all sets); 07 static layers
Outputs  data_processed/08_common_mask/common_mask.tif           1 = modelling domain
         data_processed/08_common_mask/{baseline}_{version}_candidates.nc   30 candidate predictors
         data_processed/08_common_mask/cell_area_km2.tif
QA       results/qa/08_common_mask/  08_report.txt, 08_run_info.json, 08_*.csv, 08_fig_*.png
Run      python scripts/08_common_mask.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import ref_grid, to_dataarray, nc_encoding, write_geotiff, cell_area_km2  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402

STEP = "08_common_mask"
BIO = [f"bio{i}" for i in range(1, 20)]
VERSIONS = ("original", "cleaned")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def load_nc(path, names=None):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        names = names or list(ds.data_vars)
        return {v: ds[v].values.astype("float32") for v in names}


def all_finite(layers):
    m = None
    for a in layers.values():
        f = np.isfinite(a)
        m = f if m is None else (m & f)
    return m


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    write_run_info(qa / "08_run_info.json", CONFIG_FILE)
    static_names = cfg["predictors"]["candidates"]["terrain"] + cfg["predictors"]["candidates"]["soil"]

    with Report(qa / "08_report.txt"):
        print(f"Step {STEP}")
        with rasterio.open(proc / "02_crop_and_units" / "reference_grid.tif") as r_:
            iran = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)
        area = cell_area_km2(grid)
        write_geotiff(out / "cell_area_km2.tif", area, grid)

        section("1. Building the common domain (Iran AND valid in every layer)")
        mask = iran.copy()
        rows = [dict(source="Iran (GADM 4.1, cell centre)", cells_remaining=int(mask.sum()), cells_removed=0)]

        def apply(label, valid):
            nonlocal mask
            before = int(mask.sum())
            mask &= valid
            rows.append(dict(source=label, cells_remaining=int(mask.sum()), cells_removed=before - int(mask.sum())))
            print(f"  {label:45s} removed {before - int(mask.sum()):6d} | remaining {int(mask.sum())}")

        static = load_nc(proc / "07_static_layers" / "static_predictors.nc", static_names)
        apply("static predictors (terrain + soil)", all_finite(static))
        baselines = {}
        for b in cfg["baselines"]:
            for v in VERSIONS:
                baselines[(b, v)] = load_nc(proc / "04_bioclim" / f"{b}_{v}_bioclim.nc", BIO)
        apply("baseline bioclim (4 baselines x 2 versions)", all_finite({f"{k}{i}": a for k, d in baselines.items()
                                                                          for i, a in d.items()}))
        fut_files = sorted((proc / "05_future_delta").glob("*_bioclim.nc"))
        fv = np.ones(grid.shape, dtype=bool)
        for f in fut_files:
            fv &= all_finite(load_nc(f, BIO))
        apply(f"future bioclim ({len(fut_files)} sets)", fv)
        pd.DataFrame(rows).to_csv(qa / "08_cells_removed_by_source.csv", index=False)

        section("2. Domain checks")
        r, c = occ["row"].values, occ["col"].values
        outside = occ[~mask[r, c]]
        note("OK" if outside.empty else "FAIL",
             f"occurrence records outside the domain: {len(outside)}"
             + ("" if outside.empty else f" -> {outside[['lon', 'lat']].values.tolist()}"))
        a_dom, a_iran = float(area[mask].sum()), float(area[iran].sum())
        print(f"  domain: {int(mask.sum())} cells, {a_dom:,.0f} km2 ({100 * a_dom / a_iran:.2f} % of Iran cells' area "
              f"{a_iran:,.0f} km2)")
        note("OK" if 1.55e6 < a_iran < 1.75e6 else "WARN",
             f"area of Iran on the grid {a_iran:,.0f} km2 (official ~1,648,195 km2)")
        write_geotiff(out / "common_mask.tif", np.where(mask, 1.0, np.nan).astype("float32"), grid)

        section("3. Candidate-predictor stacks (30 variables) per baseline and version")
        for (b, v), bio in baselines.items():
            layers = {**bio, **static}
            das = {k: to_dataarray(np.where(mask, a, np.nan).astype("float32"), grid, k) for k, a in layers.items()}
            ds = xr.Dataset(das)
            ds.attrs.update(baseline=b, version=v, domain="common_mask.tif",
                            variables="bio1-bio19 + " + ", ".join(static_names))
            ds.to_netcdf(out / f"{b}_{v}_candidates.nc", encoding=nc_encoding(ds))
        print(f"  saved {len(baselines)} stacks *_candidates.nc ({19 + len(static_names)} variables each)")

        removed = iran & ~mask
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.where(iran, mask.astype(float), np.nan), cmap="Greys", vmin=0, vmax=2)
        rr, cc = np.where(removed)
        ax.scatter(cc, rr, s=2, c="red", label=f"removed ({int(removed.sum())} cells)")
        ax.scatter(c, r, s=8, c="blue", label="occurrence records")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Common modelling domain")
        fig.tight_layout()
        fig.savefig(qa / "08_fig_common_mask.png", dpi=170)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Common modelling domain")
        n.number("domain_cells", int(mask.sum()), "cells", "cells of the common modelling domain", "")
        n.number("domain_area_km2", round(a_dom), "km2", "geodesic area of the modelling domain", "")
        n.number("iran_cells_removed", int(removed.sum()), "cells",
                 "Iran cells removed because a layer lacked data (coast, lakes, salt flats)", "")
        n.number("candidate_predictors", 19 + len(static_names), "variables", "candidate predictors before screening", "D10, D42")
        n.text("Methods (draft)", f"""
All analyses used one common domain: grid cells inside Iran (GADM 4.1, cell-centre rule) with valid values in
every candidate predictor of every baseline, data version and future scenario ({int(mask.sum())} cells,
{a_dom:,.0f} km2; {int(removed.sum())} coastal, lake or salt-flat cells were excluded). This guarantees that
all models, projections and comparisons are based on exactly the same cells. Areas were computed
geodesically from cell areas on the sphere.""")
        n.save()

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails + warns:
            print(f"   - {m}")
        if not fails:
            print("\n  Step 08 completed.")


if __name__ == "__main__":
    main()
