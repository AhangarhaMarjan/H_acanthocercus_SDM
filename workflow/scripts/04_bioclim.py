"""
Step 04 - Bioclimatic variables of the four baselines
=====================================================

Purpose
    Compute the 19 bioclimatic variables (bio1-bio19) for every baseline (WC/TC x 1970-1999/
    2000-2024) and version (original = main analysis; cleaned = sensitivity, D19) from the
    12-month climatologies of step 03, and quantify how they differ between baselines.

Why / method
    - dismo::biovars (R) is used, exactly as in the original manuscript (decision D05), so the
      definitions are unchanged. R is called automatically from this script.
    - The R output is verified against an independent Python implementation
      (src/hemiscorpius/bioclim.py, also covered by tests/test_bioclim.py).
    - WC_1970_1999 is compared with the official WorldClim 2.1 bioclim normals (1970-2000).
    - Quarter-based variables (bio8, bio9, bio18, bio19) can show artificial discontinuities when
      the wettest/driest quarter switches between neighbouring cells (Booth 2022). Their
      quarter assignments and spatial jumps are diagnosed here; the decision on their use is
      taken in the predictor-screening step.
    - Period effect, product effect and interaction are computed for all 19 variables, raw and
      standardised by the variable's spatial SD in Iran, to identify which predictors are most
      sensitive to the choice of climate product (reviewers R1-1, R2).

Inputs   data_processed/03_monthly_climatologies/{baseline}_{version}.nc,
         data_processed/02_crop_and_units/wc21_bio_1970_2000.nc, occurrence records
Outputs  data_processed/04_bioclim/{baseline}_{version}_bioclim.nc   (bio1-bio19)
         data_processed/04_bioclim/{baseline}_{version}_bio.tif       (19 bands, from R)
QA       results/qa/04_bioclim/  04_report.txt, 04_run_info.json, 04_*.csv, 04_fig_*.png
Manuscript results/manuscript/tables/T04_*.csv, results/manuscript/notes/04_bioclim.md
Run      python scripts/04_bioclim.py
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import CELL, ref_grid, to_dataarray, nc_encoding, write_geotiff  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402
from hemiscorpius.bioclim import biovars as py_biovars  # noqa: E402

STEP = "04_bioclim"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
VERSIONS = ("original", "cleaned")
BIO = [f"bio{i}" for i in range(1, 20)]
BIO_UNITS = {"bio1": "degC", "bio2": "degC", "bio3": "%", "bio4": "degC x 100", "bio5": "degC",
             "bio6": "degC", "bio7": "degC", "bio8": "degC", "bio9": "degC", "bio10": "degC",
             "bio11": "degC", "bio12": "mm", "bio13": "mm", "bio14": "mm", "bio15": "%",
             "bio16": "mm", "bio17": "mm", "bio18": "mm", "bio19": "mm"}
EFFECTS = ["period_WC", "period_TC", "interaction", "product_1970_1999", "product_2000_2024"]
JUMP_DEGC = 5.0          # neighbour jump in bio8/bio9 regarded as a discontinuity (QA only)
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


# --------------------------------------------------------------------------- computation
def prepare_r_inputs(src, rin, grid):
    import xarray as xr
    rin.mkdir(parents=True, exist_ok=True)
    clim = {}
    for b in BASELINES:
        for ver in VERSIONS:
            s = f"{b}_{ver}"
            with xr.open_dataset(src / f"{s}.nc") as ds:
                clim[s] = {v: ds[v].values.astype("float32") for v in ("tmin", "tmax", "prec")}
            for v, arr in clim[s].items():
                write_geotiff(rin / f"{s}_{v}.tif", arr, grid)
    (rin / "sets.txt").write_text("\n".join(clim) + "\n", encoding="utf-8")
    return clim


def run_r(cfg, rin, rout):
    rscript = cfg["paths"]["rscript"]
    script = ROOT / "scripts" / "04_bioclim_biovars.R"
    res = subprocess.run([rscript, str(script), str(rin), str(rout)], capture_output=True, text=True)
    for line in (res.stdout + res.stderr).splitlines():
        if line.strip():
            print(f"    R> {line}")
    if res.returncode != 0:
        raise RuntimeError(f"Rscript failed (exit {res.returncode})")


def read_r_output(rout, s):
    import rasterio
    with rasterio.open(rout / f"{s}_bio.tif") as r:
        a = r.read(masked=True).astype("float32").filled(np.nan)
    if a.shape[0] != 19:
        raise ValueError(f"{s}: {a.shape[0]} bands instead of 19")
    return {f"bio{i + 1}": a[i] for i in range(19)}


# --------------------------------------------------------------------------- QA
def verify_against_python(bio_r, bio_py, mask, qa):
    section("2. Verification: dismo::biovars (R) vs independent Python implementation")
    rows = []
    for s in bio_r:
        for v in BIO:
            a, b = bio_r[s][v], bio_py[s][v]
            both = mask & np.isfinite(a) & np.isfinite(b)
            nan_mismatch = int((mask & (np.isfinite(a) ^ np.isfinite(b))).sum())
            d = np.abs(a[both] - b[both])
            rng = float(np.nanmax(a[both]) - np.nanmin(a[both])) or 1.0
            rows.append(dict(set=s, variable=v, max_abs_diff=float(d.max()) if d.size else np.nan,
                             max_rel_diff=float(d.max() / rng) if d.size else np.nan,
                             cells_differing_gt_1pct=int((d > 0.01 * rng).sum()),
                             nan_mismatch=nan_mismatch))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "04_verification_r_vs_python.csv", index=False)
    worst = df.sort_values("max_rel_diff", ascending=False).head(5)
    print("  largest relative differences (all sets x 19 variables):")
    print("   " + worst.to_string(index=False).replace("\n", "\n   "))
    bad = df[(df.cells_differing_gt_1pct > 0) | (df.nan_mismatch > 0)]
    if len(bad) == 0:
        note("OK", f"R and Python agree for all {len(df)} set x variable combinations "
                   f"(max relative difference {df.max_rel_diff.max():.1e})")
    else:
        # quarter ties can differ by implementation; report rather than fail
        note("WARN", f"{len(bad)} set x variable combinations with cells differing > 1% "
                     f"(see 04_verification_r_vs_python.csv; usually quarter ties)")
    return df


def consistency_with_official(src, bio_r, mask, qa):
    import xarray as xr
    section("3. Consistency: WC_1970_1999 bioclim (built) vs official WorldClim 2.1 bioclim 1970-2000")
    with xr.open_dataset(src / "wc21_bio_1970_2000.nc") as ds:
        off = {v: ds[v].values.astype("float32") for v in BIO}
    built = bio_r["WC_1970_1999_original"]
    rows = []
    for v in BIO:
        a, b = built[v], off[v]
        ok_ = mask & np.isfinite(a) & np.isfinite(b)
        d = a[ok_] - b[ok_]
        rows.append(dict(variable=v, unit=BIO_UNITS[v], mean_diff=float(d.mean()),
                         mae=float(np.abs(d).mean()), r=float(np.corrcoef(a[ok_], b[ok_])[0, 1]),
                         official_iran_mean=float(b[ok_].mean()),
                         share_cells_abs_diff_gt_10pct_of_sd=float((np.abs(d) > 0.1 * b[ok_].std()).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "04_consistency_wc_built_vs_wc21_bio.csv", index=False)
    print("   " + df.round(3).to_string(index=False).replace("\n", "\n   "))
    low = df[df.r < 0.95]
    note("OK" if len(low) == 0 else "WARN",
         f"variables with r < 0.95 against the official WorldClim bioclim: "
         f"{list(low.variable) or 'none'} (quarter-based variables are expected to be most sensitive)")
    return df


def neighbour_jump(x):
    """Largest absolute difference between a cell and its 4 direct neighbours (no wrap-around)."""
    jump = np.zeros_like(x, dtype="float32")
    dx = np.abs(x[:, 1:] - x[:, :-1])
    dy = np.abs(x[1:, :] - x[:-1, :])
    dx[~np.isfinite(dx)] = 0
    dy[~np.isfinite(dy)] = 0
    jump[:, 1:] = np.maximum(jump[:, 1:], dx)
    jump[:, :-1] = np.maximum(jump[:, :-1], dx)
    jump[1:, :] = np.maximum(jump[1:, :], dy)
    jump[:-1, :] = np.maximum(jump[:-1, :], dy)
    return jump


def quarter_diagnostics(bio_py, bio_r, mask, occ, grid, qa, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section("4. Quarter-based variables (bio8, bio9, bio18, bio19): assignments and discontinuities")
    r, c = occ["row"].values, occ["col"].values
    near = np.zeros(mask.shape, dtype=bool)          # occurrence cells and their 8 neighbours
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = np.clip(r + dr, 0, grid.nrows - 1), np.clip(c + dc, 0, grid.ncols - 1)
            near[rr, cc] = True
    rows = []
    pairs = {"product 1970-1999 (WC vs TC)": ("WC_1970_1999", "TC_1970_1999"),
             "product 2000-2024 (WC vs TC)": ("WC_2000_2024", "TC_2000_2024"),
             "period WorldClim": ("WC_1970_1999", "WC_2000_2024"),
             "period TerraClimate": ("TC_1970_1999", "TC_2000_2024")}
    for ver in VERSIONS:
        for label, (a, b) in pairs.items():
            for q in ("_wettest_quarter_start", "_driest_quarter_start"):
                qa_, qb = bio_py[f"{a}_{ver}"][q], bio_py[f"{b}_{ver}"][q]
                diff = mask & np.isfinite(qa_) & np.isfinite(qb) & (qa_ != qb)
                rows.append(dict(version=ver, comparison=label, quarter=q.strip("_"),
                                 iran_share_changed=float(diff.sum() / mask.sum()),
                                 occurrence_records_changed=int(diff[r, c].sum())))
    for b in BASELINES:
        qa_, qb = bio_py[f"{b}_original"]["_wettest_quarter_start"], bio_py[f"{b}_cleaned"]["_wettest_quarter_start"]
        diff = mask & np.isfinite(qa_) & (qa_ != qb)
        rows.append(dict(version="original vs cleaned", comparison=b, quarter="wettest_quarter_start",
                         iran_share_changed=float(diff.sum() / mask.sum()),
                         occurrence_records_changed=int(diff[r, c].sum())))
    qdf = pd.DataFrame(rows)
    qdf.to_csv(qa / "04_quarter_assignment_changes.csv", index=False)
    print("   " + qdf.round(4).to_string(index=False).replace("\n", "\n   "))

    jrows = []
    (out / "qa_layers").mkdir(exist_ok=True)
    for s, bio in bio_r.items():
        for v in ("bio8", "bio9"):
            jump = neighbour_jump(bio[v])
            disc = mask & (jump > JUMP_DEGC)
            if v == "bio8" and s.endswith("original"):
                write_geotiff(out / "qa_layers" / f"bio8_discontinuity_{s}.tif",
                              np.where(mask, disc.astype("float32"), np.nan), grid)
            jrows.append(dict(set=s, variable=v, iran_share_discontinuous=float(disc.sum() / mask.sum()),
                              occurrence_records_at_or_next_to_discontinuity=int((disc & near)[r, c].sum()),
                              occurrence_neighbourhood_cells_discontinuous=int((disc & near).sum())))
    jdf = pd.DataFrame(jrows)
    jdf.to_csv(qa / "04_bio8_bio9_discontinuities.csv", index=False)
    print(f"\n  cells with a neighbour jump > {JUMP_DEGC} degC (Booth 2022 artefact):")
    print("   " + jdf.round(4).to_string(index=False).replace("\n", "\n   "))
    n_occ = int(jdf[jdf.set.str.endswith("original") & (jdf.variable == "bio8")]
                .occurrence_records_at_or_next_to_discontinuity.max())
    note("OK" if n_occ == 0 else "WARN",
         f"bio8: up to {n_occ} occurrence records at or next to a discontinuity (original version) "
         f"- to be considered in predictor screening")

    ext = [grid.west, grid.west + grid.ncols * CELL, grid.north - grid.nrows * CELL, grid.north]
    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    for k, b in enumerate(BASELINES):
        s = f"{b}_original"
        im = axes[0, k].imshow(np.where(mask, bio_py[s]["_wettest_quarter_start"] + 1, np.nan),
                               extent=ext, cmap="twilight", vmin=1, vmax=12)
        axes[0, k].set_title(f"{b}: first month of wettest quarter")
        fig.colorbar(im, ax=axes[0, k], shrink=0.7, ticks=range(1, 13))
        im = axes[1, k].imshow(np.where(mask, bio_r[s]["bio8"], np.nan), extent=ext, cmap="inferno")
        axes[1, k].set_title(f"{b}: bio8 (degC)")
        fig.colorbar(im, ax=axes[1, k], shrink=0.7)
    for ax in axes.ravel():
        ax.scatter(occ["lon"], occ["lat"], s=5, c="cyan", edgecolors="k", linewidths=0.3)
    fig.tight_layout()
    fig.savefig(qa / "04_fig_wettest_quarter_and_bio8.png", dpi=200)
    plt.close(fig)
    return qdf, jdf


def bioclim_effects(bio_r, mask, occ, grid, qa, manuscript_tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section("5. Period effect, product effect and interaction for bio1-bio19")
    r, c = occ["row"].values, occ["col"].values
    w = np.cos(np.deg2rad(grid.lats))[:, None] * mask
    rows = []
    for ver in VERSIONS:
        B = {b: bio_r[f"{b}_{ver}"] for b in BASELINES}
        for v in BIO:
            ref = bio_r["WC_1970_1999_original"][v]
            sd = float(np.nanstd(ref[mask]))
            maps = {"period_WC": B["WC_2000_2024"][v] - B["WC_1970_1999"][v],
                    "period_TC": B["TC_2000_2024"][v] - B["TC_1970_1999"][v],
                    "product_1970_1999": B["TC_1970_1999"][v] - B["WC_1970_1999"][v],
                    "product_2000_2024": B["TC_2000_2024"][v] - B["WC_2000_2024"][v]}
            maps["interaction"] = maps["period_TC"] - maps["period_WC"]
            for e in EFFECTS:
                m = maps[e]
                vv = np.isfinite(m) & (w > 0)
                iran = float((m * w)[vv].sum() / w[vv].sum())
                occ_vals = m[r, c]
                rows.append(dict(version=ver, variable=v, unit=BIO_UNITS[v], effect=e,
                                 iran_mean=iran, occurrence_mean=float(np.nanmean(occ_vals)),
                                 occurrence_mean_abs=float(np.nanmean(np.abs(occ_vals))),
                                 iran_sd_of_variable=sd,
                                 std_iran_mean=iran / sd if sd else np.nan,
                                 std_occurrence_mean=float(np.nanmean(occ_vals)) / sd if sd else np.nan,
                                 std_occurrence_mean_abs=float(np.nanmean(np.abs(occ_vals))) / sd if sd else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "04_bioclim_effects.csv", index=False)
    df.round(4).to_csv(manuscript_tables / "T04_bioclim_period_product_effects.csv", index=False)

    for ver in VERSIONS:
        piv = (df[df.version == ver].pivot(index="variable", columns="effect", values="std_occurrence_mean")
               .reindex(BIO)[EFFECTS])
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(EFFECTS)), EFFECTS, rotation=35, ha="right")
        ax.set_yticks(range(len(BIO)), BIO)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="effect at occurrence records / spatial SD in Iran")
        ax.set_title(f"Standardised effects at occurrence records ({ver})")
        fig.tight_layout()
        fig.savefig(qa / f"04_fig_bioclim_effects_standardised_{ver}.png", dpi=200)
        plt.close(fig)

    sub = df[(df.version == "original") & (df.effect == "product_2000_2024")]
    top = sub.reindex(sub.std_occurrence_mean_abs.sort_values(ascending=False).index).head(6)
    print("  variables most sensitive to the product (2000-2024, mean |effect| at occurrences / SD):")
    print("   " + top[["variable", "unit", "occurrence_mean", "std_occurrence_mean_abs"]]
          .round(3).to_string(index=False).replace("\n", "\n   "))
    return df


def save_bioclim(bio_r, out, grid):
    import xarray as xr
    for s, bio in bio_r.items():
        das = {v: to_dataarray(bio[v], grid, v, units=BIO_UNITS[v]) for v in BIO}
        ds = xr.Dataset(das)
        b, ver = s.rsplit("_", 1)
        ds.attrs.update(baseline=b, version=ver, method="dismo::biovars (R) on 12-month climatology",
                        units_note="bio4 = SD x 100; bio3, bio15 in %")
        ds.to_netcdf(out / f"{s}_bioclim.nc", encoding=nc_encoding(ds))


def manuscript_notes_04(cfg, ver_df, cons_df, qdf, jdf, eff_df):
    n = ManuscriptNotes(cfg, STEP, "Bioclimatic variables of the four baselines")
    n.number("verification_max_rel_diff", float(ver_df.max_rel_diff.max()), "",
             "largest relative difference dismo (R) vs independent Python implementation", "QA")
    for _, row in cons_df.iterrows():
        n.number(f"consistency_r_{row.variable}", float(row.r), "",
                 f"{row.variable}: correlation WC_1970_1999 (built) vs official WorldClim 2.1 bioclim", "QA")
    for _, row in qdf[qdf.version == "original"].iterrows():
        key = (row.comparison.split(" (")[0].replace(" ", "_").replace("-", "_") + "_" +
               row.quarter.replace("_start", ""))
        n.number(f"quarter_change_share_{key}", float(row.iran_share_changed), "fraction of Iran cells",
                 f"{row.comparison}: cells whose {row.quarter.replace('_start', '')} differs", "R1-1, Booth 2022")
        n.number(f"quarter_change_occ_{key}", int(row.occurrence_records_changed), "records",
                 f"{row.comparison}: occurrence records whose {row.quarter.replace('_start', '')} differs",
                 "R1-1, Booth 2022")
    for _, row in jdf[jdf.set.str.endswith("original")].iterrows():
        n.number(f"discontinuity_share_{row.variable}_{row.set}", float(row.iran_share_discontinuous),
                 "fraction of Iran cells", f"{row.variable} neighbour jump > {JUMP_DEGC} degC, {row.set}",
                 "Booth 2022")
    for _, row in eff_df.iterrows():
        n.number(f"bio_effect_{row.variable}_{row.effect}_{row.version}_occ", round(row.occurrence_mean, 4),
                 row.unit, f"{row.variable} {row.effect} - mean at occurrence records ({row.version})", "R1-1, R2")
        n.number(f"bio_effect_std_{row.variable}_{row.effect}_{row.version}_occ",
                 round(row.std_occurrence_mean, 4), "SD units",
                 f"{row.variable} {row.effect} - standardised by Iran SD ({row.version})", "R1-1, R2")
    n.text("Methods (draft)", f"""
The 19 bioclimatic variables were computed from each 12-month climatology with dismo::biovars
(R {cfg['software']['r']}, dismo {cfg['software']['dismo']}), the function used in the original analysis.
The R output was verified against an independent implementation of the same definitions
(maximum relative difference {ver_df.max_rel_diff.max():.1e}), and the WorldClim 1970-1999 bioclimatic
layers were compared with the official WorldClim 2.1 bioclimatic normals. Because quarter-based
variables (bio8, bio9, bio18, bio19) can contain artificial discontinuities where the wettest or driest
quarter changes between neighbouring cells (Booth 2022), quarter assignments and neighbour jumps
> {JUMP_DEGC:.0f} degC were quantified for every baseline. Period effects, product effects and their
interaction were computed for all variables and standardised by each variable's spatial standard
deviation across Iran.""")
    n.text("Results (draft placeholder)", """
Report: agreement with official WorldClim bioclim; variables most sensitive to product choice
(standardised product effect 2000-2024 at occurrence records); frequency of wettest-quarter changes
between products/periods and of bio8 discontinuities near occurrence records. Numbers: key_numbers.csv.""")
    n.save()


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    src3, src2 = proc / "03_monthly_climatologies", proc / "02_crop_and_units"
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    manuscript_tables = project_path(cfg, "results") / "manuscript" / "tables"
    manuscript_tables.mkdir(parents=True, exist_ok=True)
    write_run_info(qa / "04_run_info.json", CONFIG_FILE)

    with Report(qa / "04_report.txt"):
        print(f"Step {STEP}")
        import rasterio
        with rasterio.open(src2 / "reference_grid.tif") as r_:
            mask = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)

        section("1. dismo::biovars for 4 baselines x 2 versions")
        clim = prepare_r_inputs(src3, out / "r_inputs", grid)
        try:
            run_r(cfg, out / "r_inputs", out)
        except Exception as e:
            note("FAIL", f"R step failed: {e}")
            section("SUMMARY")
            print(f"  FAIL: 1 - {e}")
            return
        bio_r = {s: read_r_output(out, s) for s in clim}
        bio_py = {s: py_biovars(clim[s]["prec"], clim[s]["tmin"], clim[s]["tmax"]) for s in clim}
        save_bioclim(bio_r, out, grid)
        print(f"  saved {len(bio_r)} files *_bioclim.nc (bio1-bio19)")

        results = {}
        for name, fn, args in (
                ("ver", verify_against_python, (bio_r, bio_py, mask, qa)),
                ("cons", consistency_with_official, (src2, bio_r, mask, qa)),
                ("quarters", quarter_diagnostics, (bio_py, bio_r, mask, occ, grid, qa, out)),
                ("eff", bioclim_effects, (bio_r, mask, occ, grid, qa, manuscript_tables))):
            try:
                results[name] = fn(*args)
            except Exception as e:
                note("FAIL", f"{fn.__name__} crashed: {type(e).__name__}: {e}")
        try:
            qdf, jdf = results["quarters"]
            manuscript_notes_04(cfg, results["ver"], results["cons"], qdf, jdf, results["eff"])
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
            print("\n  Step 04 completed. Review figures and tables in results/qa/04_bioclim/.")


if __name__ == "__main__":
    main()
