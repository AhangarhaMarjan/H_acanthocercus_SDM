"""
Step 12 - MaxEnt inputs (samples-with-data) for every model variant
===================================================================

Purpose
    Write the presence and background files (SWD format) for the four baselines of the main analysis
    and of every sensitivity variant, so that steps 13-14 read nothing else.

Variants (config models.variants; each changes ONE element of the main design)
    main                   38 records (10 km thinning), main M, selected predictors, original data
    M_ecoregion / M_buffer100km / M_national             accessible-area sensitivity (R1-2)
    occ_thin05km / occ_thin20km / occ_onepercell          thinning sensitivity (R1 minor 3)
    predictors_original    predictor set of the original manuscript (incl. bio8)
    climate_cleaned        cleaned precipitation (D19)

Checks
    No missing predictor values; the selected predictors also satisfy |r| <= 0.7 and VIF <= 10 in the
    cleaned data version (robustness of the screening).

Outputs  data_processed/12_maxent_inputs/{variant}/{baseline}/samples.csv, background.csv
QA       results/qa/12_maxent_inputs/  12_report.txt, 12_variants.csv
Run      python scripts/12_maxent_inputs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.maxent import write_swd, SPECIES  # noqa: E402
from hemiscorpius.screening import vif  # noqa: E402

STEP = "12_maxent_inputs"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def variant_spec(cfg, name):
    v = dict(cfg["models"]["variants"]["main"])
    v.update(cfg["models"]["variants"].get(name) or {})
    return v


def main():
    import xarray as xr
    from scipy.stats import spearmanr

    cfg = load_config()
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    write_run_info(qa / "12_run_info.json", CONFIG_FILE)
    sel = pd.read_csv(proc / "11_predictor_screening" / "selected_predictors.csv")["predictor"].tolist()
    orig = pd.read_csv(proc / "11_predictor_screening" / "sensitivity_original_predictors.csv")["predictor"].tolist()
    pred_sets = {"selected": sel, "original": orig}

    with Report(qa / "12_report.txt"):
        print(f"Step {STEP}\n  selected predictors: {sel}\n  original predictors: {orig}")
        cache = {}

        def stack(baseline, version):
            key = (baseline, version)
            if key not in cache:
                with xr.open_dataset(proc / "08_common_mask" / f"{baseline}_{version}_candidates.nc") as ds:
                    cache[key] = {v: ds[v].values for v in ds.data_vars}
            return cache[key]

        section("1. Writing SWD files")
        rows = []
        for name in cfg["models"]["variants"]:
            spec = variant_spec(cfg, name)
            occ = pd.read_csv(proc / "09_occurrences_thinning" / f"occurrences_{spec['occurrences']}.csv")
            bg_file = proc / "10_accessible_area_background" / f"background_{spec['M']}.csv"
            if not bg_file.exists():
                note("WARN", f"{name}: {bg_file.name} not found (run the step that creates it) - variant skipped")
                continue
            bg = pd.read_csv(bg_file)
            preds = pred_sets[spec["predictors"]]
            for b in BASELINES:
                st = stack(b, spec["version"])
                d = out / name / b
                d.mkdir(parents=True, exist_ok=True)
                pv = pd.DataFrame({v: st[v][occ["row"].values, occ["col"].values] for v in preds})
                bv = pd.DataFrame({v: st[v][bg["row"].values, bg["col"].values] for v in preds})
                n_nan = int(pv.isna().sum().sum() + bv.isna().sum().sum())
                if n_nan:
                    note("FAIL", f"{name}/{b}: {n_nan} missing predictor values")
                write_swd(d / "samples.csv", SPECIES, occ.lon.values, occ.lat.values, pv)
                write_swd(d / "background.csv", "background", bg.lon.values, bg.lat.values, bv)
                occ[["id", "lon", "lat", "row", "col"]].to_csv(d / "samples_index.csv", index=False)
                bg.to_csv(d / "background_index.csv", index=False)
            rows.append(dict(variant=name, occurrences=spec["occurrences"], n_records=len(occ), M=spec["M"],
                             n_background=len(bg), predictors=",".join(preds), version=spec["version"]))
            print(f"  {name:22s} records {len(occ):2d} | background {len(bg):5d} | {spec['predictors']:8s} | "
                  f"{spec['version']}")
        pd.DataFrame(rows).to_csv(qa / "12_variants.csv", index=False)
        note("OK", f"{len(rows)} variants x {len(BASELINES)} baselines written")

        section("2. Robustness of the screening in the cleaned data version")
        bg = pd.read_csv(proc / "10_accessible_area_background" / "background_main.csv")
        worst_r, worst_v = 0.0, 0.0
        for b in BASELINES:
            st = stack(b, "cleaned")
            X = pd.DataFrame({v: st[v][bg["row"].values, bg["col"].values] for v in sel}).dropna()
            r = np.abs(spearmanr(X.values).correlation)
            np.fill_diagonal(r, 0)
            worst_r, worst_v = max(worst_r, r.max()), max(worst_v, vif(X.values).max())
        thr, vmax = cfg["predictors"]["screening"]["spearman_abs_r"], cfg["predictors"]["screening"]["vif_max"]
        note("OK" if worst_r <= thr and worst_v <= vmax else "WARN",
             f"cleaned version: max |r| {worst_r:.2f} (<= {thr}), max VIF {worst_v:.2f} (<= {vmax})")

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails + warns:
            print(f"   - {m}")
        if not fails:
            print("\n  Step 12 completed.")


if __name__ == "__main__":
    main()
