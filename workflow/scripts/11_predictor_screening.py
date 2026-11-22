"""
Step 11 - Predictor screening
=============================

Purpose
    Select ONE common set of predictors for all climatic baselines (reviewer 1, minor comment 4: clarify
    and justify the predictor selection).

Procedure (decision D49, fixed before any model was fitted)
    1. A-priori exclusion of quarter-based variables bio8, bio9, bio18, bio19 (D28, D38, D43; Booth 2022).
    2. Spearman correlations computed on the background cells of the MAIN accessible area for each of the
       four baselines (original data); the largest |r| over baselines is used. Walking the pre-specified
       ecological priority list, a variable is kept only if |r| <= threshold (0.7; Dormann et al. 2013)
       with every variable kept before it.
    3. VIF computed for each baseline; while the largest VIF over baselines exceeds 10, the
       lowest-priority variable among those with VIF > 10 is removed.
    4. At most 'max_predictors' (8) variables are kept, in priority order (~5 calibration records per
       predictor with 38 records).
    Sensitivity set: the predictors of the original manuscript (bio3, bio4, bio8, bio12, clay, slope);
    its correlations and VIF are reported.

Outputs  data_processed/11_predictor_screening/selected_predictors.csv  (main set, in priority order)
QA       results/qa/11_predictor_screening/  11_report.txt, 11_*.csv, 11_fig_*.png
Manuscript results/manuscript/tables/T11_*.csv, notes/11_predictor_screening.md
Run      python scripts/11_predictor_screening.py
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
from hemiscorpius.screening import vif, greedy_by_priority  # noqa: E402

STEP = "11_predictor_screening"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from scipy.stats import spearmanr

    cfg = load_config()
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "11_run_info.json", CONFIG_FILE)
    sc = cfg["predictors"]["screening"]
    thr, vif_max, cap = float(sc["spearman_abs_r"]), float(sc["vif_max"]), int(sc["max_predictors"])
    excluded = list(sc["a_priori_excluded"])
    priority = list(sc["priority_list"])
    original_set = list(cfg["predictors"]["previous_final_set"])

    with Report(qa / "11_report.txt"):
        print(f"Step {STEP}: |r| threshold {thr}, VIF max {vif_max}, max predictors {cap}")
        cand = cfg["predictors"]["candidates"]
        all_vars = cand["bioclim"] + cand["terrain"] + cand["soil"]
        pool = [v for v in all_vars if v not in excluded]
        missing = sorted(set(pool) - set(priority))
        extra = sorted(set(priority) - set(pool))
        note("OK" if not missing and not extra else "FAIL",
             f"priority list covers the {len(pool)} candidates (missing {missing or 'none'}; unknown {extra or 'none'})")
        if missing or extra:
            return
        print(f"  a-priori excluded: {excluded}")

        n_cal = len(pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_thin10km.csv"))
        bg = pd.read_csv(proc / "10_accessible_area_background" / "background_main.csv")
        r_i, c_i = bg["row"].values, bg["col"].values
        data = {}
        for b in BASELINES:
            with xr.open_dataset(proc / "08_common_mask" / f"{b}_original_candidates.nc") as ds:
                data[b] = pd.DataFrame({v: ds[v].values[r_i, c_i] for v in all_vars}).dropna()
        print(f"  background cells in the main M: {len(bg)} (complete rows per baseline: "
              f"{', '.join(str(len(d)) for d in data.values())})")

        section("1. Spearman correlations (maximum |r| over the four baselines)")
        mats = {b: pd.DataFrame(spearmanr(d[all_vars]).correlation, index=all_vars, columns=all_vars)
                for b, d in data.items()}
        maxabs = pd.concat([m.abs() for m in mats.values()]).groupby(level=0).max().loc[all_vars, all_vars]
        maxabs.to_csv(qa / "11_spearman_max_abs_over_baselines.csv")
        lookup = {(a, b_): maxabs.loc[a, b_] for a in all_vars for b_ in all_vars}
        kept, log = greedy_by_priority(priority, lookup, thr)
        logdf = pd.DataFrame(log)
        logdf.insert(0, "priority_rank", np.arange(1, len(logdf) + 1))
        print("   " + logdf.to_string(index=False).replace("\n", "\n   "))
        print(f"\n  kept after Spearman: {kept}")

        section("2. Variance inflation factors (maximum over baselines)")
        vif_log = []
        while True:
            v_tab = pd.DataFrame({b: vif(d[kept].values) for b, d in data.items()}, index=kept)
            vmax = v_tab.max(axis=1)
            vif_log.append(vmax.rename(f"iteration_{len(vif_log) + 1}"))
            if vmax.max() <= vif_max or len(kept) <= 1:
                break
            worst = [v for v in kept if vmax[v] > vif_max]
            drop = max(worst, key=priority.index)
            print(f"  VIF > {vif_max:g}: {', '.join(f'{v} {vmax[v]:.1f}' for v in worst)} -> remove {drop}")
            kept.remove(drop)
            logdf.loc[logdf.variable == drop, ["decision", "reason"]] = ["removed", f"VIF {vmax[drop]:.1f}"]
        print("   " + vmax.round(2).to_string().replace("\n", "\n   "))
        pd.concat(vif_log, axis=1).to_csv(qa / "11_vif_iterations.csv")

        section("3. Cap on the number of predictors")
        if len(kept) > cap:
            dropped = kept[cap:]
            kept = kept[:cap]
            for v in dropped:
                logdf.loc[logdf.variable == v, ["decision", "reason"]] = ["removed", f"beyond the cap of {cap}"]
            print(f"  {len(kept) + len(dropped)} variables passed; lowest-priority removed: {dropped}")
        else:
            print(f"  {len(kept)} variables passed (cap {cap} not reached)")
        note("OK", f"MAIN predictor set ({len(kept)}): {kept}")
        final_vif = pd.DataFrame({b: vif(d[kept].values) for b, d in data.items()}, index=kept)
        final_r = maxabs.loc[kept, kept]
        r_off = final_r.where(~np.eye(len(kept), dtype=bool)).max().max() if len(kept) > 1 else 0.0
        note("OK" if r_off <= thr and final_vif.values.max() <= vif_max else "FAIL",
             f"final set: max |r| {r_off:.2f}, max VIF {final_vif.values.max():.2f}")

        section("4. Sensitivity set: predictors of the original manuscript")
        o_r = maxabs.loc[original_set, original_set]
        o_vif = pd.DataFrame({b: vif(d[original_set].values) for b, d in data.items()}, index=original_set)
        o_off = o_r.where(~np.eye(len(original_set), dtype=bool)).max().max()
        print(f"  {original_set}: max |r| {o_off:.2f}; max VIF {o_vif.values.max():.2f}")
        print("   " + o_r.round(2).to_string().replace("\n", "\n   "))

        # ------------------------------------------------------------------ outputs
        sel = pd.DataFrame({"predictor": kept, "priority_rank": [priority.index(v) + 1 for v in kept],
                            "max_vif_over_baselines": final_vif.max(axis=1).round(2).values})
        sel.to_csv(out / "selected_predictors.csv", index=False)
        pd.DataFrame({"predictor": original_set}).to_csv(out / "sensitivity_original_predictors.csv", index=False)
        logdf.to_csv(qa / "11_screening_log.csv", index=False)
        logdf.to_csv(tables / "T11_screening_log.csv", index=False)
        sel.to_csv(tables / "T11_selected_predictors.csv", index=False)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [2.2, 1]})
        order = [v for v in priority]
        im = axes[0].imshow(maxabs.loc[order, order].values, cmap="Reds", vmin=0, vmax=1)
        axes[0].set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        axes[0].set_yticks(range(len(order)), order, fontsize=7)
        axes[0].set_title(f"max |Spearman r| over baselines (main M); candidates in priority order")
        fig.colorbar(im, ax=axes[0], shrink=0.7)
        im2 = axes[1].imshow(final_r.values, cmap="Reds", vmin=0, vmax=1)
        axes[1].set_xticks(range(len(kept)), kept, rotation=90)
        axes[1].set_yticks(range(len(kept)), kept)
        for i in range(len(kept)):
            for j in range(len(kept)):
                axes[1].text(j, i, f"{final_r.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        axes[1].set_title("selected predictors")
        fig.tight_layout()
        fig.savefig(qa / "11_fig_correlations.png", dpi=170)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Predictor screening")
        n.number("n_candidates", len(all_vars), "variables", "candidate predictors", "R1 minor 4")
        n.number("n_after_a_priori_exclusion", len(pool), "variables", "after excluding bio8/9/18/19", "D28")
        n.number("spearman_threshold", thr, "", "|Spearman r| threshold", "R1 minor 4")
        n.number("vif_threshold", vif_max, "", "VIF threshold", "R1 minor 4")
        n.number("n_selected", len(kept), "variables", "selected predictors", "R1 minor 4")
        n.number("selected_predictors", ", ".join(kept), "", "main predictor set", "R1 minor 4")
        n.number("final_max_abs_r", round(float(r_off), 3), "", "largest |r| in the selected set", "")
        n.number("final_max_vif", round(float(final_vif.values.max()), 2), "", "largest VIF in the selected set", "")
        n.number("original_set_max_abs_r", round(float(o_off), 3), "", "largest |r| in the original set", "")
        n.text("Methods (draft)", f"""
Predictors were screened once for all baselines. Quarter-based variables (bio8, bio9, bio18, bio19) were excluded
a priori because their quarter assignment changed between products and periods and they showed artificial
discontinuities (Booth 2022), and because both products overestimate summer precipitation in the species'
range (Section X). For the remaining {len(pool)} candidates, Spearman correlations were computed on the background
cells of the main calibration area for each of the four baselines, and the largest absolute value was used.
Following a priority list defined before any model was fitted - proximal physiological limits (minimum
winter and maximum summer temperature, annual precipitation) and habitat structure (slope, coarse fragments)
first, distal or derived variables (elevation, summary bioclimatic variables) last (Austin 2002) - a variable was
retained only if |r| <= {thr} with all higher-priority retained variables (Dormann et al. 2013). Variables with
VIF > {vif_max:g} in any baseline were then removed in reverse priority order, and at most {cap} predictors were kept
given {n_cal} calibration records. The selected set was {', '.join(kept)} (largest |r| = {r_off:.2f}, largest VIF =
{final_vif.values.max():.1f}). The predictor set of the original analysis ({', '.join(original_set)}) was used as a
sensitivity analysis.""")
        n.save()

        section("SUMMARY")
        fails = [m for lvl, m in ISSUES if lvl == "FAIL"]
        warns = [m for lvl, m in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for m in fails + warns:
            print(f"   - {m}")
        if not fails:
            print("\n  Step 11 completed.")


if __name__ == "__main__":
    main()
