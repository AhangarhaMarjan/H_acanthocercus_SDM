"""
Step 18 - Robustness of the key messages across all sensitivity variants and tuning modes
=========================================================================================

Purpose
    Test whether the main conclusions of step 17 hold in every sensitivity analysis requested or implied by the
    reviewers: accessible area (R1-2), thinning distance (R1 minor 3), predictor set (R1 minor 4, R1-7),
    cleaned precipitation, MaxEnt tuning mode (R1-4) and - in the Supplement - sampling-bias correction.

Key messages tested (decision D68), each per variant x mode, using the full models of steps 14-16:
    M1 equal performance   range of spatial-CV validation AUC among the four baselines <= 0.05
    M2 period effect       future (SSP5-8.5, GCM mean) suitable area inside the main M is larger with the 2000-2024
                           baselines (mean of products), for p10 and for the threshold-free suitability-weighted area
    M3 baseline > GCM/SSP  among analyst choices, baseline choice (period, product, period:product) explains more
                           of the variance of future suitable area (p10) than GCM, SSP and GCM:SSP
    M4 extrapolation       the 1970-1999 models extrapolate in a larger share of M under SSP5-8.5 than the 2000-2024
                           models (MESS < 0)
    M5 mechanism           at bio12 = 150 mm the 1970-1999 models assign lower suitability than the 2000-2024 models
                           (only where bio12 is a predictor)
    Areas are computed inside the MAIN accessible area for comparability, and inside each variant's own M.

Outputs  results/qa/18_sensitivity_summary/  18_report.txt, 18_robustness_table.csv, 18_fig_robustness.png
Manuscript results/manuscript/tables/T18_robustness.csv, notes/18_sensitivity_summary.md
Run      python scripts/18_sensitivity_summary.py
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
from hemiscorpius.grid import ref_grid, cell_area_km2  # noqa: E402
from hemiscorpius.comparison import anova_partition  # noqa: E402

STEP = "18_sensitivity_summary"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def period_effect(a):
    return 0.5 * ((a["WC_2000_2024"] - a["WC_1970_1999"]) + (a["TC_2000_2024"] - a["TC_1970_1999"]))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "18_run_info.json", CONFIG_FILE)
    gcms, ssps = cfg["future"]["gcms"], cfg["future"]["ssps"]
    area_full = cell_area_km2(grid)

    def mask_of(name):
        with rasterio.open(proc / "10_accessible_area_background" / f"M_{name}.tif") as r_:
            return r_.read(1) == 1
    m_main = mask_of("main")
    summ14 = pd.read_csv(proc / "14_final_models" / "final_models_summary.csv")
    sel = pd.read_csv(proc / "13_tuning_spatial_cv" / "selected_settings.csv")
    ext = pd.read_csv(project_path(cfg, "results") / "qa" / "16_extrapolation_mess" / "16_extrapolation_shares.csv")
    rc = pd.read_csv(proc / "14_final_models" / "response_curves.csv")
    rc = rc[(rc.source == "full") & (rc.kind == "marginal")]

    combos = summ14[["variant", "mode"]].drop_duplicates().values.tolist()
    with Report(qa / "18_report.txt"):
        print(f"Step {STEP}: {len(combos)} variant x mode combinations")
        rows = []
        for v, mode in combos:
            s14 = summ14[(summ14.variant == v) & (summ14["mode"] == mode)].set_index("baseline")
            if not set(BASELINES) <= set(s14.index):
                note("WARN", f"{v}/{mode}: incomplete baselines - skipped")
                continue
            spec = dict(cfg["models"]["variants"]["main"])
            spec.update(cfg["models"]["variants"].get(v) or {})
            own = spec["M"].replace("_biascorr", "")
            m_own = mask_of(own) if own != "main" else m_main
            data = {}
            for b in BASELINES:
                with xr.open_dataset(proc / "15_projections" / v / mode / f"{b}.nc") as ds:
                    data[b] = {k: ds[k].values for k in ds.data_vars
                               if k in ("present", "ens_present_ssp585_mean") or k.startswith("fut_present")}
            r = dict(variant=v, mode=mode, setting=f"{s14.fc.iloc[0]} rm{s14.beta.iloc[0]:g}")
            # M1
            sv = sel[(sel.variant == v) & (sel["mode"] == mode)].set_index("baseline")
            if set(BASELINES) <= set(sv.index):
                r["auc_val_min"], r["auc_val_max"] = sv.auc_val.min(), sv.auc_val.max()
                r["M1_equal_performance"] = bool(sv.auc_val.max() - sv.auc_val.min() <= 0.05)
            # M2
            for dom, mk in (("mainM", m_main), ("ownM", m_own)):
                a10 = {b: float(area_full[(data[b]["ens_present_ssp585_mean"] >= s14.loc[b, "thr_p10"]) & mk].sum())
                       for b in BASELINES}
                atf = {b: float(np.nansum(np.where(mk, area_full * data[b]["ens_present_ssp585_mean"], 0)))
                       for b in BASELINES}
                r[f"period_effect_p10_{dom}_km2"] = period_effect(a10)
                r[f"period_effect_tf_{dom}_km2"] = period_effect(atf)
                if dom == "mainM":
                    for b in BASELINES:
                        r[f"area_p10_ssp585_{b}"] = a10[b]
            r["M2_period_effect_positive"] = bool(r["period_effect_p10_mainM_km2"] > 0 and r["period_effect_tf_mainM_km2"] > 0)
            # M3 (full models; GCM x SSP factorial, no replicates)
            A = np.empty((1, 2, 2, len(gcms), len(ssps)))
            for b in BASELINES:
                prod, per = b.split("_", 1)
                for gi, g in enumerate(gcms):
                    for si, ssp in enumerate(ssps):
                        A[0, 0 if per == "1970_1999" else 1, 0 if prod == "WC" else 1, gi, si] = \
                            area_full[(data[b][f"fut_present_{g}_{ssp}"] >= s14.loc[b, "thr_p10"]) & m_main].sum()
            vp = anova_partition(A, ["period", "product", "GCM", "SSP"])
            base_c = sum(vp.get(k, 0) for k in ("period", "product", "period:product"))
            proj_c = sum(vp.get(k, 0) for k in ("GCM", "SSP", "GCM:SSP"))
            r["baseline_choice_share"], r["projection_share"] = base_c, proj_c
            r["M3_baseline_over_projection"] = bool(base_c > proj_c)
            # M4
            e = ext[(ext.variant == v) & (ext["mode"] == mode) & ext.target.str.startswith("fut_present") &
                    ext.target.str.endswith("ssp585")].groupby("baseline").share_M_extrapolated.mean()
            if set(BASELINES) <= set(e.index):
                r["extrap_old"] = 0.5 * (e["WC_1970_1999"] + e["TC_1970_1999"])
                r["extrap_new"] = 0.5 * (e["WC_2000_2024"] + e["TC_2000_2024"])
                r["M4_more_extrapolation_old"] = bool(r["extrap_old"] > r["extrap_new"])
            # M5
            c = rc[(rc.variant == v) & (rc["mode"] == mode) & (rc.variable == "bio12")]
            if len(c) and set(BASELINES) <= set(c.baseline):
                val = {b: float(np.interp(150.0, *c[c.baseline == b].sort_values("x")[["x", "y"]].values.T)) for b in BASELINES}
                r["resp150_old"] = 0.5 * (val["WC_1970_1999"] + val["TC_1970_1999"])
                r["resp150_new"] = 0.5 * (val["WC_2000_2024"] + val["TC_2000_2024"])
                r["M5_drier_edge_penalised_old"] = bool(r["resp150_old"] < r["resp150_new"])
            rows.append(r)
        tab = pd.DataFrame(rows)
        tab.to_csv(qa / "18_robustness_table.csv", index=False)
        tab.round(3).to_csv(tables / "T18_robustness.csv", index=False)

        section("1. Robustness table (one row per variant x tuning mode)")
        show = ["variant", "mode", "setting", "auc_val_min", "auc_val_max", "period_effect_p10_mainM_km2",
                "period_effect_tf_mainM_km2", "baseline_choice_share", "projection_share", "extrap_old", "extrap_new",
                "resp150_old", "resp150_new"]
        pd.set_option("display.width", 250)
        print("   " + tab[[c for c in show if c in tab]].round(3).to_string(index=False).replace("\n", "\n   "))

        section("2. How often does each key message hold?")
        msgs = [c for c in tab.columns if c.startswith("M") and c[1].isdigit()]
        summ = pd.DataFrame([dict(message=m_, holds=int(tab[m_].fillna(False).sum()), tested=int(tab[m_].notna().sum()))
                             for m_ in msgs])
        summ["share"] = summ.holds / summ.tested
        summ.to_csv(qa / "18_message_support.csv", index=False)
        summ.round(3).to_csv(tables / "T18_message_support.csv", index=False)
        print("   " + summ.round(2).to_string(index=False).replace("\n", "\n   "))
        for _, s_ in summ.iterrows():
            failing = tab.loc[~tab[s_.message].fillna(True).astype(bool), ["variant", "mode"]]
            if len(failing):
                print(f"   {s_.message} does not hold in: " + ", ".join(f"{a}/{b}" for a, b in failing.values))

        # ------------------------------------------------------------------ figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        lab = tab.variant + " / " + tab["mode"]
        y = np.arange(len(tab))
        axes[0].barh(y, tab.period_effect_p10_mainM_km2 / 1000, color="#0072B2", label="p10")
        axes[0].barh(y, tab.period_effect_tf_mainM_km2 / 1000, color="#E69F00", alpha=0.6, label="threshold-free")
        axes[0].axvline(0, color="k", lw=0.8)
        axes[0].set_yticks(y, lab, fontsize=7)
        axes[0].set_xlabel("period effect on SSP5-8.5 area inside M (1000 km2)")
        axes[0].legend(fontsize=7)
        axes[1].barh(y, tab.baseline_choice_share, color="#009E73", label="baseline choice")
        axes[1].barh(y, -tab.projection_share, color="#D55E00", label="GCM / SSP")
        axes[1].set_yticks(y, [""] * len(y))
        axes[1].set_xlabel("share of variance among analyst choices")
        axes[1].legend(fontsize=7)
        if "extrap_old" in tab:
            axes[2].scatter(tab.extrap_new, tab.extrap_old, c="k")
            lim = [0, max(tab.extrap_old.max(), tab.extrap_new.max()) * 1.05]
            axes[2].plot(lim, lim, color="grey", lw=0.8)
            axes[2].set_xlabel("share of M extrapolated, 2000-2024 models")
            axes[2].set_ylabel("share of M extrapolated, 1970-1999 models")
        fig.tight_layout()
        fig.savefig(qa / "18_fig_robustness.png", dpi=160)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Robustness of the key messages")
        for _, s_ in summ.iterrows():
            n.number(f"support_{s_.message}", f"{s_.holds}/{s_.tested}", "", f"variants x modes supporting {s_.message}", "all")
        n.text("Results (draft)", "The key messages were tested in every sensitivity analysis (accessible area, thinning distance, "
               "predictor set, cleaned precipitation, tuning mode; sampling-bias correction in the Supplement); the number of "
               "analyses supporting each message is given in Table S? (T18_message_support).")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 18 completed.")


if __name__ == "__main__":
    main()
