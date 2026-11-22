"""
Step 17 - Comparison of the four baselines (main/common): equal performance, divergent predictions?
====================================================================================================

Purpose
    The core analysis of the revision (R1-1, R2): quantify how the choice of baseline period and product changes
    model performance and predictions, whether differences exceed the noise of the occurrence sample, and why.

Analyses (decision D63, fixed before this step was run)
    A  Equivalence of validation performance: paired bootstrap (records and block background resampled jointly,
       1000 replicates) of the spatial-CV validation AUC of every pair of baselines; equivalence if the 90 % CI
       of the difference lies within +/- margin (TOST; margin 0.05 AUC).
    B2 Paired replicates: replicate i of baseline A vs replicate i of baseline B (same occurrence sample if MaxEnt drew
       identical bootstrap samples; verified) against different replicates of one baseline.
    B  Divergence of predictions vs noise (inside M): Schoener's D, Spearman r and Jaccard (p10) between
       baselines for present, SSP2-4.5 and SSP5-8.5 (GCM means); the same metrics between the available bootstrap
       replicates of each baseline give the noise distribution; a between-baseline value below the 5th
       percentile of the within-baseline values is 'beyond replicate noise'.
    C  Decision-relevant divergence: direction of future change against the 2000-2024 climate (gain > 0.05,
       loss < -0.05, otherwise stable) - pairwise agreement and cells where all four baselines agree.
    D  Variance partitioning of future suitability inside M: balanced factorial period x product x GCM x SSP with
       bootstrap replicates as residual (sum of squares per cell, pooled over M); also for the suitable area.
    E  Model-based explanation: climate at calibration records under each baseline (standardised shift between periods)
       and the response curves at the warm and dry edges.
    H  Uncertainty: exploratory percentile intervals of suitable areas and period / product / interaction contrasts from 30 paired
       bootstrap replicates (identical record resamples across baselines, D67), for p10, MTP, maxSSS and a
       threshold-free suitability-weighted area (sum of suitability x cell area).   [R1-1, R1-5]
    I  Threshold robustness of the variance partitioning and of the period effect (p10, MTP, maxSSS).  [R1-5]
    F  Where: SD of future suitability among baselines vs the period and product differences of key predictors
       (Spearman correlation over the cells of M).

Outputs  data_processed/17_baseline_comparison/divergence_maps.nc
QA       results/qa/17_baseline_comparison/  17_report.txt, 17_*.csv, 17_fig_*.png
Manuscript results/manuscript/tables/T17_*.csv, notes/17_baseline_comparison.md
Run      python scripts/17_baseline_comparison.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, processed_step, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import ref_grid, to_dataarray, nc_encoding, cell_area_km2  # noqa: E402
from hemiscorpius.comparison import schoener_d, jaccard, anova_partition  # noqa: E402
from hemiscorpius.revision_checks import bootstrap_interval  # noqa: E402

STEP = "17_baseline_comparison"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
KEY_VARS = ("bio5", "bio6", "bio12", "bio15")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def fast_auc(pres, bg_sorted):
    lo = np.searchsorted(bg_sorted, pres, side="left")
    hi = np.searchsorted(bg_sorted, pres, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(bg_sorted)))


def fold_data(proc, b, tag):
    """Validation records and block-background predictions of the 4 spatial folds (step 13)."""
    rdir = proc / "13_tuning_spatial_cv" / "main" / b / "results"
    lut = pd.read_csv(proc / "13_tuning_spatial_cv" / "main" / b / "partitions_block" / "bg_groups.csv")
    grp = dict(zip(zip(lut.lon.round(5), lut.lat.round(5)), lut.group))
    folds = []
    for k in range(1, 5):
        z = np.load(rdir / f"{tag}_f{k}.npz")
        keys = list(zip(np.round(z["bg_x"].astype(float), 5), np.round(z["bg_y"].astype(float), 5)))
        # codex_20261112: same precision for both sides, as in step 13.
        bgk = {kk: round(float(p), 5) for kk, p in zip(keys, z["background"]) if grp.get(kk) == k}
        folds.append((np.round(z["test"].astype(float), 5), bgk))
    return folds


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import xarray as xr
    from scipy.stats import spearmanr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = processed_step(cfg, STEP)
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "17_run_info.json", CONFIG_FILE)
    cc = cfg.get("comparison", {})
    margin = float(cc.get("equivalence_margin_auc", 0.05))
    n_boot = int(cc.get("n_bootstrap", 1000))
    change_eps = float(cc.get("change_threshold", 0.05))
    rng = np.random.default_rng(int(cfg["project"]["seed"]) + 17)
    gcms, ssps = cfg["future"]["gcms"], cfg["future"]["ssps"]
    with rasterio.open(proc / "08_common_mask" / "common_mask.tif") as r_:
        mask = r_.read(1) == 1
    with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
        m_full = r_.read(1) == 1
    mi = np.where(m_full & mask)
    area = cell_area_km2(grid)[mi]
    sel = pd.read_csv(proc / "13_tuning_spatial_cv" / "selected_settings.csv")
    s0 = sel[(sel.variant == "main") & (sel["mode"] == "common")].iloc[0]
    tag = f"{s0.fc}_rm{float(s0.beta):g}"
    summ14 = pd.read_csv(proc / "14_final_models" / "final_models_summary.csv")
    s14m = summ14[(summ14.variant == "main") & (summ14["mode"] == "common")].set_index("baseline")
    thr = s14m.thr_p10
    thr_all = {k: s14m[f"thr_{k}"] for k in ("p10", "mtp", "maxsss")}

    def load_full(b):
        with xr.open_dataset(proc / "15_projections" / "main" / "common" / f"{b}.nc") as ds:
            return {k: ds[k].values[mi] for k in ds.data_vars}

    def load_boot(b):
        with xr.open_dataset(proc / "15_projections" / "main" / "common" / f"{b}_bootstrap.nc") as ds:
            return {k: ds[k].values[:, mi[0], mi[1]] for k in ds.data_vars}

    full = {b: load_full(b) for b in BASELINES}
    boot = {b: load_boot(b) for b in BASELINES}
    for b in BASELINES:
        for ssp in ssps:
            boot[b][f"ens_{ssp}"] = np.mean([boot[b][f"fut_present_{g}_{ssp}"] for g in gcms], axis=0)
            full[b][f"ens_{ssp}"] = full[b][f"ens_present_{ssp}_mean"]

    with Report(qa / "17_report.txt"):
        print(f"Step {STEP}: common setting {tag}; cells in M {len(area):,}; bootstrap {n_boot}; "
              f"equivalence margin +/-{margin} AUC")

        # ------------------------------------------------------------------ A
        section("A. Equivalence of validation AUC among baselines (paired bootstrap, spatial folds)")
        fd = {b: fold_data(proc, b, tag) for b in BASELINES}
        common_keys = []
        for k in range(4):
            ks = set.intersection(*[set(fd[b][k][1]) for b in BASELINES])
            common_keys.append(sorted(ks))
        n_test = [len(fd[BASELINES[0]][k][0]) for k in range(4)]
        if any(len(fd[b][k][0]) != n_test[k] for b in BASELINES for k in range(4)):
            note("FAIL", "validation records differ among baselines")
            return
        bgarr = {b: [np.array([fd[b][k][1][kk] for kk in common_keys[k]]) for k in range(4)] for b in BASELINES}
        est = {b: np.mean([fast_auc(fd[b][k][0], np.sort(bgarr[b][k])) for k in range(4)]) for b in BASELINES}
        boots = {b: np.empty(n_boot) for b in BASELINES}
        for i in range(n_boot):
            ti = [rng.integers(0, n_test[k], n_test[k]) for k in range(4)]
            bi = [rng.integers(0, len(common_keys[k]), len(common_keys[k])) for k in range(4)]
            for b in BASELINES:
                v = []
                for k in range(4):
                    v.append(fast_auc(fd[b][k][0][ti[k]], np.sort(bgarr[b][k][bi[k]])))
                boots[b][i] = np.mean(v)
        eq_rows = []
        for a, b in itertools.combinations(BASELINES, 2):
            d = boots[a] - boots[b]
            lo90, hi90 = np.percentile(d, [5, 95])
            lo95, hi95 = np.percentile(d, [2.5, 97.5])
            eq_rows.append(dict(pair=f"{a} - {b}", diff=est[a] - est[b], ci90_lo=lo90, ci90_hi=hi90, ci95_lo=lo95,
                                ci95_hi=hi95, equivalent=bool(lo90 > -margin and hi90 < margin),
                                different=bool(lo95 > 0 or hi95 < 0)))
        eq = pd.DataFrame(eq_rows)
        eq.to_csv(qa / "17_A_equivalence_auc.csv", index=False)
        eq.round(4).to_csv(tables / "T17_A_equivalence_auc.csv", index=False)
        print("   validation AUC (mean of four blocks with matched background): " + ", ".join(f"{b} {est[b]:.3f}" for b in BASELINES))
        print("   " + eq.round(3).to_string(index=False).replace("\n", "\n   "))
        note("OK", f"{int(eq.equivalent.sum())} of {len(eq)} pairs equivalent within +/-{margin} AUC; "
                   f"{int(eq.different.sum())} pairs significantly different")

        # ------------------------------------------------------------------ B
        section("B. Divergence among baselines vs replicate noise (inside M)")
        rows, noise = [], []
        for t in ("present", "ens_ssp245", "ens_ssp585"):
            for b in BASELINES:
                reps = boot[b][t]
                for i, j in itertools.combinations(range(reps.shape[0]), 2):
                    noise.append(dict(target=t, baseline=b, D=schoener_d(reps[i], reps[j]),
                                      spearman=spearmanr(reps[i], reps[j]).correlation,
                                      jaccard=jaccard(reps[i] >= thr[b], reps[j] >= thr[b])))
            for a, b in itertools.combinations(BASELINES, 2):
                rows.append(dict(target=t, pair=f"{a} - {b}", D=schoener_d(full[a][t], full[b][t]),
                                 spearman=spearmanr(full[a][t], full[b][t]).correlation,
                                 jaccard=jaccard(full[a][t] >= thr[a], full[b][t] >= thr[b])))
        nz = pd.DataFrame(noise)
        dv = pd.DataFrame(rows)

        # B2: paired comparison - replicate i of baseline A vs replicate i of baseline B (same bootstrap sample of
        # records if MaxEnt drew the same resample for every baseline; checked below), against the noise between
        # different replicates of the same baseline. Isolates the climate effect with the occurrence sample fixed.
        idx_files = [proc / "14_final_models" / "main" / "common" / b / "paired_bootstrap" / "indices.csv" for b in BASELINES]
        paired_ok = all(f.exists() for f in idx_files) and all(
            pd.read_csv(f).equals(pd.read_csv(idx_files[0])) for f in idx_files)
        note("OK" if paired_ok else "WARN",
             "paired bootstrap: identical record resamples across baselines" if paired_ok else
             "paired bootstrap indices missing or different - paired comparison approximate")
        prow = []
        for t in ("present", "ens_ssp245", "ens_ssp585"):
            within = nz[nz.target == t].D.values
            for a, b in itertools.combinations(BASELINES, 2):
                dp = [schoener_d(boot[a][t][i], boot[b][t][i]) for i in range(boot[a][t].shape[0])]
                jp = [jaccard(boot[a][t][i] >= thr[a], boot[b][t][i] >= thr[b]) for i in range(boot[a][t].shape[0])]
                prow.append(dict(target=t, pair=f"{a} - {b}", D_paired_mean=float(np.mean(dp)),
                                 D_within_mean=float(within.mean()), jaccard_paired_mean=float(np.mean(jp)),
                                 jaccard_within_mean=float(nz[nz.target == t].jaccard.mean()),
                                 share_paired_D_below_within_median=float(np.mean(np.array(dp) < np.median(within)))))
        pdv = pd.DataFrame(prow)
        pdv.to_csv(qa / "17_B2_paired_replicates.csv", index=False)
        pdv.round(3).to_csv(tables / "T17_B2_paired_replicates.csv", index=False)
        print("\n   B2. paired replicates (same occurrence sample, different baseline) vs different samples (same baseline):")
        print("   " + pdv.round(3).to_string(index=False).replace("\n", "\n   "))
        q = nz.groupby("target")[["D", "spearman", "jaccard"]].quantile(0.05)
        for m_ in ("D", "spearman", "jaccard"):
            dv[f"{m_}_noise_p05"] = dv.target.map(q[m_])
            dv[f"{m_}_beyond_noise"] = dv[m_] < dv[f"{m_}_noise_p05"]
        dv.to_csv(qa / "17_B_divergence_vs_noise.csv", index=False)
        dv.round(3).to_csv(tables / "T17_B_divergence_vs_noise.csv", index=False)
        nz.to_csv(qa / "17_B_replicate_noise.csv", index=False)
        print("   noise (5th percentile of replicate pairs): "
              + "; ".join(f"{t}: D {q.loc[t, 'D']:.3f}, rho {q.loc[t, 'spearman']:.3f}, J {q.loc[t, 'jaccard']:.3f}"
                          for t in q.index))
        print("   " + dv[["target", "pair", "D", "spearman", "jaccard", "D_beyond_noise", "jaccard_beyond_noise"]]
              .round(3).to_string(index=False).replace("\n", "\n   "))

        # ------------------------------------------------------------------ C
        section("C. Direction of future change against the 2000-2024 climate (inside M)")
        cls, crow = {}, []
        for ssp in ssps:
            for b in BASELINES:
                ch = full[b][f"change_{ssp}_vs_2000_2024"]
                cls[(ssp, b)] = np.where(ch > change_eps, 1, np.where(ch < -change_eps, -1, 0))
                crow.append(dict(ssp=ssp, baseline=b, gain_km2=float(area[cls[(ssp, b)] == 1].sum()),
                                 loss_km2=float(area[cls[(ssp, b)] == -1].sum()),
                                 stable_km2=float(area[cls[(ssp, b)] == 0].sum())))
            for a, b in itertools.combinations(BASELINES, 2):
                agree = cls[(ssp, a)] == cls[(ssp, b)]
                crow.append(dict(ssp=ssp, baseline=f"{a} vs {b}", agreement_share=float(area[agree].sum() / area.sum()),
                                 opposite_share=float(area[(cls[(ssp, a)] * cls[(ssp, b)]) == -1].sum() / area.sum())))
            allag = np.all([cls[(ssp, b)] == cls[(ssp, BASELINES[0])] for b in BASELINES], axis=0)
            crow.append(dict(ssp=ssp, baseline="all four agree", agreement_share=float(area[allag].sum() / area.sum())))
        cd = pd.DataFrame(crow)
        cd.to_csv(qa / "17_C_change_direction.csv", index=False)
        cd.round(3).to_csv(tables / "T17_C_change_direction.csv", index=False)
        print("   " + cd.round(3).to_string(index=False).replace("\n", "\n   "))

        # ------------------------------------------------------------------ D
        section("D. Variance partitioning of future suitability (period x product x GCM x SSP, replicates)")
        nrep = boot[BASELINES[0]]["present"].shape[0]
        Y = np.empty((len(area), 2, 2, len(gcms), len(ssps), nrep), "float32")
        A = np.empty((1, 2, 2, len(gcms), len(ssps), nrep))
        for b in BASELINES:
            prod, per = b.split("_", 1)
            ip, ir = (0 if per == "1970_1999" else 1), (0 if prod == "WC" else 1)
            for gi, g in enumerate(gcms):
                for si, ssp in enumerate(ssps):
                    arr = boot[b][f"fut_present_{g}_{ssp}"]                     # (rep, cells)
                    Y[:, ip, ir, gi, si, :] = arr.T
                    A[0, ip, ir, gi, si, :] = [(area * (arr[r] >= thr[b])).sum() for r in range(nrep)]
        vp = anova_partition(Y, ["period", "product", "GCM", "SSP"])
        va = anova_partition(A, ["period", "product", "GCM", "SSP"])
        vpart = pd.DataFrame({"suitability (cells of M)": vp, "suitable area (p10)": va})
        vpart = vpart.sort_values("suitability (cells of M)", ascending=False)
        choice = vpart.drop(index="replicate (residual)")
        choice = choice / choice.sum()
        grp_ = {"baseline choice (period, product, period:product)": ["period", "product", "period:product"],
                "climate projection (GCM, SSP, GCM:SSP)": ["GCM", "SSP", "GCM:SSP"]}
        gsum = pd.DataFrame({k: choice.reindex(v).sum() for k, v in grp_.items()}).T
        gsum.loc["other interactions"] = 1 - gsum.sum()
        gsum.to_csv(qa / "17_D_choice_groups.csv")
        gsum.round(4).to_csv(tables / "T17_D_choice_groups.csv")
        vpart.to_csv(qa / "17_D_variance_partitioning.csv")
        vpart.round(4).to_csv(tables / "T17_D_variance_partitioning.csv")
        print("   share of total sum of squares:")
        print("   " + (100 * vpart).round(1).to_string().replace("\n", "\n   "))
        print("\n   share among ANALYST CHOICES only (replicate residual excluded):")
        print("   " + (100 * gsum).round(1).to_string().replace("\n", "\n   "))
        Yp = np.empty((len(area), 2, 2, nrep), "float32")
        for b in BASELINES:
            prod, per = b.split("_", 1)
            Yp[:, 0 if per == "1970_1999" else 1, 0 if prod == "WC" else 1, :] = boot[b]["present"].T
        vpp = anova_partition(Yp, ["period", "product"])
        print("   present suitability: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in vpp.items()))
        pd.Series(vpp).to_csv(qa / "17_D_variance_partitioning_present.csv")

        # ------------------------------------------------------------------ E
        section("E. Mechanism: climate at the calibration records and response at the warm/dry edges")
        mech = []
        for prod in ("WC", "TC"):
            s_old = pd.read_csv(proc / "12_maxent_inputs" / "main" / f"{prod}_1970_1999" / "samples.csv")
            s_new = pd.read_csv(proc / "12_maxent_inputs" / "main" / f"{prod}_2000_2024" / "samples.csv")
            bg_new = pd.read_csv(proc / "12_maxent_inputs" / "main" / f"{prod}_2000_2024" / "background.csv")
            for v in KEY_VARS:
                if v in s_old:
                    d = s_new[v].values - s_old[v].values
                    mech.append(dict(product=prod, variable=v, records_mean_1970_1999=s_old[v].mean(),
                                     records_mean_2000_2024=s_new[v].mean(), mean_shift=d.mean(),
                                     shift_in_background_sd=d.mean() / bg_new[v].std(),
                                     records_outside_1970_1999_record_range=int(
                                         ((s_new[v] < s_old[v].min()) | (s_new[v] > s_old[v].max())).sum())))
        me = pd.DataFrame(mech)
        me.to_csv(qa / "17_E_record_climate_shift.csv", index=False)
        me.round(3).to_csv(tables / "T17_E_record_climate_shift.csv", index=False)
        print("   climate at the 38 calibration records, 2000-2024 minus 1970-1999 (same records):")
        print("   " + me.round(2).to_string(index=False).replace("\n", "\n   "))
        rc = pd.read_csv(proc / "14_final_models" / "response_curves.csv")
        rc = rc[(rc.variant == "main") & (rc["mode"] == "common") & (rc.source == "full") & (rc.kind == "marginal")]
        probes = {"bio5": [38.0, 40.0, 42.0], "bio12": [100.0, 150.0, 200.0], "bio6": [5.0, 8.0, 11.0]}
        edge = []
        for v, xs in probes.items():
            for b in BASELINES:
                c = rc[(rc.baseline == b) & (rc.variable == v)].sort_values("x")
                if len(c):
                    for x in xs:
                        edge.append(dict(variable=v, x=x, baseline=b, suitability=float(np.interp(x, c.x, c.y))))
        ed = pd.DataFrame(edge).pivot_table(index=["variable", "x"], columns="baseline", values="suitability")
        ed = ed[[b for b in BASELINES if b in ed.columns]]
        ed.to_csv(qa / "17_E_response_at_edges.csv")
        ed.round(3).to_csv(tables / "T17_E_response_at_edges.csv")
        print("\n   marginal response at the warm / dry / cold edges:")
        print("   " + ed.round(3).to_string().replace("\n", "\n   "))

        # ------------------------------------------------------------------ F
        section("F. Where do the baselines diverge, and why? (SSP5-8.5, inside M)")
        sd585 = np.std([full[b]["ens_ssp585"] for b in BASELINES], axis=0, ddof=1)
        sdpres = np.std([full[b]["present"] for b in BASELINES], axis=0, ddof=1)
        drivers = {}
        cand = {}
        for b in BASELINES:
            with xr.open_dataset(proc / "08_common_mask" / f"{b}_original_candidates.nc") as ds:
                cand[b] = {v: ds[v].values[mi] for v in KEY_VARS}
        for v in KEY_VARS:
            drivers[f"|period change| {v} (mean of products)"] = 0.5 * (
                np.abs(cand["WC_2000_2024"][v] - cand["WC_1970_1999"][v]) + np.abs(cand["TC_2000_2024"][v] - cand["TC_1970_1999"][v]))
            drivers[f"|product difference| {v} (mean of periods)"] = 0.5 * (
                np.abs(cand["TC_1970_1999"][v] - cand["WC_1970_1999"][v]) + np.abs(cand["TC_2000_2024"][v] - cand["WC_2000_2024"][v]))
        fr = pd.DataFrame([dict(driver=k, rho_with_SD_ssp585=spearmanr(x, sd585).correlation,
                                rho_with_SD_present=spearmanr(x, sdpres).correlation) for k, x in drivers.items()])
        fr = fr.sort_values("rho_with_SD_ssp585", ascending=False)
        fr.to_csv(qa / "17_F_divergence_drivers.csv", index=False)
        fr.round(3).to_csv(tables / "T17_F_divergence_drivers.csv", index=False)
        print("   " + fr.round(3).to_string(index=False).replace("\n", "\n   "))

        section("H. Uncertainty of suitable areas and of period / product effects (paired bootstrap replicates)")
        nrep_ = boot[BASELINES[0]]["present"].shape[0]
        confidence = float(cc.get("area_confidence_level", 0.95))
        min_inferential = int(cc.get("minimum_inferential_replicates", 200))
        interval_status = "exploratory_small_bootstrap_ensemble" if nrep_ < min_inferential else "percentile_bootstrap"
        if not paired_ok:
            raise RuntimeError("Contrasts require verified paired bootstrap samples")
        if nrep_ < min_inferential:
            note("WARN", f"Only {nrep_} fitted bootstrap replicates: percentile bounds are exploratory; "
                 "do not describe interval exclusion as a definitive significance test.")
        targets_h = ("present", "ens_ssp245", "ens_ssp585")

        def rep_areas(tname, key):
            return {b: np.array([(area * (boot[b][tname][r] >= thr_all[key][b])).sum() for r in range(nrep_)])
                    for b in BASELINES}
        def rep_weighted(tname):
            """Threshold-free 'suitability-weighted area' = sum(suitability x cell area) (km2)."""
            return {b: np.array([(area * boot[b][tname][r]).sum() for r in range(nrep_)]) for b in BASELINES}
        hrows, crows, saved_reps = [], [], []
        for key in ("p10", "mtp", "maxsss", "threshold_free"):
            for tname in targets_h:
                ar = rep_weighted(tname) if key == "threshold_free" else rep_areas(tname, key)
                for b in BASELINES:
                    x = ar[b]
                    lo, hi, se = bootstrap_interval(x, confidence)
                    full_area = float((area * full[b][tname]).sum()) if key == "threshold_free" else float(
                        (area * (full[b][tname] >= thr_all[key][b])).sum())
                    hrows.append(dict(threshold=key, target=tname, baseline=b, area_full_model_km2=full_area,
                                      area_mean_km2=x.mean(), bootstrap_se_km2=se, ci95_lo=lo, ci95_hi=hi,
                                      n=nrep_, interval_method="percentile", inference_status=interval_status,
                                      threshold_policy="full_model_fixed", reference="own_calibration_climate" if tname == "present" else "common_future_climate"))
                    saved_reps.extend(dict(threshold=key, target=tname, baseline=b, replicate=i, area_km2=float(val))
                                      for i, val in enumerate(x))
                contr = {"period effect (2000-2024 minus 1970-1999, mean of products)":
                         0.5 * ((ar["WC_2000_2024"] - ar["WC_1970_1999"]) + (ar["TC_2000_2024"] - ar["TC_1970_1999"])),
                         "product effect (TC minus WC, mean of periods)":
                         0.5 * ((ar["TC_1970_1999"] - ar["WC_1970_1999"]) + (ar["TC_2000_2024"] - ar["WC_2000_2024"])),
                         "interaction (period effect TC minus period effect WC)":
                         (ar["TC_2000_2024"] - ar["TC_1970_1999"]) - (ar["WC_2000_2024"] - ar["WC_1970_1999"])}
                for a, b in itertools.combinations(BASELINES, 2):
                    contr[f"{b} minus {a}"] = ar[b] - ar[a]
                for cname, d in contr.items():
                    lo, hi, se = bootstrap_interval(d, confidence)
                    crows.append(dict(threshold=key, target=tname, contrast=cname, mean_km2=d.mean(),
                                      bootstrap_se_km2=se, ci95_lo=lo, ci95_hi=hi,
                                      replicates_positive=int((d > 0).sum()), n=nrep_,
                                      interval_excludes_zero=bool(lo > 0 or hi < 0),
                                      interval_method="percentile", inference_status=interval_status,
                                      threshold_policy="full_model_fixed"))
        ha, hc = pd.DataFrame(hrows), pd.DataFrame(crows)
        pd.DataFrame(saved_reps).to_csv(qa / "17_H_replicate_areas.csv", index=False)
        ha.to_csv(qa / "17_H_area_ci.csv", index=False)
        hc.to_csv(qa / "17_H_contrasts_ci.csv", index=False)
        ha.round(0).to_csv(tables / "T17_H_area_ci.csv", index=False)
        hc.round(1).to_csv(tables / "T17_H_contrasts_ci.csv", index=False)
        print(f"   paired bootstrap replicates (n = {nrep_}; same occurrence sample across baselines: {paired_ok}); "
              f"95 % percentile bounds of replicate estimates; {interval_status}; thresholds of the full models")
        main_c = hc[hc.threshold.isin(["p10", "threshold_free"]) & hc.contrast.str.contains("effect|interaction")]
        print("   " + main_c[["threshold", "target", "contrast", "mean_km2", "ci95_lo", "ci95_hi", "replicates_positive", "interval_excludes_zero"]]
              .round(0).to_string(index=False).replace("\n", "\n   "))

        section("I. Robustness of the main results to the threshold (p10, MTP, maxSSS)  [R1-5]")
        irows = []
        for key in ("p10", "mtp", "maxsss"):
            A_ = np.empty((1, 2, 2, len(gcms), len(ssps), nrep))
            for b in BASELINES:
                prod, per = b.split("_", 1)
                ip, ir = (0 if per == "1970_1999" else 1), (0 if prod == "WC" else 1)
                for gi, g in enumerate(gcms):
                    for si, ssp in enumerate(ssps):
                        arr = boot[b][f"fut_present_{g}_{ssp}"]
                        A_[0, ip, ir, gi, si, :] = [(area * (arr[r] >= thr_all[key][b])).sum() for r in range(nrep)]
            vk = anova_partition(A_, ["period", "product", "GCM", "SSP"])
            ch = {k_: v for k_, v in vk.items() if k_ != "replicate (residual)"}
            tot_c = sum(ch.values())
            base_c = sum(ch.get(k_, 0) for k_ in ("period", "product", "period:product")) / tot_c if tot_c else np.nan
            proj_c = sum(ch.get(k_, 0) for k_ in ("GCM", "SSP", "GCM:SSP")) / tot_c if tot_c else np.nan
            pe = hc[(hc.threshold == key) & (hc.target == "ens_ssp585") & hc.contrast.str.startswith("period effect")].iloc[0]
            irows.append(dict(threshold=key, period_share_total=vk.get("period", np.nan), GCM_share_total=vk.get("GCM", np.nan),
                              SSP_share_total=vk.get("SSP", np.nan), baseline_choice_share_of_choices=base_c,
                              projection_share_of_choices=proj_c, period_effect_ssp585_km2=pe.mean_km2,
                              period_interval_excludes_zero=pe.interval_excludes_zero, inference_status=interval_status))
        # Scalar suitability-weighted area must use its own SS, not cellwise suitability SS.
        Atf = np.empty((1, 2, 2, len(gcms), len(ssps), nrep))
        for b in BASELINES:
            prod, per = b.split("_", 1)
            ip, ir = (0 if per == "1970_1999" else 1), (0 if prod == "WC" else 1)
            for gi, g in enumerate(gcms):
                for si, ssp in enumerate(ssps):
                    Atf[0, ip, ir, gi, si, :] = boot[b][f"fut_present_{g}_{ssp}"] @ area
        vtf = anova_partition(Atf, ["period", "product", "GCM", "SSP"])
        chs = {k_: v for k_, v in vtf.items() if k_ != "replicate (residual)"}
        tot_s = sum(chs.values())
        pe_tf = hc[(hc.threshold == "threshold_free") & (hc.target == "ens_ssp585") & hc.contrast.str.startswith("period effect")].iloc[0]
        irows.append(dict(threshold="threshold_free (suitability-weighted area)", period_share_total=vtf.get("period", np.nan),
                          GCM_share_total=vtf.get("GCM", np.nan), SSP_share_total=vtf.get("SSP", np.nan),
                          baseline_choice_share_of_choices=sum(chs.get(k_, 0) for k_ in ("period", "product", "period:product")) / tot_s,
                          projection_share_of_choices=sum(chs.get(k_, 0) for k_ in ("GCM", "SSP", "GCM:SSP")) / tot_s,
                          period_effect_ssp585_km2=pe_tf.mean_km2, period_interval_excludes_zero=pe_tf.interval_excludes_zero,
                          inference_status=interval_status))
        ir_ = pd.DataFrame(irows)
        ir_.to_csv(qa / "17_I_threshold_robustness.csv", index=False)
        ir_.round(3).to_csv(tables / "T17_I_threshold_robustness.csv", index=False)
        print("   " + ir_.round(3).to_string(index=False).replace("\n", "\n   "))
        consistent = bool((ir_.baseline_choice_share_of_choices > ir_.projection_share_of_choices).all())
        note("OK" if consistent else "WARN",
             "baseline choice explains more variance of suitable area than GCM/SSP under all three thresholds"
             if consistent else "the ranking of baseline choice vs GCM/SSP depends on the threshold")

        section("G. Key messages (automatic summary for the authors)")
        a_ssp = rs_area = None
        print(f"   1. Performance: {int(eq.equivalent.sum())}/{len(eq)} pairs equivalent within +/-{margin} AUC "
              f"(validation AUC {min(est.values()):.3f}-{max(est.values()):.3f}).")
        print(f"   2. Continuous patterns: Schoener's D between baselines {dv.D.min():.2f}-{dv.D.max():.2f}; "
              f"paired-replicate D {pdv.D_paired_mean.min():.2f}-{pdv.D_paired_mean.max():.2f} vs "
              f"{pdv.D_within_mean.mean():.2f} between occurrence samples.")
        g585 = cd[(cd.ssp == "ssp585") & cd.gain_km2.notna()]
        print(f"   3. Area with continuous suitability increase > {change_eps} (SSP5-8.5) {g585.gain_km2.min():,.0f}-{g585.gain_km2.max():,.0f} km2 "
              f"({g585.gain_km2.max() / max(g585.gain_km2.min(), 1):.1f}-fold); opposite directions "
              f"{cd.opposite_share.max():.1%} of M; all four agree in "
              f"{cd[(cd.ssp == 'ssp585') & (cd.baseline == 'all four agree')].agreement_share.item():.0%} of M.")
        print(f"   4. Variance among analyst choices, suitable area: baseline choice "
              f"{gsum.loc[list(grp_)[0], 'suitable area (p10)']:.0%} vs climate projection "
              f"{gsum.loc[list(grp_)[1], 'suitable area (p10)']:.0%}.")
        wc12 = me[(me["product"] == "WC") & (me.variable == "bio12")].iloc[0]
        print(f"   5. Model-based explanation (not physiological validation): same records are {abs(wc12.mean_shift):.0f} mm drier in WC 2000-2024 than in WC 1970-1999 "
              f"({wc12.shift_in_background_sd:.2f} SD); 1970-1999 models assign lower suitability at the dry/hot edges.")

        # ------------------------------------------------------------------ maps + figure
        allag = {ssp: np.all([cls[(ssp, b)] == cls[(ssp, BASELINES[0])] for b in BASELINES], axis=0) for ssp in ssps}
        das = []
        for name, arr in (("sd_among_baselines_present", sdpres), ("sd_among_baselines_ssp585", sd585),
                          ("all_baselines_agree_direction_ssp585", allag["ssp585"].astype(float)),
                          ("all_baselines_agree_direction_ssp245", allag["ssp245"].astype(float))):
            g_ = np.full(grid.shape, np.nan, "float32")
            g_[mi] = arr
            das.append(to_dataarray(g_, grid, name))
        dsm = xr.merge(das)
        dsm.to_netcdf(out / "divergence_maps.nc", encoding=nc_encoding(dsm))

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        ax = axes[0, 0]
        y_ = np.arange(len(eq))
        ax.errorbar(eq["diff"], y_, xerr=[eq["diff"] - eq.ci90_lo, eq.ci90_hi - eq["diff"]], fmt="o", color="k")
        ax.axvspan(-margin, margin, color="lightgreen", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(y_, eq.pair, fontsize=7)
        ax.set_title("A. validation AUC difference, 90% CI (green = equivalence margin)", fontsize=9)
        ax = axes[0, 1]
        for i, t in enumerate(("present", "ens_ssp245", "ens_ssp585")):
            ax.boxplot(nz[nz.target == t].D, positions=[i * 2], widths=0.6)
            ax.scatter(np.full(6, i * 2 + 0.8) + np.linspace(-0.2, 0.2, 6), dv[dv.target == t].D, c="red", s=12)
        ax.set_xticks([0.4, 2.4, 4.4], ["present", "SSP2-4.5", "SSP5-8.5"])
        ax.set_ylabel("Schoener's D")
        ax.set_title("B. D between replicates (box) vs between baselines (red)", fontsize=9)
        ax = axes[1, 0]
        top = vpart.head(8)
        ax.barh(top.index[::-1], 100 * top["suitability (cells of M)"][::-1], color="#0072B2", label="suitability")
        ax.barh(top.index[::-1], 100 * top["suitable area (p10)"][::-1], color="#E69F00", alpha=0.6, label="area")
        ax.set_xlabel("% of total sum of squares")
        ax.legend(fontsize=7)
        ax.set_title("D. variance partitioning of future suitability", fontsize=9)
        ax = axes[1, 1]
        g_ = np.full(grid.shape, np.nan)
        g_[mi] = sd585
        im = ax.imshow(g_, cmap="magma", extent=(grid.west, grid.west + grid.ncols / 24,
                                                 grid.north - grid.nrows / 24, grid.north))
        ax.set_xlim(50.5, 61)
        ax.set_ylim(24.8, 30.5)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title("F. SD of SSP5-8.5 suitability among the four baselines (inside M)", fontsize=9)
        fig.tight_layout()
        fig.savefig(qa / "17_fig_comparison_main.png", dpi=170)
        plt.close(fig)

        # ------------------------------------------------------------------ notes
        n = ManuscriptNotes(cfg, STEP, "Comparison of baselines")
        for _, r in eq.iterrows():
            n.number(f"auc_diff_{r.pair}", round(r["diff"], 4), "AUC", "validation AUC difference", "R1-1")
        for k_, v in vp.items():
            n.number(f"varpart_suitability_{k_}", round(v, 4), "share", "share of total SS (future suitability)", "R2 novelty")
        for k_, v in va.items():
            n.number(f"varpart_area_{k_}", round(v, 4), "share", "share of total SS (future suitable area)", "R2 novelty")
        n.text("Methods (draft)", f"""
Validation performance of the four baselines was compared with a paired bootstrap ({n_boot} replicates) of the spatial
cross-validation AUC, resampling validation records and block background jointly; baselines were considered equivalent
if the 90 % confidence interval of the difference lay within +/-{margin} AUC (two one-sided tests). Agreement between
predictions inside the calibration area was measured with Schoener's D, Spearman correlation and the Jaccard index of
suitable cells (10th-percentile threshold) and compared with the same metrics among the {nrep_} paired bootstrap replicates of each
baseline (sampling noise). Directions of future change (gain/loss > {change_eps}) were compared among baselines. The variance
of future suitability was partitioned among baseline period, baseline product, GCM, SSP and their interactions in a balanced
factorial design, with bootstrap replicates as residual (cf. Baker et al. 2016). Finally, the shift of the climate attributed
to the same records between periods and the response curves at the warm and dry edges were examined to identify the
possible statistical explanation of divergence. These are predictive associations, not causal physiological tests.
Area uncertainty used the {100*(1-confidence)/2:g}th and {100*(1-(1-confidence)/2):g}th percentiles of the paired replicate estimates,
with full-model thresholds held fixed. With {nrep_} fitted replicates these bounds are exploratory, conditional on the
selected configuration, fixed background and thresholds; no definitive significance claim is made from them.
The target named 'present' is each model's own calibration climate, not the common C2000 reference of Table T19.
The uncertainty of a scalar suitability-weighted area is computed from sums over cells, separately from cellwise variation.""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 17 completed.")


if __name__ == "__main__":
    main()
