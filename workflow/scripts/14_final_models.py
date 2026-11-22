"""
Step 14 - Final models
======================

Purpose
    Fit the final MaxEnt model of every baseline (and of every sensitivity variant) with the setting
    selected in step 13, and describe it: variable importance, response curves, thresholds, and a
    consistency check with the records removed by thinning (reviewer 1 comments 5 and 7).

Method (decision D54, fixed before the final models were fitted)
    settings    MAIN: the common setting of the four baselines (D58); the baseline-specific settings are fitted
                for the main variant as a sensitivity analysis.
    full model  all calibration records, selected feature classes and regularisation; response curves
                (marginal and single-variable) with plot data, jackknife of regularised training gain,
                percent contribution and permutation importance; cloglog output.
    replicates  'n_replicates' (30) PAIRED bootstrap replicates (D67): the same resampled record indices for all four
                baselines (removeduplicates=false), replacing MaxEnt's internal bootstrap (its samples differed among
                baselines). Former note: (MaxEnt replicatetype=bootstrap, fixed default
                seed) for the MAIN variant: uncertainty of response curves, importance and (step 15)
                predictions; used in step 17 to compare baseline differences with replicate noise.
    thresholds  10th-percentile training presence (p10), minimum training presence (MTP) and maximum
                training sensitivity plus specificity (maxSSS), all on the cloglog scale.
    consistency the known localities NOT used for calibration (9 records for the 10-km set) are passed
                as test records; the share predicted at or above each threshold is reported. They lie
                within 10 km of calibration records and are therefore not an independent validation (D46).
    checks      warning if a final model has fewer than 2 parameters; response-curve figure for an
                ecological plausibility check by the authors.

Outputs  data_processed/14_final_models/{variant}/{baseline}/full/      MaxEnt outputs (lambdas, results)
         data_processed/14_final_models/{variant}/{baseline}/bootstrap/ (main variant)
         data_processed/14_final_models/final_models_summary.csv, importance.csv, response_curves.csv
QA       results/qa/14_final_models/  14_report.txt, 14_*.csv, 14_fig_*.png
Manuscript results/manuscript/tables/T14_*.csv, notes/14_final_models.md
Run      python scripts/14_final_models.py           main variant (full + bootstrap)
         python scripts/14_final_models.py --all     adds full models of the sensitivity variants
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.maxent import (run_maxent, read_predictions, n_parameters, read_results,  # noqa: E402
                                 results_value, read_plot_data, write_swd, resolve_java, java_version, SPECIES)
from hemiscorpius.evaluation import auc, tss_at  # noqa: E402

STEP = "14_final_models"
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


def consistency_file(cfg, proc, variant, baseline, preds, dst):
    """SWD file with the known localities not used for calibration in this variant."""
    import xarray as xr
    spec = variant_spec(cfg, variant)
    allr = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_all.csv")
    used = pd.read_csv(proc / "09_occurrences_thinning" / f"occurrences_{spec['occurrences']}.csv")
    rest = allr[~allr["id"].isin(used["id"])]
    if rest.empty:
        return None, 0
    with xr.open_dataset(proc / "08_common_mask" / f"{baseline}_{spec['version']}_candidates.nc") as ds:
        vals = pd.DataFrame({v: ds[v].values[rest["row"].values, rest["col"].values] for v in preds})
    write_swd(dst, SPECIES, rest.lon.values, rest.lat.values, vals)
    return dst, len(rest)


def job(task):
    """Fit one final model (full or bootstrap) and return a summary dict."""
    out = Path(task["out"])
    run_maxent(task["jar"], out, task["samples"], task["background"], task["fc"], task["beta"],
               test=task.get("test"), output="cloglog", java=task["java"], memory_mb=task["memory"],
               analysis=True, replicates=task["replicates"], jackknife=task["jackknife"], extra=task.get("extra"))
    return task["key"]


def col(res, name):
    """Exact column lookup (case-insensitive) - avoids bio1 matching bio12."""
    for k, v in res.items():
        if str(k).strip().lower() == name.lower():
            return v
    return np.nan


def summarise_full(out, preds, n_cons):
    res, _ = read_results(out)
    p = read_predictions(out, "cloglog")
    thr = {"p10": results_value(res, "10 percentile training presence", "cloglog threshold"),
           "mtp": results_value(res, "minimum training presence", "cloglog threshold"),
           "maxsss": results_value(res, "maximum training sensitivity plus specificity", "cloglog threshold")}
    s = dict(n_params=n_parameters(out), auc_train=auc(p["train"], p["background"]),
             auc_train_maxent=results_value(res, "training auc"), **{f"thr_{k}": v for k, v in thr.items()})
    for k, t in thr.items():
        s[f"tss_train_{k}"] = tss_at(t, p["train"], p["background"])
        s[f"consistency_share_{k}"] = float(np.mean(p["test"] >= t)) if n_cons else np.nan
    s["n_consistency"] = n_cons
    imp = []
    for v in preds:
        imp.append(dict(variable=v, contribution=col(res, f"{v} contribution"),
                        permutation_importance=col(res, f"{v} permutation importance"),
                        jackknife_gain_only=col(res, f"Training gain with only {v}"),
                        jackknife_gain_without=col(res, f"Training gain without {v}")))
    return s, pd.DataFrame(imp)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    proc = project_path(cfg, "processed")
    out_root = proc / STEP
    out_root.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "14_run_info.json", CONFIG_FILE)
    m = cfg["models"]
    n_rep, n_jobs = int(m.get("n_replicates", 10)), int(m.get("n_jobs", 4))
    run_all = "--all" in sys.argv
    sel = pd.read_csv(proc / "13_tuning_spatial_cv" / "selected_settings.csv")
    if "mode" not in sel:
        sel["mode"] = "specific"
    variants = [v for v in m["variants"] if run_all or v == "main"]
    # MAIN analysis = common setting (D58); baseline-specific settings only for the main variant (sensitivity)
    sel = sel[sel.variant.isin(variants) & ((sel["mode"] == "common") | (sel.variant == "main"))]

    with Report(qa / "14_report.txt"):
        print(f"Step {STEP}: variants {variants}; bootstrap replicates {n_rep} (main)")
        java = resolve_java(m.get("java", "java"))
        print(f"  Java: {java} ({java_version(java)})")
        jar = str(ROOT / cfg["paths"]["maxent_jar"])

        section("1. Fitting final models")
        tasks, meta = [], {}
        n_rec_main = len(pd.read_csv(proc / "12_maxent_inputs" / "main" / BASELINES[0] / "samples.csv"))
        rng_b = np.random.default_rng(int(cfg["project"]["seed"]) + 14)
        boot_idx = [rng_b.choice(n_rec_main, n_rec_main, replace=True) for _ in range(n_rep)]
        for _, s in sel.iterrows():
            v, b, mode = s.variant, s.baseline, s["mode"]
            spec = variant_spec(cfg, v)
            src = proc / "12_maxent_inputs" / v / b
            preds = [c for c in pd.read_csv(src / "samples.csv", nrows=1).columns if c not in ("species", "lon", "lat")]
            d = out_root / v / mode / b
            d.mkdir(parents=True, exist_ok=True)
            cons, n_cons = consistency_file(cfg, proc, v, b, preds, d / "consistency_records.csv")
            meta[(v, mode, b)] = dict(preds=preds, n_cons=n_cons, fc=s.fc, beta=float(s.beta), spec=spec)
            common = dict(jar=jar, java=java, memory=int(m.get("memory_mb", 1024)), fc=s.fc, beta=float(s.beta),
                          samples=str(src / "samples.csv"), background=str(src / "background.csv"))
            if not (d / "full" / "maxentResults.csv").exists():
                tasks.append(dict(common, key=(v, mode, b, "full"), out=str(d / "full"), replicates=1, jackknife=True,
                                  test=str(cons) if cons else None))
            if v == "main" and mode == "common":
                # PAIRED bootstrap (D67): the same resampled record indices for every baseline, so that replicate i
                # of two baselines differs only in climate; duplicates kept (removeduplicates=false)
                pb = d / "paired_bootstrap"
                pb.mkdir(parents=True, exist_ok=True)
                smp = pd.read_csv(src / "samples.csv")
                for i in range(n_rep):
                    rd = pb / f"rep_{i:02d}"
                    rd.mkdir(parents=True, exist_ok=True)
                    smp.iloc[boot_idx[i]].to_csv(rd / "samples.csv", index=False)
                    if not (rd / "model" / "maxentResults.csv").exists():
                        tasks.append(dict(common, key=(v, mode, b, f"boot{i:02d}"), out=str(rd / "model"),
                                          samples=str(rd / "samples.csv"), replicates=1, jackknife=False,
                                          extra=["removeduplicates=false"]))
                pd.DataFrame(np.array(boot_idx)).to_csv(pb / "indices.csv", index=False)
        print(f"  {len(tasks)} MaxEnt jobs to run ({len(sel)} variant x baseline models)")
        if tasks:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futs = {ex.submit(job, t): t["key"] for t in tasks}
                for i, f in enumerate(as_completed(futs), 1):
                    try:
                        f.result()
                        print(f"    {i}/{len(tasks)} done: {'/'.join(map(str, futs[f]))}")
                    except Exception as e:
                        note("FAIL", f"{'/'.join(map(str, futs[f]))}: {str(e)[:200]}")

        section("2. Summaries: parameters, thresholds, importance, consistency records")
        rows, imps, curves = [], [], []
        for (v, mode, b), mt in meta.items():
            d = out_root / v / mode / b
            if not (d / "full" / "maxentResults.csv").exists():
                continue
            s, imp = summarise_full(d / "full", mt["preds"], mt["n_cons"])
            rows.append(dict(variant=v, mode=mode, baseline=b, fc=mt["fc"], beta=mt["beta"], **s))
            imps.append(imp.assign(variant=v, mode=mode, baseline=b, source="full"))
            pc = read_plot_data(d / "full", mt["preds"]).assign(variant=v, mode=mode, baseline=b, source="full")
            curves.append(pc)
            if s["n_params"] < 2:
                note("WARN", f"{v}/{mode}/{b}: final model has only {s['n_params']} parameter(s) - check response curves")
            if v == "main" and mode == "common":
                for i in range(n_rep):
                    md = d / "paired_bootstrap" / f"rep_{i:02d}" / "model"
                    if not (md / "maxentResults.csv").exists():
                        continue
                    pr = read_plot_data(md, mt["preds"])
                    if len(pr):
                        curves.append(pr.assign(variant=v, mode=mode, baseline=b, source=f"boot{i}"))
                    rec, _ = read_results(md)
                    imps.append(pd.DataFrame([dict(variable=x, contribution=col(rec, f"{x} contribution"),
                                                   permutation_importance=col(rec, f"{x} permutation importance"))
                                              for x in mt["preds"]]).assign(variant=v, mode=mode, baseline=b,
                                                                            source=f"boot{i}"))
        summ = pd.DataFrame(rows)
        imp_all = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()
        cur_all = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
        summ.to_csv(out_root / "final_models_summary.csv", index=False)
        imp_all.to_csv(out_root / "importance.csv", index=False)
        cur_all.to_csv(out_root / "response_curves.csv", index=False)
        summ.round(4).to_csv(tables / "T14_final_models.csv", index=False)
        main_s = summ[(summ.variant == "main") & (summ["mode"] == "common")]
        print("  MAIN (common setting):")
        print("   " + main_s[["baseline", "fc", "beta", "n_params", "auc_train", "thr_p10", "thr_mtp", "thr_maxsss",
                               "consistency_share_p10", "consistency_share_maxsss"]].round(3)
              .to_string(index=False).replace("\n", "\n   "))

        spec_s = summ[(summ.variant == "main") & (summ["mode"] == "specific")]
        if len(spec_s):
            print("  sensitivity (baseline-specific settings):")
            print("   " + spec_s[["baseline", "fc", "beta", "n_params", "auc_train", "thr_p10", "consistency_share_p10"]]
                  .round(3).to_string(index=False).replace("\n", "\n   "))
        imp_main = imp_all[(imp_all.variant == "main") & (imp_all["mode"] == "common")] if len(imp_all) else imp_all
        if len(imp_main):
            # codex_20261112: the full fit is not an additional bootstrap draw.
            from hemiscorpius.revision_checks import importance_summary
            agg = importance_summary(imp_main)
            agg.to_csv(qa / "14_importance_main.csv", index=False)
            agg.round(2).to_csv(tables / "T14_variable_importance.csv", index=False)
            top = (imp_main[imp_main.source == "full"].sort_values("permutation_importance", ascending=False)
                   .groupby("baseline").head(3)[["baseline", "variable", "permutation_importance", "contribution"]])
            print("\n  three most important predictors per baseline (permutation importance, full model):")
            print("   " + top.round(1).to_string(index=False).replace("\n", "\n   "))

        section("3. Response-curve figure (for the ecological plausibility check)")
        if len(cur_all):
            preds = meta[("main", "common", BASELINES[0])]["preds"]
            fig, axes = plt.subplots(2, int(np.ceil(len(preds) / 2)), figsize=(18, 7.5))
            colours = dict(zip(BASELINES, ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]))
            for ax, var in zip(axes.ravel(), preds):
                for b in BASELINES:
                    c = cur_all[(cur_all.variant == "main") & (cur_all["mode"] == "common") & (cur_all.baseline == b)
                                & (cur_all.variable == var)
                                & (cur_all.kind == "marginal")]
                    boot = c[c.source.str.startswith("boot")]
                    if len(boot):
                        bb = boot.assign(i=boot.groupby("source").cumcount()).groupby("i")
                        x, lo, hi = bb.x.mean(), bb.y.quantile(0.05), bb.y.quantile(0.95)
                        ax.fill_between(x, lo, hi, color=colours[b], alpha=0.15, lw=0)
                    f = c[c.source == "full"]
                    ax.plot(f.x, f.y, color=colours[b], lw=1.2, label=b)
                ax.set_title(var)
                ax.set_ylim(0, 1)
            axes.ravel()[0].legend(fontsize=7)
            for ax in axes.ravel()[len(preds):]:
                ax.axis("off")
            fig.suptitle("Marginal response curves (cloglog), main models with the COMMON setting; "
                         "bands = 5-95 % of bootstrap replicates")
            fig.tight_layout()
            fig.savefig(qa / "14_fig_response_curves_main.png", dpi=170)
            plt.close(fig)
            print("  saved 14_fig_response_curves_main.png - check that the responses are ecologically plausible")

        n = ManuscriptNotes(cfg, STEP, "Final models")
        for _, r in main_s.iterrows():
            k = r.baseline
            n.number(f"final_{k}_setting", f"{r.fc} rm{r.beta:g}", "", "final setting", "R1-4")
            n.number(f"final_{k}_n_params", int(r.n_params), "parameters", "non-zero parameters", "")
            for t in ("p10", "mtp", "maxsss"):
                n.number(f"final_{k}_thr_{t}", round(float(r[f"thr_{t}"]), 4), "cloglog", f"{t} threshold", "R1-5")
                n.number(f"final_{k}_consistency_{t}", round(float(r[f"consistency_share_{t}"]), 3), "fraction",
                         f"share of non-calibration records predicted >= {t}", "D46")
        n.text("Methods (draft)", f"""
Final models were fitted for each baseline with all calibration records and the selected setting, using cloglog
output. Variable importance was assessed with percent contribution, permutation importance and a jackknife of the
regularised training gain. Uncertainty was described with {n_rep} bootstrap replicates, from which 5-95 % bands of
the response curves were derived. Binary maps were produced with three thresholds (10th-percentile training
presence, minimum training presence and maximum training sensitivity plus specificity) to assess threshold
dependence. The known localities not used for calibration were used only to check that they fell in areas
predicted suitable, because they lie within 10 km of calibration records.""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 14 completed.")


if __name__ == "__main__":
    main()
