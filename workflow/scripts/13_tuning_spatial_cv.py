"""
Step 13 - MaxEnt tuning with spatial cross-validation, explicit selection rule and null models
=============================================================================================

Purpose
    For every model variant and baseline, evaluate 6 feature-class combinations x 11 regularisation
    multipliers (66 settings) and select one setting with an explicitly documented rule
    (reviewer 1 comments 3-4; reviewer 2, line 234).

Method (initial decision D51, revised in D53/D55/D58/D59 after tuning diagnostics;
        do not describe the final rule as fixed before all model fitting)
    Partition  ENMeval 'block' (Muscarella et al. 2014): 4 spatial blocks with ~equal records, split at
               the longitude median and then at the latitude medians. Each block is withheld once.
               Training uses the full background of the accessible area. Validation AUC and CBI compare the
               withheld records with the background of the SAME block (ENMeval validation.bg = 'partition';
               D53): against the full background, null models scored AUC ~0.2 because the withheld block's
               environment is under-represented in training, which biases the null comparison. Full-
               background values are also reported.
    Metrics    per fold: AUC train / validation, AUC difference, omission at the 10th-percentile training
               presence threshold (OR10) and at the minimum training presence (ORmtp), continuous Boyce
               index (CBI; Hirzel et al. 2006) of the validation records. Full model: number of non-zero
               parameters and AICc (Warren & Seifert 2011).
    Grid       regularisation 0.5-5 (original) extended with 6, 8, 10 (D53).
    Common     MAIN analysis (D58): ONE setting for the four baselines of a variant, chosen with the same rule
               on the mean over baselines (OR10 tolerance = one record in one baseline), so that the 2 x 2
               comparison differs only in climate. Baseline-specific settings are a sensitivity analysis.
    Selection  (1) lowest mean OR10, differences up to ONE omitted record (1 / (4 x smallest block size))
               treated as ties (D55); (2) highest mean validation AUC; (3) fewest parameters (Radosavljevic & Anderson
               2014; Kass et al. 2021). The strict rule (no tolerance, D51) and AICc are reported.
    Comparison the selected setting is also evaluated with random 4-fold CV (the original manuscript used
               random splits) to quantify the optimism of random validation (main variant).
    Null models 'n_null' null models per baseline for the selected setting (main variant): records drawn
               at random from the background cells of M, same number, same partition and settings;
               z-scores and one-sided p-values of validation AUC, OR10 and CBI (Bohl et al. 2019;
               Kass et al. 2020).

Runtime    Thousands of MaxEnt runs; parallel ('n_jobs') and resumable (finished runs are skipped).
Outputs    data_processed/13_tuning_spatial_cv/{variant}/{baseline}/  partitions and run results
           data_processed/13_tuning_spatial_cv/selected_settings.csv
QA         results/qa/13_tuning_spatial_cv/  13_report.txt, 13_*.csv, 13_fig_*.png
Manuscript results/manuscript/tables/T13_*.csv, notes/13_tuning_spatial_cv.md
Run        python scripts/13_tuning_spatial_cv.py          main variant (+ random CV, null models)
           python scripts/13_tuning_spatial_cv.py --all    adds the 8 sensitivity variants (resumes)
"""
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.maxent import (run_maxent, read_predictions, n_parameters, SPECIES,  # noqa: E402
                                resolve_java, java_version)
from hemiscorpius.evaluation import block_partition, random_kfold, aicc, validation_metrics, select_setting  # noqa: E402

STEP = "13_tuning_spatial_cv"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


# --------------------------------------------------------------------------- partitions on disk
def check_inputs_unchanged(base_dir, run_dir):
    """If the SWD inputs of a variant/baseline changed since the cached runs, delete the stale results."""
    import hashlib
    h = hashlib.sha256()
    for f in ("samples.csv", "background.csv"):
        h.update((base_dir / f).read_bytes())
    tag, f = h.hexdigest(), run_dir / "inputs_sha256.txt"
    if f.exists() and f.read_text().strip() != tag:
        for sub in ("results", "partitions_block", "partitions_random", "random_cv", "null"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
        print(f"    inputs changed since the cached runs -> stale results of {run_dir.parent.name}/{run_dir.name} deleted")
    run_dir.mkdir(parents=True, exist_ok=True)
    f.write_text(tag)


def write_partitions(src, dst, scheme, seed):
    """Write train/test SWD files for 4 folds; returns group labels of the records."""
    samples = pd.read_csv(src / "samples.csv")
    bg = pd.read_csv(src / "background.csv")
    if scheme == "block":
        g, gb = block_partition(samples.lon, samples.lat, bg.lon, bg.lat)
    else:
        g, gb = random_kfold(len(samples), 4, seed), np.zeros(len(bg), dtype=int)   # 0 = not partitioned
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lon": bg.lon.round(5), "lat": bg.lat.round(5), "group": gb}).to_csv(dst / "bg_groups.csv",
                                                                                     index=False)
    for k in range(1, 5):
        samples[g != k].to_csv(dst / f"train_{k}.csv", index=False)
        samples[g == k].to_csv(dst / f"test_{k}.csv", index=False)
    pd.DataFrame({"record": np.arange(len(g)), "group": g}).to_csv(dst / "groups.csv", index=False)
    return g


# --------------------------------------------------------------------------- one MaxEnt task
METRICS_VERSION = 5  # Complete spatial-block matching on the 2.5 arc-minute lattice.

def _bg_groups(task, pred):
    lut = pd.read_csv(task["bg_groups"])

    def grid_key(lon, lat):
        x = np.rint(np.asarray(lon, dtype=float) * 48).astype(int)
        y = np.rint(np.asarray(lat, dtype=float) * 48).astype(int)
        return x * 10000 + y

    lookup = lut.groupby(grid_key(lut.lon, lut.lat)).group
    if lookup.nunique().max() != 1:
        raise ValueError("Conflicting block assignments for a background cell")
    groups = lookup.first()
    ids = grid_key(pred["bg_x"], pred["bg_y"])
    if not np.array_equal(np.unique(ids), np.sort(groups.index)):
        raise ValueError("Background predictions and partition cover different cells")
    return groups.loc[ids].to_numpy()


def task_run(task):
    """Run one MaxEnt model (fold or full) and return its metrics; predictions are kept (npz) so that
    metrics can be recomputed without re-running MaxEnt."""
    res_file, pred_file = Path(task["result"]), Path(task["result"]).with_suffix(".npz")
    if res_file.exists():
        old = json.loads(res_file.read_text())
        if old.get("metrics_version") == METRICS_VERSION:
            return old
    out = Path(task["workdir"])
    try:
        if pred_file.exists():
            z = np.load(pred_file)
            pred = {k: z[k] for k in z.files}
            k_par = int(pred.get("n_params", np.array(-1)))
        else:
            run_maxent(task["jar"], out, task["samples"], task["background"], task["fc"], task["beta"],
                       test=task.get("test"), output=task["output"], java=task["java"], memory_mb=task["memory"])
            pred = read_predictions(out, task["output"])
            k_par = n_parameters(out)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pred_file, **{k: np.asarray(v, dtype="float32") for k, v in pred.items()},
                                n_params=np.array(k_par))
        r = dict(task["meta"])
        if task["kind"] == "fold":
            r.update(validation_metrics(pred, task["meta"]["fold"], _bg_groups(task, pred)))
        else:
            r.update(n_params=k_par, aicc=aicc(pred["train"], pred["background"], k_par))
        r.update(ok=True, metrics_version=METRICS_VERSION)
    except Exception as e:                                         # keep going, report later
        r = dict(task["meta"], ok=False, error=str(e)[:300], metrics_version=METRICS_VERSION)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text(json.dumps(r, default=float))
    return r


def run_tasks(tasks, n_jobs, label):
    t0, done, results = time.time(), 0, []
    def current(t):
        f = Path(t["result"])
        return f.exists() and json.loads(f.read_text()).get("metrics_version") == METRICS_VERSION
    todo = [t for t in tasks if not current(t)]
    results += [json.loads(Path(t["result"]).read_text()) for t in tasks if current(t)]
    print(f"  {label}: {len(tasks)} runs ({len(tasks) - len(todo)} already done)")
    if not todo:
        return results
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(task_run, t) for t in todo]
        step = max(1, len(todo) // 20)
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % step == 0 or done == len(todo):
                el = (time.time() - t0) / 60
                print(f"    {done}/{len(todo)} runs ({el:.1f} min; ~{el / done * (len(todo) - done):.0f} min left)")
    return results


def make_tasks(cfg, base_dir, part_dir, run_dir, settings, meta, n_folds=4, full=True, output="cloglog"):
    m = cfg["models"]
    common = dict(jar=str(ROOT / cfg["paths"]["maxent_jar"]), java=cfg["_java"],
                  memory=int(m.get("memory_mb", 1024)), background=str(base_dir / "background.csv"))
    tasks = []
    for fc, beta in settings:
        tag = f"{fc}_rm{beta:g}"
        for k in range(1, n_folds + 1):
            tasks.append(dict(common, kind="fold", fc=fc, beta=beta, output=output,
                              samples=str(part_dir / f"train_{k}.csv"), test=str(part_dir / f"test_{k}.csv"),
                              bg_groups=str(part_dir / "bg_groups.csv"),
                              workdir=str(run_dir / "tmp" / f"{tag}_f{k}"),
                              result=str(run_dir / "results" / f"{tag}_f{k}.json"),
                              meta=dict(meta, fc=fc, beta=beta, fold=k)))
        if full:
            tasks.append(dict(common, kind="full", fc=fc, beta=beta, output="raw",
                              samples=str(base_dir / "samples.csv"),
                              workdir=str(run_dir / "tmp" / f"{tag}_full"),
                              result=str(run_dir / "results" / f"{tag}_full.json"),
                              meta=dict(meta, fc=fc, beta=beta, fold=0)))
    return tasks


def summarise(results):
    df = pd.DataFrame(results)
    if "ok" in df and (~df.ok.astype(bool)).any():
        bad = df[~df.ok.astype(bool)]
        note("WARN", f"{len(bad)} MaxEnt runs failed (first: {bad.iloc[0].get('error', '')[:120]})")
    df = df[df.ok.astype(bool)]
    folds = df[df.fold > 0]
    keys = [k for k in ("variant", "baseline", "scheme", "fc", "beta") if k in folds]
    agg = folds.groupby(keys).agg(
        auc_val_mean=("auc_val", "mean"), auc_val_sd=("auc_val", "std"), auc_train_mean=("auc_train", "mean"),
        auc_val_fullbg_mean=("auc_val_fullbg", "mean"), cbi_val_fullbg_mean=("cbi_val_fullbg", "mean"),
        auc_diff_mean=("auc_diff", "mean"), or10_mean=("or10", "mean"), or10_sd=("or10", "std"),
        ormtp_mean=("ormtp", "mean"), cbi_val_mean=("cbi_val", "mean"), n_folds=("fold", "count")).reset_index()
    full = df[df.fold == 0]
    if len(full):
        agg = agg.merge(full[[k for k in keys if k != "scheme"] + ["n_params", "aicc"]],
                        on=[k for k in keys if k != "scheme"], how="left")
    return agg


def one_record_tolerance(fold_sizes, k=4):
    """Change of the mean OR10 caused by ONE omitted record in the smallest fold: 1 / (k x min fold size).
    With folds of 10, 9, 10, 9 records this is 1/36 = 0.028 (D55)."""
    return 1.0 / (k * min(fold_sizes)) + 1e-9


def select(agg, tolerance):
    """Main rule (D55): OR10 ties within one omitted record -> max AUCval -> fewest parameters.
    Also returns the strict rule (D51, tolerance 0) and the AICc choice for reporting."""
    best = select_setting(agg, tolerance).copy()
    strict = select_setting(agg, 0.0)
    a = agg[(agg.n_folds == 4) & (agg.n_params > 0)]
    fin = a[np.isfinite(a.aicc)]
    best_aicc = fin.sort_values("aicc").iloc[0] if len(fin) else None
    return best, strict, best_aicc


# --------------------------------------------------------------------------- main
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    proc = project_path(cfg, "processed")
    inp, out = proc / "12_maxent_inputs", proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "13_run_info.json", CONFIG_FILE)
    m = cfg["models"]
    fcs, betas = list(m["feature_classes"]), [float(b) for b in cfg["maxent"]["regularization"]]
    settings = [(fc, b) for fc in fcs for b in betas]
    n_jobs, n_null, seed = int(m.get("n_jobs", 4)), int(m.get("n_null", 100)), int(cfg["project"]["seed"])
    run_all = "--all" in sys.argv or bool(m.get("run_sensitivity", False))
    variants = list(m["variants"]) if run_all else ["main"]

    with Report(qa / "13_report.txt"):
        print(f"Step {STEP}: {len(settings)} settings ({fcs} x {betas}); variants {variants}; "
              f"n_jobs {n_jobs}; null models {n_null}")
        jar = ROOT / cfg["paths"]["maxent_jar"]
        note("OK" if jar.exists() else "FAIL", f"maxent.jar: {jar}")
        try:
            cfg["_java"] = resolve_java(m.get("java", "java"))
            note("OK", f"Java: {cfg['_java']} ({java_version(cfg['_java'])})")
        except FileNotFoundError as e:
            note("FAIL", str(e))
            return
        if not jar.exists():
            return
        print("  mode: " + ("main + all sensitivity variants" if run_all else
                            "main variant only (run with --all for the sensitivity variants)"))

        section("1. Tuning (spatial block CV, 4 folds) for every variant and baseline")
        all_agg, selected = [], []
        for v in variants:
            if not (inp / v / BASELINES[0] / "samples.csv").exists():
                note("WARN", f"variant {v}: MaxEnt inputs not found (run step 12) - skipped")
                continue
            for b in BASELINES:
                base_dir = inp / v / b
                check_inputs_unchanged(base_dir, out / v / b)
                part = out / v / b / "partitions_block"
                g = write_partitions(base_dir, part, "block", seed)
                tasks = make_tasks(cfg, base_dir, part, out / v / b, settings,
                                   dict(variant=v, baseline=b, scheme="block"))
                agg = summarise(run_tasks(tasks, n_jobs, f"{v}/{b} (records per block {np.bincount(g)[1:].tolist()})"))
                all_agg.append(agg)
                tol = one_record_tolerance(np.bincount(g)[1:])
                best, strict, best_a = select(agg, tol)
                selected.append(dict(variant=v, mode="specific", baseline=b, fc=best.fc, beta=best.beta, or10=best.or10_mean,
                                     strict_fc=strict.fc, strict_beta=strict.beta, strict_or10=strict.or10_mean,
                                     strict_auc_val=strict.auc_val_mean, or10_tolerance=tol,
                                     ormtp=best.ormtp_mean, auc_val=best.auc_val_mean, auc_val_sd=best.auc_val_sd,
                                     auc_diff=best.auc_diff_mean, cbi_val=best.cbi_val_mean, n_params=best.n_params,
                                     auc_val_fullbg=best.auc_val_fullbg_mean, cbi_val_fullbg=best.cbi_val_fullbg_mean,
                                     beta_at_grid_edge=bool(best.beta == max(betas)),
                                     aicc_fc=None if best_a is None else best_a.fc,
                                     aicc_beta=None if best_a is None else best_a.beta))
                print(f"    selected {best.fc} rm{best.beta:g}: OR10 {best.or10_mean:.3f}, AUCval "
                      f"{best.auc_val_mean:.3f}, CBI {best.cbi_val_mean:.2f}, params {int(best.n_params)}"
                      f" | strict rule {strict.fc} rm{strict.beta:g} (OR10 {strict.or10_mean:.3f}, AUCval "
                      f"{strict.auc_val_mean:.3f})"
                      + ("" if best_a is None else f" | AICc choice {best_a.fc} rm{best_a.beta:g}"))
        edge = [f"{r['variant']}/{r['baseline']}" for r in selected if r["beta_at_grid_edge"]]
        note("OK" if not edge else "WARN",
             f"selected regularisation at the upper edge of the grid ({max(betas):g}): {edge or 'none'}")
        tuning = pd.concat(all_agg, ignore_index=True)
        tuning.to_csv(qa / "13_tuning_all_settings.csv", index=False)
        section("1b. Common setting for the four baselines of every variant (MAIN analysis, D58)")
        for v in tuning.variant.unique():
            t = tuning[(tuning.variant == v) & (tuning.n_folds == 4) & (tuning.n_params > 0)]
            g = (t.groupby(["fc", "beta"]).agg(or10=("or10_mean", "mean"), auc=("auc_val_mean", "mean"),
                                               npar=("n_params", "mean"), nb=("baseline", "nunique")).reset_index())
            g = g[g.nb == len(BASELINES)]
            tol_b = min(r["or10_tolerance"] for r in selected if r["variant"] == v)
            # MAIN (D59): same resolution as the per-baseline rule - mean OR10 differences up to one omitted record
            # per model are ties. Sensitivity 'common_strict' (D58 as first implemented): one record in ONE baseline.
            picks = {}
            for mode_c, tol_c in (("common", tol_b), ("common_strict", tol_b / len(BASELINES))):
                elig = g[g.or10 <= g.or10.min() + tol_c + 1e-12]
                picks[mode_c] = (elig.sort_values(["auc", "npar"], ascending=[False, True]).iloc[0], tol_c)
            for mode_c, (c, tol_c) in picks.items():
                print(f"  {v}: {mode_c} setting {c.fc} rm{c.beta:g} (mean OR10 {c.or10:.3f}, mean AUCval {c.auc:.3f}, "
                      f"mean CBI {t[(t.fc == c.fc) & (t.beta == c.beta)].cbi_val_mean.mean():.2f}, "
                      f"mean parameters {c.npar:.1f})")
            for mode_c, (c, tol_c) in picks.items():
              for b in BASELINES:
                rb = t[(t.baseline == b) & (t.fc == c.fc) & (t.beta == c.beta)].iloc[0]
                selected.append(dict(variant=v, mode=mode_c, baseline=b, fc=c.fc, beta=c.beta, or10=rb.or10_mean,
                                     ormtp=rb.ormtp_mean, auc_val=rb.auc_val_mean, auc_val_sd=rb.auc_val_sd,
                                     auc_diff=rb.auc_diff_mean, cbi_val=rb.cbi_val_mean, n_params=rb.n_params,
                                     auc_val_fullbg=rb.auc_val_fullbg_mean, cbi_val_fullbg=rb.cbi_val_fullbg_mean,
                                     or10_tolerance=tol_c, beta_at_grid_edge=bool(c.beta == max(betas))))
                if v == "main":
                    print(f"      {mode_c} {b}: OR10 {rb.or10_mean:.3f}, AUCval {rb.auc_val_mean:.3f}, "
                          f"CBI {rb.cbi_val_mean:.2f}, params {int(rb.n_params)}")
        sel = pd.DataFrame(selected)
        sel.to_csv(out / "selected_settings.csv", index=False)
        sel.to_csv(tables / "T13_selected_settings.csv", index=False)

        section("2. Random 4-fold CV for the selected settings (optimism of random validation; main)")
        cmp_rows = []
        for mode, b in [(m_, b_) for m_ in ("common", "common_strict", "specific") for b_ in BASELINES]:
            s = sel[(sel.variant == "main") & (sel["mode"] == mode) & (sel.baseline == b)].iloc[0]
            base_dir = inp / "main" / b
            part = out / "main" / b / "partitions_random"
            write_partitions(base_dir, part, "random", seed)
            tasks = make_tasks(cfg, base_dir, part, out / "main" / b / "random_cv", [(s.fc, float(s.beta))],
                               dict(variant="main", baseline=b, scheme="random"), full=False)
            r = summarise(run_tasks(tasks, n_jobs, f"random CV {mode} {b}"))
            cmp_rows.append(dict(mode=mode, baseline=b, setting=f"{s.fc} rm{s.beta:g}", auc_val_block=s.auc_val, auc_val_random=r.auc_val_mean.iloc[0],
                                 or10_block=s.or10, or10_random=r.or10_mean.iloc[0],
                                 cbi_block=s.cbi_val, cbi_random=r.cbi_val_mean.iloc[0]))
        cmp = pd.DataFrame(cmp_rows)
        cmp.to_csv(qa / "13_block_vs_random_cv.csv", index=False)
        cmp.to_csv(tables / "T13_block_vs_random_cv.csv", index=False)
        print("   " + cmp.round(3).to_string(index=False).replace("\n", "\n   "))

        section(f"3. Null models ({n_null} per baseline, main variant, selected settings)")
        rng = np.random.default_rng(seed)
        null_rows = []
        for mode, b in [(m_, b_) for m_ in ("common", "common_strict", "specific") for b_ in BASELINES]:
            s = sel[(sel.variant == "main") & (sel["mode"] == mode) & (sel.baseline == b)].iloc[0]
            base_dir = inp / "main" / b
            samples = pd.read_csv(base_dir / "samples.csv")
            bg = pd.read_csv(base_dir / "background.csv")
            tasks = []
            for i in range(1, n_null + 1):
                nd = out / "main" / b / "null" / f"null_{i:03d}"
                if not (nd / "samples.csv").exists():
                    pick = rng.choice(len(bg), size=len(samples), replace=False)
                    ns = bg.iloc[pick].copy()
                    ns["species"] = SPECIES
                    nd.mkdir(parents=True, exist_ok=True)
                    ns.to_csv(nd / "samples.csv", index=False)
                    shutil.copy(base_dir / "background.csv", nd / "background.csv")
                write_partitions(nd, nd / "partitions_block", "block", seed)
                tasks += make_tasks(cfg, nd, nd / "partitions_block", nd, [(s.fc, float(s.beta))],
                                    dict(variant="main", baseline=b, scheme="block", null=i), full=False)
            res = pd.DataFrame(run_tasks(tasks, n_jobs, f"null models {mode} {b} ({s.fc} rm{s.beta:g})"))
            res = res[(res.fc == s.fc) & (res.beta == float(s.beta))]
            res = res[res.ok.astype(bool)]
            nm = res.groupby("null")[["auc_val", "auc_val_fullbg", "or10", "cbi_val", "bg_pred_sd"]].mean()
            emp = dict(auc_val=s.auc_val, or10=s.or10, cbi_val=s.cbi_val)
            row = dict(mode=mode, baseline=b, setting=f"{s.fc} rm{s.beta:g}", n_null=len(nm))
            for k, better in (("auc_val", "higher"), ("cbi_val", "higher"), ("or10", "lower")):
                mu, sd = nm[k].mean(), nm[k].std()
                row[f"{k}_empirical"], row[f"{k}_null_mean"], row[f"{k}_null_sd"] = emp[k], mu, sd
                row[f"{k}_z"] = (emp[k] - mu) / sd if sd > 0 else np.nan
                # codex_20261112: undefined CBI cannot count as a null non-exceedance.
                from hemiscorpius.revision_checks import null_tail_summary
                ns = null_tail_summary(nm[k].to_numpy(), emp[k], better)
                row[f"{k}_p"] = ns["p"]
                row[f"{k}_n_valid"] = ns["n_valid"]
                row[f"{k}_n_undefined"] = ns["n_undefined"]
                row[f"{k}_test_status"] = ns["status"]
            row["auc_val_fullbg_null_mean"] = nm["auc_val_fullbg"].mean()
            row["share_constant_null_models"] = float((nm["bg_pred_sd"] < 1e-6).mean())
            print(f"  {mode} {b}: {row['share_constant_null_models']:.0%} of null models are constant (no parameter); "
                  f"OR10 is uninformative for constant models; undefined CBI is excluded from inferential claims; AUC is primary")
            null_rows.append(row)
            nm.to_csv(qa / f"13_null_models_{mode}_{b}.csv")
            mu = nm["auc_val"].mean()
            note("OK" if 0.4 <= mu <= 0.6 else "WARN",
                 f"{mode} {b}: mean null validation AUC {mu:.3f} (expected ~0.5 for random records)")
        nul = pd.DataFrame(null_rows)
        nul.to_csv(qa / "13_null_model_summary.csv", index=False)
        nul.to_csv(tables / "T13_null_models.csv", index=False)
        print("   " + nul[["mode", "baseline", "setting", "auc_val_empirical", "auc_val_null_mean", "auc_val_p",
                           "cbi_val_empirical", "cbi_val_p", "or10_empirical", "or10_p"]].round(3)
              .to_string(index=False).replace("\n", "\n   "))
        for _, r in nul.iterrows():
            note("OK" if r.auc_val_p < 0.05 else "WARN",
                 f"{r['mode']} {r.baseline}: validation AUC better than null models p = {r.auc_val_p:.3f}")

        # ------------------------------------------------------------------ figure
        main_t = tuning[tuning.variant == "main"]
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        for j, b in enumerate(BASELINES):
            t = main_t[main_t.baseline == b]
            for i, (metric, cmap) in enumerate((("or10_mean", "viridis_r"), ("auc_val_mean", "viridis"))):
                piv = t.pivot(index="fc", columns="beta", values=metric).reindex(fcs)
                ax = axes[i, j]
                im = ax.imshow(piv.values, cmap=cmap, aspect="auto")
                ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns], fontsize=7)
                ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
                for mode, mk, col in (("specific", "*", "red"), ("common_strict", "s", "white"), ("common", "o", "white")):
                    s = sel[(sel.variant == "main") & (sel["mode"] == mode) & (sel.baseline == b)].iloc[0]
                    ax.scatter(list(piv.columns).index(s.beta), list(piv.index).index(s.fc), marker=mk, s=110,
                               facecolors="none" if mode != "specific" else col,
                               edgecolors={"common": "red", "common_strict": "orange"}.get(mode, col),
                               linewidths=1.5, label=mode)
                ax.set_title(f"{b}: {metric}", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.8)
        axes[0, 0].legend(fontsize=7, loc="lower left")
        fig.suptitle("circle = common setting (MAIN, D59); square = common, strict tolerance; star = baseline-specific")
        fig.tight_layout()
        fig.savefig(qa / "13_fig_tuning_main.png", dpi=170)
        plt.close(fig)

        # ------------------------------------------------------------------ notes
        n = ManuscriptNotes(cfg, STEP, "Model tuning, selection and null models")
        for _, r in sel.iterrows():
            k = f"{r.variant}_{r['mode']}_{r.baseline}"
            n.number(f"selected_{k}", f"{r.fc} rm{r.beta:g}", "", f"selected setting {k}", "R1-4")
            for mtr in ("or10", "auc_val", "cbi_val", "auc_diff"):
                n.number(f"{mtr}_{k}", round(float(r[mtr]), 4), "", f"{mtr} of selected setting, {k}", "R1-3")
        for _, r in cmp.iterrows():
            n.number(f"auc_val_random_minus_block_{r['mode']}_{r.baseline}", round(float(r.auc_val_random - r.auc_val_block), 4),
                     "", "optimism of random CV in validation AUC", "R1-3")
        for _, r in nul.iterrows():
            n.number(f"null_auc_p_{r['mode']}_{r.baseline}", round(float(r.auc_val_p), 4), "", "null-model p for validation AUC", "R1-2")
            n.number(f"null_cbi_p_{r['mode']}_{r.baseline}", round(float(r.cbi_val_p), 4), "", "null-model p for CBI", "R1-2")
        n.text("Methods (draft)", f"""
MaxEnt 3.4.4 models were tuned for each baseline over {len(fcs)} feature-class combinations ({', '.join(fcs)}) and
{len(betas)} regularisation multipliers ({', '.join(f'{b:g}' for b in betas)}). Performance was estimated with spatial
block cross-validation (four blocks with similar numbers of records, ENMeval 'block' method; Muscarella et al.
2014), training on three blocks and validating the withheld records against the background of the same block
(validation against the full background biased null models to AUC ~0.2 and is reported only for completeness). The final setting was chosen with a sequential rule: lowest mean omission rate at the 10th-percentile training
presence threshold (OR10), treating differences smaller than one omitted record as ties because OR10 is
estimated from about ten records per block, then highest mean validation AUC, then fewest parameters
(Radosavljevic & Anderson 2014; Kass et al. 2021); the rule without tolerance and selection by AICc (Warren &
Seifert 2011) are reported for comparison. The continuous Boyce index (Hirzel et al. 2006) of the validation records was also computed. To
quantify the optimism of random validation used in the original analysis, the selected settings were also
evaluated with random four-fold cross-validation. Significance was assessed against {n_null} null models per
baseline built from random background cells with the same number of records, partitions and settings
(Bohl et al. 2019; Kass et al. 2020).""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 13 completed.")


if __name__ == "__main__":
    main()
