"""
Step 15 - Projections: present, cross-period transfer and future (2041-2060)
============================================================================

Purpose
    Map habitat suitability for every final model on (a) its own baseline climate, (b) the other period of the
    same product (temporal transfer: 1970-1999 models onto 2000-2024 climate and vice versa), and (c) the
    product-consistent future climate of five GCMs x two SSPs (R1-6), with GCM ensembles and agreement maps.

Method (decision D60)
    Projections are computed in Python from each model's .lambdas file (src/hemiscorpius/lambdas.py: linear,
    quadratic, product, hinge and threshold features, clamping to the training range, cloglog). BEFORE any
    projection is used, the evaluator is checked against MaxEnt's own background predictions of every full
    model (max |difference| must be < 1e-3; FAIL otherwise).
    Models   main/common (MAIN; full + 10 bootstrap replicates), main/common_strict, main/specific, and the
             common-setting full model of every sensitivity variant (with --all).
    Targets  present (own baseline) | cross_period (other period, same product) |
             fut_present_{gcm}_{ssp}  (MAIN future, D36; cleaned variant uses cleaned futures) |
             fut_anchor1970_{gcm}_{ssp}, fut_published_{gcm}_{ssp}  (future sensitivities, main/common only)
    Domain    projections cover Iran, but areas and clamping are summarised INSIDE the main accessible area M
             (primary, D61); outside M the models extrapolate (e.g. humid Caspian lowlands) - see MESS (step 16).
    Change    future change is measured against the projection on the 2000-2024 climate of the same product
             (own present for 2000-2024 models, cross_period for 1970-1999 models), so that it contains only the
             GCM change for every model (D61); agreement = number of GCMs with gain / loss > 0.05.
    Summaries geodesic suitable area at p10 / MTP / maxSSS thresholds, mean suitability, share of the domain
             with any predictor outside the training range (clamped), GCM ensemble mean / SD and the number of
             GCMs predicting suitable conditions (agreement), temporal-transfer evaluation with the records
             (AUC and CBI against the main background, share of records above p10), records predicted below
             p10 (omitted) for each baseline.

Outputs  data_processed/15_projections/{variant}/{mode}/{baseline}.nc            (full model, all targets)
         data_processed/15_projections/main/common/{baseline}_bootstrap.nc      (replicates x targets)
QA       results/qa/15_projections/  15_report.txt, 15_*.csv, 15_fig_*.png
Manuscript results/manuscript/tables/T15_*.csv, notes/15_projections.md
Run      python scripts/15_projections.py          (main variant: common, common_strict, specific)
         python scripts/15_projections.py --all    (adds the sensitivity variants)
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
from hemiscorpius.grid import ref_grid, to_dataarray, nc_encoding, cell_area_km2  # noqa: E402
from hemiscorpius.lambdas import predict, outside_range  # noqa: E402
from hemiscorpius.evaluation import auc, boyce  # noqa: E402
from hemiscorpius.maxent import SPECIES  # noqa: E402

STEP = "15_projections"
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


class Climates:
    """Lazy loader of predictor values (domain cells only) for every projection target."""

    def __init__(self, proc, cfg, mask):
        import xarray as xr
        self.xr, self.proc, self.cfg, self.mask = xr, proc, cfg, mask
        self.idx = np.where(mask)
        self.cache = {}
        self.per = cfg["future"]["period"].replace("-", "_")

    def _read(self, path, names):
        key = (str(path), tuple(names))
        if key not in self.cache:
            with self.xr.open_dataset(path) as ds:
                self.cache[key] = {v: ds[v].values[self.idx].astype("float64") for v in names if v in ds}
        return self.cache[key]

    def baseline(self, b, version, preds):
        return self._read(self.proc / "08_common_mask" / f"{b}_{version}_candidates.nc", preds)

    def future(self, product, method, gcm, ssp, version, preds, static_from):
        stat = [p for p in preds if not p.startswith("bio")]
        bio = [p for p in preds if p.startswith("bio")]
        if method == "published":
            f = self.proc / "05_future_delta" / f"CMIP6_future_published_{gcm}_{ssp}_{self.per}_bioclim.nc"
        else:
            f = self.proc / "05_future_delta" / f"{product}_future_{method}_{gcm}_{ssp}_{self.per}_{version}_bioclim.nc"
        env = dict(self._read(f, bio))
        env.update(self.baseline(static_from, version, stat))
        return env


def targets_for(b, version, cfg, main_common):
    prod, period = b.split("_", 1)
    other = f"{prod}_{'2000_2024' if period == '1970_1999' else '1970_1999'}"
    t = [("present", ("baseline", b, version)), ("cross_period", ("baseline", other, version))]
    methods = ["present"] + (["anchor1970", "published"] if main_common else [])
    for method in methods:
        for gcm in cfg["future"]["gcms"]:
            for ssp in cfg["future"]["ssps"]:
                ver = version if method == "present" else "original"
                t.append((f"fut_{method}_{gcm}_{ssp}", ("future", prod, method, gcm, ssp, ver)))
    return t


def env_for(clim, spec, preds, b):
    if spec[0] == "baseline":
        return clim.baseline(spec[1], spec[2], preds)
    _, prod, method, gcm, ssp, ver = spec
    return clim.future(prod, method, gcm, ssp, ver, preds, static_from=b)


def validate(lam, bg_csv, bgpred_csv, preds):
    """Max |Python - MaxEnt| on the background of the model."""
    bg = pd.read_csv(bg_csv).drop_duplicates(["lon", "lat"])
    mp = pd.read_csv(bgpred_csv)
    col = [c for c in mp.columns if "cloglog" in c.lower()][0]
    mp = mp.rename(columns={mp.columns[0]: "lon", mp.columns[1]: "lat"})
    mp["lon"], mp["lat"] = mp.lon.round(5), mp.lat.round(5)
    bg["lon_r"], bg["lat_r"] = bg.lon.round(5), bg.lat.round(5)
    m = bg.merge(mp.drop_duplicates(["lon", "lat"]), left_on=["lon_r", "lat_r"], right_on=["lon", "lat"])
    py = predict(lam, {p: m[p].values for p in preds})
    return float(np.nanmax(np.abs(py - m[col].values))), len(m)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out_root = proc / STEP
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "15_run_info.json", CONFIG_FILE)
    run_all = "--all" in sys.argv
    with rasterio.open(proc / "08_common_mask" / "common_mask.tif") as r_:
        mask = r_.read(1) == 1
    area = cell_area_km2(grid)[mask]
    with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
        in_m = (r_.read(1) == 1)[mask]          # main accessible area, on the domain cells (D61)
    clim = Climates(proc, cfg, mask)
    summ14 = pd.read_csv(proc / "14_final_models" / "final_models_summary.csv")
    if "mode" not in summ14:
        summ14["mode"] = "common"
    models = summ14[summ14.variant.isin(["main"] if not run_all else summ14.variant.unique())]
    occ_all = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_all.csv")
    cal_ids = set(pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_thin10km.csv")["id"])
    bg_main = pd.read_csv(proc / "10_accessible_area_background" / "background_main.csv")
    pos = {(r, c): i for i, (r, c) in enumerate(zip(*clim.idx))}
    occ_i = np.array([pos.get((r, c), -1) for r, c in zip(occ_all.row, occ_all.col)])
    bg_i = np.array([pos.get((r, c), -1) for r, c in zip(bg_main.row, bg_main.col)])

    with Report(qa / "15_report.txt"):
        print(f"Step {STEP}: {len(models)} final models; domain cells {mask.sum():,}")
        if (occ_i < 0).any():
            note("WARN", f"{int((occ_i < 0).sum())} records outside the domain")

        section("1. Validation of the Python projection against MaxEnt (background of every full model)")
        val_rows = []
        for _, m in models.iterrows():
            d = proc / "14_final_models" / m.variant / m["mode"] / m.baseline / "full"
            src = proc / "12_maxent_inputs" / m.variant / m.baseline
            preds = [c for c in pd.read_csv(src / "samples.csv", nrows=1).columns if c not in ("species", "lon", "lat")]
            err, n = validate(d / f"{SPECIES}.lambdas", src / "background.csv",
                              d / f"{SPECIES}_backgroundPredictions.csv", preds)
            val_rows.append(dict(variant=m.variant, mode=m["mode"], baseline=m.baseline, max_abs_diff=err, n=n))
        val = pd.DataFrame(val_rows)
        val.to_csv(qa / "15_validation_vs_maxent.csv", index=False)
        worst = val.max_abs_diff.max()
        note("OK" if worst < 1e-3 else "FAIL", f"Python vs MaxEnt cloglog: max |difference| {worst:.2e} over "
                                              f"{len(val)} models (tolerance 1e-3)")
        if worst >= 1e-3:
            print(val.sort_values("max_abs_diff", ascending=False).head().to_string(index=False))
            return

        section("2. Projections")
        rows, transfer_rows, omit_rows = [], [], []
        for _, m in models.iterrows():
            v, mode, b = m.variant, m["mode"], m.baseline
            spec = variant_spec(cfg, v)
            src = proc / "12_maxent_inputs" / v / b
            preds = [c for c in pd.read_csv(src / "samples.csv", nrows=1).columns if c not in ("species", "lon", "lat")]
            main_common = (v == "main" and mode == "common")
            tg = targets_for(b, spec["version"], cfg, main_common)
            lam = proc / "14_final_models" / v / mode / b / "full" / f"{SPECIES}.lambdas"
            od = out_root / v / mode
            od.mkdir(parents=True, exist_ok=True)
            fout = od / f"{b}.nc"
            thr = {k: float(m[f"thr_{k}"]) for k in ("p10", "mtp", "maxsss")}
            data = {}
            for name, s in tg:
                env = env_for(clim, s, preds, b)
                p = predict(lam, env)
                data[name] = p
                o = outside_range(lam, env)
                r = dict(variant=v, mode=mode, baseline=b, target=name, mean_suitability=float(np.nanmean(p)),
                         mean_suitability_M=float(np.nanmean(p[in_m])), share_clamped=float(o.mean()),
                         share_clamped_M=float(o[in_m].mean()))
                for k, t in thr.items():
                    r[f"area_km2_{k}"] = float(area[p >= t].sum())
                    r[f"area_km2_{k}_M"] = float(area[(p >= t) & in_m].sum())
                rows.append(r)
            # reference = projection on the 2000-2024 climate of the same product, so that future change is the
            # GCM change only for every model (D61): own present for 2000-2024 models, cross_period for 1970-1999
            ref = data["present"] if b.endswith("2000_2024") else data["cross_period"]
            data["reference_2000_2024"] = ref
            # GCM ensembles and agreement (main future method)
            for ssp in cfg["future"]["ssps"]:
                st = np.stack([data[f"fut_present_{g}_{ssp}"] for g in cfg["future"]["gcms"]])
                data[f"ens_present_{ssp}_mean"] = st.mean(axis=0)
                data[f"ens_present_{ssp}_sd"] = st.std(axis=0, ddof=1)
                data[f"ens_present_{ssp}_n_suitable_p10"] = (st >= thr["p10"]).sum(axis=0).astype(float)
                pm = data[f"ens_present_{ssp}_mean"]
                data[f"change_{ssp}_vs_2000_2024"] = pm - ref
                ch = np.stack([data[f"fut_present_{g}_{ssp}"] - ref for g in cfg["future"]["gcms"]])
                data[f"change_{ssp}_n_gcm_gain"] = (ch > 0.05).sum(axis=0).astype(float)
                data[f"change_{ssp}_n_gcm_loss"] = (ch < -0.05).sum(axis=0).astype(float)
                rows.append(dict(variant=v, mode=mode, baseline=b, target=f"ens_present_{ssp}_mean",
                                 mean_suitability=float(np.nanmean(pm)), mean_suitability_M=float(np.nanmean(pm[in_m])),
                                 share_clamped=np.nan, share_clamped_M=np.nan,
                                 **{f"area_km2_{k}": float(area[pm >= t].sum()) for k, t in thr.items()},
                                 **{f"area_km2_{k}_M": float(area[(pm >= t) & in_m].sum()) for k, t in thr.items()}))
            full = np.full(grid.shape, np.nan, "float32")
            das = []
            for name, arr in data.items():
                g_ = full.copy()
                g_[clim.idx] = arr
                das.append(to_dataarray(g_, grid, name))
            ds = xr.merge(das)
            ds.attrs.update(variant=v, mode=mode, baseline=b, fc=m.fc, beta=float(m.beta),
                            thr_p10=thr["p10"], thr_mtp=thr["mtp"], thr_maxsss=thr["maxsss"])
            ds.to_netcdf(fout, encoding=nc_encoding(ds))
            # temporal transfer and omitted records (main variant)
            if v == "main":
                for name in ("present", "cross_period"):
                    p = data[name]
                    pr, pb = p[occ_i[occ_i >= 0]], p[bg_i[bg_i >= 0]]
                    transfer_rows.append(dict(mode=mode, baseline=b, target=name, auc_records_vs_bg=auc(pr, pb),
                                              cbi_records=boyce(pr, pb), share_records_above_p10=float(np.mean(pr >= thr["p10"])),
                                              mean_suitability_records=float(np.mean(pr))))
                if mode == "common":
                    p = data["present"][occ_i]
                    for rid, val_, ins in zip(occ_all["id"], p, occ_all["id"].isin(cal_ids)):
                        omit_rows.append(dict(baseline=b, id=rid, calibration=bool(ins), suitability=float(val_),
                                              below_p10=bool(val_ < thr["p10"])))
            print(f"  {v}/{mode}/{b}: {len(tg)} targets -> {fout.name}")

        # bootstrap replicates (MAIN)
        section("3. Bootstrap replicates (main/common): present and main futures")
        for _, m in models[(models.variant == "main") & (models["mode"] == "common")].iterrows():
            b = m.baseline
            src = proc / "12_maxent_inputs" / "main" / b
            preds = [c for c in pd.read_csv(src / "samples.csv", nrows=1).columns if c not in ("species", "lon", "lat")]
            bdir = proc / "14_final_models" / "main" / "common" / b / "paired_bootstrap"
            lams = sorted(bdir.glob(f"rep_*/model/{SPECIES}.lambdas"))           # paired bootstrap (D67)
            tg = [t for t in targets_for(b, "original", cfg, False) if t[0] == "present" or t[0].startswith("fut_present")]
            arrs = {name: [] for name, _ in tg}
            for lam in lams:
                for name, s in tg:
                    arrs[name].append(predict(lam, env_for(clim, s, preds, b)))
            das = []
            for name, lst in arrs.items():
                g_ = np.full((len(lst),) + grid.shape, np.nan, "float32")
                for i, a in enumerate(lst):
                    g_[i][clim.idx] = a
                das.append(to_dataarray(g_, grid, name, dim0="rep", coord0=np.arange(len(lst))))
            ds = xr.merge(das)
            ds.to_netcdf(out_root / "main" / "common" / f"{b}_bootstrap.nc", encoding=nc_encoding(ds))
            sd = np.nanmean(np.std(np.stack(arrs["present"]), axis=0, ddof=1))
            print(f"  {b}: {len(lams)} replicates; mean SD of present suitability across replicates {sd:.3f}")

        # ------------------------------------------------------------------ tables
        section("4. Summaries")
        st = pd.DataFrame(rows)
        st.to_csv(qa / "15_projection_summary.csv", index=False)
        tr = pd.DataFrame(transfer_rows)
        tr.to_csv(qa / "15_temporal_transfer.csv", index=False)
        tr.round(4).to_csv(tables / "T15_temporal_transfer.csv", index=False)
        om = pd.DataFrame(omit_rows)
        om.to_csv(qa / "15_records_suitability_main.csv", index=False)
        mc = st[(st.variant == "main") & (st["mode"] == "common")]
        key = mc[mc.target.isin(["present", "cross_period", "ens_present_ssp245_mean", "ens_present_ssp585_mean"])]
        for col_, lab in (("area_km2_p10_M", "INSIDE the main accessible area M (primary, D61)"),
                          ("area_km2_p10", "whole of Iran (display only; extrapolation outside M, see step 16)")):
            piv = key.pivot(index="baseline", columns="target", values=col_).reindex(list(BASELINES))
            piv.to_csv(tables / f"T15_suitable_{col_}_main.csv")
            print(f"  suitable area at p10 (km2), main/common, {lab}:")
            print("   " + piv.round(0).to_string().replace("\n", "\n   "))
        print("\n  temporal transfer (records vs main background):")
        print("   " + tr.round(3).to_string(index=False).replace("\n", "\n   "))
        clamp = mc[mc.target.str.startswith("fut_present")].groupby("baseline")[["share_clamped", "share_clamped_M"]].max()
        print("\n  largest share with clamped predictors in a main future (whole Iran | inside M): "
              + ", ".join(f"{k} {r.share_clamped:.1%} | {r.share_clamped_M:.1%}" for k, r in clamp.iterrows()))
        if len(om):
            low = om[om.below_p10].groupby("baseline")["id"].apply(lambda s: ", ".join(sorted(s)))
            print("\n  records below the p10 threshold (present climate, main/common):")
            for b in BASELINES:
                print(f"   {b}: {low.get(b, 'none')}")

        # ------------------------------------------------------------------ figure
        with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
            m_full = r_.read(1).astype(float)
        fig, axes = plt.subplots(4, 4, figsize=(18, 15))
        for i, b in enumerate(BASELINES):
            with xr.open_dataset(out_root / "main" / "common" / f"{b}.nc") as ds:
                panels = [("present", ds["present"].values, "viridis", (0, 1)),
                          ("future SSP2-4.5 (GCM mean)", ds["ens_present_ssp245_mean"].values, "viridis", (0, 1)),
                          ("future SSP5-8.5 (GCM mean)", ds["ens_present_ssp585_mean"].values, "viridis", (0, 1)),
                          ("change SSP5-8.5 vs 2000-2024 climate", ds["change_ssp585_vs_2000_2024"].values,
                           "RdBu_r", (-0.5, 0.5))]
            for j, (t, a, cm, lim) in enumerate(panels):
                ax = axes[i, j]
                im = ax.imshow(a, cmap=cm, vmin=lim[0], vmax=lim[1],
                               extent=(grid.west, grid.west + grid.ncols / 24, grid.north - grid.nrows / 24, grid.north))
                ax.contour(grid.lons, grid.lats, np.where(np.isfinite(m_full), m_full, 0), levels=[0.5],
                           colors="white", linewidths=0.6)
                ax.scatter(occ_all.lon, occ_all.lat, s=3, c="red")
                ax.set_title(f"{b}: {t}", fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(qa / "15_fig_projections_main.png", dpi=150)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Projections")
        for _, r in key.iterrows():
            n.number(f"area_p10_{r.baseline}_{r.target}", round(r.area_km2_p10, 0), "km2",
                     f"suitable area (p10) {r.baseline} {r.target}", "R1 minor 5")
        for _, r in tr.iterrows():
            n.number(f"transfer_auc_{r['mode']}_{r.baseline}_{r.target}", round(r.auc_records_vs_bg, 4), "",
                     f"AUC records vs background ({r.target})", "transferability")
        n.text("Methods (draft)", """
Final models were projected onto (i) their own baseline climate, (ii) the other period of the same product (temporal
transfer) and (iii) product-consistent climates for 2041-2060 from five CMIP6 GCMs under SSP2-4.5 and SSP5-8.5. Projections
were computed from the MaxEnt lambda files with clamping to the training range and cloglog output; the implementation
was verified against MaxEnt's own predictions (maximum absolute difference < 0.001). GCM ensemble means and standard
deviations, and the number of GCMs projecting suitable conditions, were derived for each baseline. Uncertainty due to
the occurrence sample was represented by 30 paired bootstrap replicates (identical resamples of records for all
baselines).""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 15 completed.")


if __name__ == "__main__":
    main()
