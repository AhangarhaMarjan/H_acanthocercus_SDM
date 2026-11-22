"""
Step 16 - Extrapolation: MESS, most dissimilar variable (MoD) and extrapolation-aware areas
==========================================================================================

Purpose
    Identify where each projection extrapolates beyond the climate the model was trained on (Elith et al. 2010),
    which variable causes it, and whether differences between baselines persist where NO model extrapolates
    (answers 'is the divergence of future areas an artefact of extrapolation?').

Method (decision D62)
    Reference  training environment of each model = its presence + background SWD values (step 12).
    MESS/MoD   per projection target of step 15 (present, cross_period, every future GCM x SSP; for main/common
               also anchor1970 and published futures); MESS < 0 = at least one predictor outside the training range.
    Areas      main/common: suitable area (p10) inside the main M, (a) all cells, (b) only cells that are analog
               (MESS >= 0) for ALL four baselines for the same target ('common analog domain'); futures averaged
               over the five GCMs. Share of each baseline's suitable area that lies in extrapolated cells.
    Maps       national maps with extrapolated cells greyed (display), M outline, records.

Outputs  data_processed/16_extrapolation_mess/{variant}/{mode}/{baseline}_mess.nc
QA       results/qa/16_extrapolation_mess/  16_report.txt, 16_*.csv, 16_fig_*.png
Manuscript results/manuscript/tables/T16_*.csv, notes/16_extrapolation_mess.md
Run      python scripts/16_extrapolation_mess.py          (main variant)
         python scripts/16_extrapolation_mess.py --all    (adds the sensitivity variants)
"""
import importlib.util
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
from hemiscorpius.mess import mess  # noqa: E402

STEP = "16_extrapolation_mess"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
ISSUES = []

_spec = importlib.util.spec_from_file_location("s15", ROOT / "scripts" / "15_projections.py")
s15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s15)


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import rasterio
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out_root = proc / STEP
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "16_run_info.json", CONFIG_FILE)
    run_all = "--all" in sys.argv
    with rasterio.open(proc / "08_common_mask" / "common_mask.tif") as r_:
        mask = r_.read(1) == 1
    with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
        m_full = r_.read(1) == 1
    in_m = m_full[mask]
    area = cell_area_km2(grid)[mask]
    clim = s15.Climates(proc, cfg, mask)
    summ14 = pd.read_csv(proc / "14_final_models" / "final_models_summary.csv")
    models = summ14 if run_all else summ14[summ14.variant == "main"]
    gcms, ssps = cfg["future"]["gcms"], cfg["future"]["ssps"]

    with Report(qa / "16_report.txt"):
        print(f"Step {STEP}: {len(models)} models")

        section("1. MESS and MoD for every model and projection target")
        rows, mod_rows, store = [], [], {}
        for _, m in models.iterrows():
            v, mode, b = m.variant, m["mode"], m.baseline
            spec = s15.variant_spec(cfg, v)
            src = proc / "12_maxent_inputs" / v / b
            smp, bg = pd.read_csv(src / "samples.csv"), pd.read_csv(src / "background.csv")
            preds = [c for c in smp.columns if c not in ("species", "lon", "lat")]
            ref = {p: np.r_[smp[p].values, bg[p].values] for p in preds}
            main_common = (v == "main" and mode == "common")
            tg = [t for t in s15.targets_for(b, spec["version"], cfg, main_common)]
            das, attrs = [], {}
            for name, s in tg:
                env = s15.env_for(clim, s, preds, b)
                ms, mod, names = mess(ref, env)
                if main_common:
                    store[(b, name)] = ms
                full = np.full(grid.shape, np.nan, "float32")
                full[clim.idx] = ms
                das.append(to_dataarray(full.copy(), grid, f"mess_{name}"))
                full[clim.idx] = mod
                das.append(to_dataarray(full.copy(), grid, f"mod_{name}"))
                neg = ms < 0
                rows.append(dict(variant=v, mode=mode, baseline=b, target=name, share_iran_extrapolated=float(neg.mean()),
                                 share_M_extrapolated=float(neg[in_m].mean())))
                if main_common and name.startswith("fut_present"):
                    vc = pd.Series(np.array(names)[mod[in_m & neg]]).value_counts(normalize=True)
                    for var, sh in vc.items():
                        mod_rows.append(dict(baseline=b, target=name, variable=var, share_of_extrapolated_M=float(sh)))
            ds = xr.merge(das)
            ds.attrs.update(variables=",".join(preds), variant=v, mode=mode, baseline=b)
            od = out_root / v / mode
            od.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(od / f"{b}_mess.nc", encoding=nc_encoding(ds))
            print(f"  {v}/{mode}/{b}: {len(tg)} targets")
        ext = pd.DataFrame(rows)
        ext.to_csv(qa / "16_extrapolation_shares.csv", index=False)

        section("2. Extrapolation inside the main M (main/common)")
        mc = ext[(ext.variant == "main") & (ext["mode"] == "common")].copy()
        mc["group"] = mc.target.str.replace(r"_(BCC-CSM2-MR|EC-Earth3-Veg|IPSL-CM6A-LR|MIROC6|MRI-ESM2-0)_", "_", regex=True)
        g = mc.groupby(["baseline", "group"]).share_M_extrapolated.mean().unstack()
        cols = [c for c in ["present", "cross_period", "fut_present_ssp245", "fut_present_ssp585",
                            "fut_anchor1970_ssp585", "fut_published_ssp585"] if c in g]
        g = g.reindex(list(BASELINES))[cols]
        g.to_csv(tables / "T16_share_M_extrapolated_main.csv")
        print("   share of M with MESS < 0 (futures: mean over 5 GCMs):")
        print("   " + (100 * g).round(1).to_string().replace("\n", "\n   "))
        mods = pd.DataFrame(mod_rows)
        if len(mods):
            mm = mods.groupby(["baseline", "variable"]).share_of_extrapolated_M.mean().unstack().fillna(0)
            mm.to_csv(qa / "16_mod_main_futures.csv")
            print("\n   most dissimilar variable in extrapolated cells of M (main futures, mean share):")
            print("   " + (100 * mm.reindex(list(BASELINES))).round(1).to_string().replace("\n", "\n   "))

        section("3. Is the divergence among baselines caused by extrapolation? (main/common)")
        thr = summ14[(summ14.variant == "main") & (summ14["mode"] == "common")].set_index("baseline").thr_p10
        proj = {}
        for b in BASELINES:
            with xr.open_dataset(proc / "15_projections" / "main" / "common" / f"{b}.nc") as ds:
                proj[b] = {k: ds[k].values[mask] for k in ds.data_vars
                           if k in ("present", "cross_period") or k.startswith("fut_present")}
        res = []
        groups = {"present": ["present"], "cross_period": ["cross_period"]}
        for ssp in ssps:
            groups[f"future_{ssp}"] = [f"fut_present_{gc}_{ssp}" for gc in gcms]
        for grp, tlist in groups.items():
            for b in BASELINES:
                a_all, a_an, a_ext, sh_an = [], [], [], []
                for t in tlist:
                    analog = in_m.copy()
                    for b2 in BASELINES:
                        analog &= store[(b2, t)] >= 0
                    suit = proj[b][t] >= thr[b]
                    a_all.append(area[suit & in_m].sum())
                    a_an.append(area[suit & analog].sum())
                    a_ext.append(area[suit & in_m & (store[(b, t)] < 0)].sum())
                    sh_an.append(area[analog].sum() / area[in_m].sum())
                res.append(dict(target=grp, baseline=b, area_M_km2=np.mean(a_all), area_common_analog_km2=np.mean(a_an),
                                share_suitable_area_extrapolated=np.mean(a_ext) / max(np.mean(a_all), 1),
                                common_analog_share_of_M=np.mean(sh_an)))
        rs = pd.DataFrame(res)
        rs.to_csv(qa / "16_areas_analog_vs_all.csv", index=False)
        rs.round(3).to_csv(tables / "T16_areas_analog_vs_all.csv", index=False)
        for col, lab in (("area_M_km2", "all cells of M"), ("area_common_analog_km2",
                                                           "only cells analog for ALL four baselines")):
            piv = rs.pivot(index="baseline", columns="target", values=col).reindex(list(BASELINES))
            print(f"\n   suitable area p10 (km2), {lab}:")
            print("   " + piv.round(0).to_string().replace("\n", "\n   "))
        sh = rs.groupby("target").common_analog_share_of_M.first()
        print("\n   common analog domain as share of M: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sh.items()))
        for grp in groups:
            r = rs[rs.target == grp]
            ratio_all = r.area_M_km2.max() / max(r.area_M_km2.min(), 1)
            ratio_an = r.area_common_analog_km2.max() / max(r.area_common_analog_km2.min(), 1)
            print(f"   {grp}: max/min area among baselines {ratio_all:.2f} (all M) vs {ratio_an:.2f} (common analog)")

        # ------------------------------------------------------------------ figure
        grey = ListedColormap(["#bdbdbd"])
        ext_ = (grid.west, grid.west + grid.ncols / 24, grid.north - grid.nrows / 24, grid.north)
        occ = pd.read_csv(proc / "09_occurrences_thinning" / "occurrences_all.csv")
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        for i, b in enumerate(BASELINES):
            with xr.open_dataset(proc / "15_projections" / "main" / "common" / f"{b}.nc") as ds:
                pres, f585 = ds["present"].values, ds["ens_present_ssp585_mean"].values
            mp = np.full(grid.shape, np.nan)
            mp[mask] = store[(b, "present")]
            mf = np.full(grid.shape, np.nan)
            mf[mask] = np.min(np.stack([store[(b, f"fut_present_{gc}_ssp585")] for gc in gcms]), axis=0)
            for j, (t, a, mz) in enumerate((("present (grey = extrapolation)", pres, mp),
                                            ("SSP5-8.5 GCM mean (grey = extrapolation in any GCM)", f585, mf))):
                ax = axes[i, j]
                im = ax.imshow(a, cmap="viridis", vmin=0, vmax=1, extent=ext_)
                ax.imshow(np.where(mz < 0, 1.0, np.nan), cmap=grey, extent=ext_)
                ax.contour(grid.lons, grid.lats, m_full.astype(float), levels=[0.5], colors="white", linewidths=0.6)
                ax.scatter(occ.lon, occ.lat, s=3, c="red")
                ax.set_title(f"{b}: {t}", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7)
            ax = axes[i, 2]
            im = ax.imshow(np.clip(mf, -50, 50), cmap="RdBu", vmin=-50, vmax=50, extent=ext_)
            ax.contour(grid.lons, grid.lats, m_full.astype(float), levels=[0.5], colors="black", linewidths=0.6)
            ax.set_title(f"{b}: minimum MESS over GCMs, SSP5-8.5", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        fig.savefig(qa / "16_fig_mess_main.png", dpi=150)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Extrapolation (MESS)")
        for b in BASELINES:
            for c in g.columns:
                n.number(f"share_M_extrapolated_{b}_{c}", round(float(g.loc[b, c]), 4), "fraction",
                         f"share of M with MESS < 0, {b}, {c}", "R1-6")
        for _, r in rs.iterrows():
            n.number(f"area_analog_{r.baseline}_{r.target}", round(r.area_common_analog_km2, 0), "km2",
                     f"suitable area in the common analog domain, {r.baseline}, {r.target}", "extrapolation")
        n.text("Methods (draft)", """
Extrapolation was assessed with multivariate environmental similarity surfaces (MESS) and the most dissimilar variable
(Elith et al. 2010), using the presence and background data of each model as reference. Suitable areas were compared
among baselines both for the whole calibration area and for the 'common analog domain', i.e. the cells where no
baseline model extrapolated, to test whether differences among baselines were caused by extrapolation. Areas with
MESS < 0 are shown in grey on the national maps and are not interpreted.""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 16 completed.")


if __name__ == "__main__":
    main()
