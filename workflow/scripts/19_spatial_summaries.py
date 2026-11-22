"""
Step 19 - Spatial summaries requested by the reviewers or updated from the original manuscript
==============================================================================================

All summaries use the main models (common setting) and the main accessible area M; areas are geodesic (km2).
Every model is projected on the SAME climate sequence of its product (decision D69):
    C1970  = 1970-1999 climate      (own climate for 1970-1999 models, cross-period for 2000-2024 models)
    C2000  = 2000-2024 climate      (cross-period for 1970-1999 models, own climate for 2000-2024 models)
    F245, F585 = 2041-2060 GCM-mean projections (product-consistent, present-anchored)
so that every change has a COMMON reference (R1 minor 5) and contains only the climate change of the transition.

Tables
    T19_A_areas_common_reference     Table 4 replacement: suitable area for C1970, C2000, F245, F585, change vs
                                     C2000 (and C1970 -> C2000), range over the five GCMs  [R1 minor 5, 6]
    T19_B_threshold_sensitivity      for p10, MTP, maxSSS: areas, Jaccard between baselines, net change and its
                                     direction, centroid shift  [R1-5, exactly as requested]
    T19_C_dynamics_centroids         four-class dynamics (stable unsuitable, loss, gain, stable suitable) and
                                     centroid distance / azimuth for C1970->C2000, C2000->F245, C2000->F585
                                     (update of original Figs 3-4 / Table S3; Supplement)
    T19_D_agreement                  agreement of transferred 1970-1999 models with 2000-2024 models on the
                                     2000-2024 climate, and between products (update of original Table 5)
    T19_E_future_constructions       F585 / F245 areas and extrapolated share under product-consistent (main),
                                     1970-anchored and published CMIP6 futures  [R1-6]
    T19_F_trajectories               eight-class trajectories C1970 -> C2000 -> F (update of Table S4; Supplement)

QA       results/qa/19_spatial_summaries/  19_report.txt, 19_fig_*.png
Run      python scripts/19_spatial_summaries.py
"""
import itertools
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
from hemiscorpius.comparison import binary_agreement, spherical_centroid, distance_azimuth, jaccard  # noqa: E402

STEP = "19_spatial_summaries"
BASELINES = ("WC_1970_1999", "WC_2000_2024", "TC_1970_1999", "TC_2000_2024")
THRESHOLDS = ("p10", "mtp", "maxsss")
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors  # noqa: F401
    import matplotlib.patches  # noqa: F401
    import rasterio
    import xarray as xr

    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    write_run_info(qa / "19_run_info.json", CONFIG_FILE)
    gcms, ssps = cfg["future"]["gcms"], cfg["future"]["ssps"]
    with rasterio.open(proc / "10_accessible_area_background" / "M_main.tif") as r_:
        m = r_.read(1) == 1
    mi = np.where(m)
    area = cell_area_km2(grid)[mi]
    lon = grid.lons[mi[1]]
    lat = grid.lats[mi[0]]
    s14 = pd.read_csv(proc / "14_final_models" / "final_models_summary.csv")
    s14 = s14[(s14.variant == "main") & (s14["mode"] == "common")].set_index("baseline")
    thr = {k: s14[f"thr_{k}"] for k in THRESHOLDS}

    # ---------------------------------------------------------------- climate sequence per baseline
    seq, futs, mess_any = {}, {}, {}
    for b in BASELINES:
        with xr.open_dataset(proc / "15_projections" / "main" / "common" / f"{b}.nc") as ds:
            d = {k: ds[k].values[mi] for k in ds.data_vars}
        old = b.endswith("1970_1999")
        seq[b] = {"C1970": d["present"] if old else d["cross_period"],
                  "C2000": d["cross_period"] if old else d["present"],
                  "F245": d["ens_present_ssp245_mean"], "F585": d["ens_present_ssp585_mean"]}
        futs[b] = {k: v for k, v in d.items() if k.startswith("fut_")}
        with xr.open_dataset(proc / "16_extrapolation_mess" / "main" / "common" / f"{b}_mess.nc") as ds:
            for method in ("present", "anchor1970", "published"):
                for ssp in ssps:
                    keys = [f"mess_fut_{method}_{g}_{ssp}" for g in gcms]
                    if all(k in ds for k in keys):
                        mess_any[(b, method, ssp)] = np.any([ds[k].values[mi] < 0 for k in keys], axis=0)
    periods = ("C1970", "C2000", "F245", "F585")

    def suit(b, p, k="p10"):
        return seq[b][p] >= thr[k][b]

    def a(mask_):
        return float(area[mask_].sum())

    def centroid(mask_):
        return spherical_centroid(lon[mask_], lat[mask_], area[mask_]) if mask_.any() else (np.nan, np.nan)

    with Report(qa / "19_report.txt"):
        print(f"Step {STEP}: main models (common setting), inside the main M ({len(area):,} cells, {area.sum():,.0f} km2)")

        # ---------------------------------------------------------------- A
        section("A. Suitable area with a common reference (p10) and range over GCMs  [Table 4; R1 minor 5-6]")
        rows = []
        for b in BASELINES:
            r = {"baseline": b}
            for p in periods:
                r[f"{p}_km2"] = a(suit(b, p))
            r["change_C1970_to_C2000_pct"] = 100 * (r["C2000_km2"] / r["C1970_km2"] - 1)
            for ssp, p in (("ssp245", "F245"), ("ssp585", "F585")):
                r[f"change_C2000_to_{p}_pct"] = 100 * (r[f"{p}_km2"] / r["C2000_km2"] - 1)
                g_area = [a(futs[b][f"fut_present_{g}_{ssp}"] >= thr["p10"][b]) for g in gcms]
                r[f"{p}_gcm_min_km2"], r[f"{p}_gcm_max_km2"] = min(g_area), max(g_area)
                r[f"{p}_gcm_change_min_pct"] = 100 * (min(g_area) / r["C2000_km2"] - 1)
                r[f"{p}_gcm_change_max_pct"] = 100 * (max(g_area) / r["C2000_km2"] - 1)
                for g, ga in zip(gcms, g_area):
                    r[f"{p}_{g}_km2"] = ga
            rows.append(r)
        ta = pd.DataFrame(rows)
        ta.to_csv(qa / "19_A_areas_common_reference.csv", index=False)
        ta.round(1).to_csv(tables / "T19_A_areas_common_reference.csv", index=False)
        show = ["baseline", "C1970_km2", "C2000_km2", "F245_km2", "F585_km2", "change_C1970_to_C2000_pct",
                "change_C2000_to_F245_pct", "change_C2000_to_F585_pct", "F585_gcm_change_min_pct", "F585_gcm_change_max_pct"]
        print("   " + ta[show].round(1).to_string(index=False).replace("\n", "\n   "))

        # ---------------------------------------------------------------- B
        section("B. Threshold sensitivity: areas, Jaccard between baselines, direction of change, centroid  [R1-5]")
        rows = []
        for k in THRESHOLDS:
            for p in ("C2000", "F245", "F585"):
                r = {"threshold": k, "target": p}
                for b in BASELINES:
                    r[f"area_{b}"] = a(suit(b, p, k))
                r["period_effect_km2"] = 0.5 * ((r["area_WC_2000_2024"] - r["area_WC_1970_1999"]) +
                                                (r["area_TC_2000_2024"] - r["area_TC_1970_1999"]))
                r["product_effect_km2"] = 0.5 * ((r["area_TC_1970_1999"] - r["area_WC_1970_1999"]) +
                                                 (r["area_TC_2000_2024"] - r["area_WC_2000_2024"]))
                js = {f"J_{x}_{y}": jaccard(suit(x, p, k), suit(y, p, k)) for x, y in itertools.combinations(BASELINES, 2)}
                r.update(js)
                r["jaccard_min"], r["jaccard_mean"] = min(js.values()), float(np.mean(list(js.values())))
                if p != "C2000":
                    ch = {b: 100 * (a(suit(b, p, k)) / a(suit(b, "C2000", k)) - 1) for b in BASELINES}
                    for b in BASELINES:
                        r[f"change_pct_{b}"] = ch[b]
                        c0, c1 = centroid(suit(b, "C2000", k)), centroid(suit(b, p, k))
                        r[f"centroid_km_{b}"], r[f"azimuth_{b}"] = distance_azimuth(*c0, *c1)
                    r["direction_same_sign_all_baselines"] = bool(len({np.sign(v) for v in ch.values()}) == 1)
                rows.append(r)
        tb = pd.DataFrame(rows)
        tb.to_csv(qa / "19_B_threshold_sensitivity.csv", index=False)
        tb.round(3).to_csv(tables / "T19_B_threshold_sensitivity.csv", index=False)
        show = ["threshold", "target"] + [f"area_{b}" for b in BASELINES] + ["period_effect_km2", "jaccard_min", "jaccard_mean",
                                                                              "direction_same_sign_all_baselines"]
        print("   " + tb[show].round(2).to_string(index=False).replace("\n", "\n   "))
        fut_rows = tb[tb.target != "C2000"]
        note("OK" if fut_rows.direction_same_sign_all_baselines.all() else "WARN",
             "direction of future change identical among baselines under every threshold" if
             fut_rows.direction_same_sign_all_baselines.all() else "direction of future change differs among baselines under some threshold")
        note("OK" if (tb.period_effect_km2 > 0).all() else "WARN",
             f"period effect positive under all thresholds and targets: {bool((tb.period_effect_km2 > 0).all())}")

        # ---------------------------------------------------------------- C
        section("C. Four-class dynamics and centroid shifts (p10)  [original Figs 3-4 / Table S3 -> Supplement]")
        rows = []
        for b in BASELINES:
            for p0, p1 in (("C1970", "C2000"), ("C2000", "F245"), ("C2000", "F585")):
                s0, s1 = suit(b, p0), suit(b, p1)
                c0, c1 = centroid(s0), centroid(s1)
                dist, az = distance_azimuth(*c0, *c1)
                rows.append(dict(baseline=b, transition=f"{p0}->{p1}", stable_unsuitable_km2=a(~s0 & ~s1),
                                 loss_km2=a(s0 & ~s1), gain_km2=a(~s0 & s1), stable_suitable_km2=a(s0 & s1),
                                 centroid_from_lon=c0[0], centroid_from_lat=c0[1], centroid_to_lon=c1[0],
                                 centroid_to_lat=c1[1], centroid_shift_km=dist, azimuth_deg=az))
        tc = pd.DataFrame(rows)
        tc.to_csv(qa / "19_C_dynamics_centroids.csv", index=False)
        tc.round(2).to_csv(tables / "T19_C_dynamics_centroids.csv", index=False)
        print("   " + tc[["baseline", "transition", "loss_km2", "gain_km2", "stable_suitable_km2", "centroid_shift_km",
                          "azimuth_deg"]].round(1).to_string(index=False).replace("\n", "\n   "))

        # ---------------------------------------------------------------- D
        section("D. Agreement on the 2000-2024 climate (p10, area-weighted)  [original Table 5]")
        pairs = [("WC_2000_2024", "WC_1970_1999", "period (WorldClim): transferred 1970-1999 model vs 2000-2024 model"),
                 ("TC_2000_2024", "TC_1970_1999", "period (TerraClimate): transferred 1970-1999 model vs 2000-2024 model"),
                 ("WC_2000_2024", "TC_2000_2024", "product (2000-2024): TerraClimate vs WorldClim model"),
                 ("WC_1970_1999", "TC_1970_1999", "product (1970-1999): TerraClimate vs WorldClim model")]
        rows = []
        for ref, test, lab in pairs:
            r = dict(comparison=lab, reference=ref, test=test)
            r.update(binary_agreement(suit(ref, "C2000"), suit(test, "C2000"), area))
            rows.append(r)
        td = pd.DataFrame(rows)
        td.to_csv(qa / "19_D_agreement.csv", index=False)
        td.round(3).to_csv(tables / "T19_D_agreement.csv", index=False)
        print("   " + td.drop(columns=["reference", "test"]).round(3).to_string(index=False).replace("\n", "\n   "))

        # ---------------------------------------------------------------- E
        section("E. Future suitable area under three constructions of future climate (p10)  [R1-6]")
        rows = []
        for b in BASELINES:
            for method in ("present", "anchor1970", "published"):
                for ssp in ssps:
                    keys = [f"fut_{method}_{g}_{ssp}" for g in gcms]
                    if not all(k in futs[b] for k in keys):
                        continue
                    ens = np.mean([futs[b][k] for k in keys], axis=0)
                    s_ = ens >= thr["p10"][b]
                    ext = mess_any.get((b, method, ssp))
                    rows.append(dict(baseline=b, construction={"present": "product-consistent, 2000-2024 anchored (main)",
                                                               "anchor1970": "product-consistent, 1970-1999 anchored",
                                                               "published": "published WorldClim CMIP6"}[method],
                                     ssp=ssp, area_km2=a(s_), change_vs_C2000_pct=100 * (a(s_) / a(suit(b, "C2000")) - 1),
                                     share_suitable_extrapolated=float(area[s_ & ext].sum() / max(a(s_), 1)) if ext is not None else np.nan))
        te = pd.DataFrame(rows)
        te.to_csv(qa / "19_E_future_constructions.csv", index=False)
        te.round(3).to_csv(tables / "T19_E_future_constructions.csv", index=False)
        piv = te[te.ssp == "ssp585"].pivot(index="baseline", columns="construction", values="area_km2").reindex(list(BASELINES))
        print("   SSP5-8.5 suitable area (km2):")
        print("   " + piv.round(0).to_string().replace("\n", "\n   "))

        # ---------------------------------------------------------------- F
        section("F. Eight-class trajectories C1970 -> C2000 -> future (p10)  [Table S4 update]")
        rows = []
        for b in BASELINES:
            for p in ("F245", "F585"):
                code = suit(b, "C1970").astype(int) * 100 + suit(b, "C2000").astype(int) * 10 + suit(b, p).astype(int)
                for c in (0, 1, 10, 11, 100, 101, 110, 111):
                    rows.append(dict(baseline=b, future=p, trajectory=f"{c // 100}->{(c // 10) % 10}->{c % 10}",
                                     area_km2=a(code == c)))
        tf = pd.DataFrame(rows).pivot_table(index=["future", "trajectory"], columns="baseline", values="area_km2")
        tf = tf[list(BASELINES)]
        tf.to_csv(qa / "19_F_trajectories.csv")
        tf.round(0).to_csv(tables / "T19_F_trajectories.csv")
        print("   " + tf.round(0).to_string().replace("\n", "\n   "))

        # ---------------------------------------------------------------- figure (QA preview of a possible figure)
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#D55E00", "#0072B2", "#009E73"])
        for j, b in enumerate(BASELINES):
            for i, p in enumerate(("F245", "F585")):
                s0, s1 = suit(b, "C2000"), suit(b, p)
                cls = np.full(grid.shape, np.nan)
                cls[mi] = np.select([~s0 & ~s1, s0 & ~s1, ~s0 & s1, s0 & s1], [0, 1, 2, 3])
                ax = axes[i, j]
                ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=3.5,
                          extent=(grid.west, grid.west + grid.ncols / 24, grid.north - grid.nrows / 24, grid.north))
                c0, c1 = centroid(s0), centroid(s1)
                ax.annotate("", xy=c1, xytext=c0, arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
                ax.set_xlim(50.5, 61)
                ax.set_ylim(24.8, 30.5)
                ax.set_title(f"{b}: 2000-2024 -> {p}", fontsize=9)
        handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
                   zip(cmap.colors, ["stable unsuitable", "loss", "gain", "stable suitable"])]
        fig.legend(handles=handles, loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(qa / "19_fig_dynamics_preview.png", dpi=150)
        plt.close(fig)

        n = ManuscriptNotes(cfg, STEP, "Spatial summaries")
        for _, r in ta.iterrows():
            for p in periods:
                n.number(f"area_{r.baseline}_{p}", round(r[f"{p}_km2"], 0), "km2", f"suitable area (p10) {p}", "R1 minor 5")
        n.text("Methods (draft)", """
To use a common reference for all comparisons, each model was projected onto both baseline periods of its climate product
and onto the corresponding future climates, and changes were expressed relative to the projection on the 2000-2024
climate. Suitable areas, Jaccard agreement, direction of change and centroid shifts were compared inside the
calibration area for the 10th-percentile, minimum training presence and maximum training sensitivity plus specificity
thresholds. Detailed four-class dynamics (stable unsuitable, loss, gain, stable suitable) and eight-class
trajectories were computed for the 10th-percentile threshold. Centroids used area-weighted spherical coordinates;
shifts used geodesic distance and azimuth. Agreement between binary maps was summarised with area-weighted
confusion-matrix statistics.""")
        n.save()

        section("SUMMARY")
        fails = [x for lvl, x in ISSUES if lvl == "FAIL"]
        warns = [x for lvl, x in ISSUES if lvl == "WARN"]
        print(f"  FAIL: {len(fails)}   WARN: {len(warns)}")
        for x in fails + warns:
            print(f"   - {x}")
        if not fails:
            print("\n  Step 19 completed.")


if __name__ == "__main__":
    main()
