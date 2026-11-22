"""
Step 05 - Product-consistent future climate (2041-2060) by the delta method
===========================================================================

Purpose
    Build future monthly climatologies and bioclimatic variables for every climate product, GCM,
    SSP and data version, so that each model is projected onto a future consistent with the product
    it was calibrated on (reviewer R1-6), starting from the most recent observed climate.

Methods (decisions fixed before any model was fitted)
    present (MAIN, D36)
        future = baseline 2000-2024 of the product + GCM change from the baseline mid-year (~2012)
        to the future mid-year (~2051). The GCM state at ~2012 is interpolated linearly between the
        WorldClim 2.1 normals (1970-2000; the GCM layers are bias-corrected to these, so they represent
        the GCM state at ~1985) and the GCM period 2021-2040 (~2031):
            G_2012 = N + w (F_2031 - N),  w = (2012.5 - 1985.5) / (2031.0 - 1985.5)
        Temperature: future = P2000 + (F_2051 - G_2012)
        Precipitation: future = P2000 x (F_2051 + 1) / (G_2012 + 1), ratio capped (D30)
        This keeps the observed present climate (including the observed drying of the south) and adds
        only the change the GCMs project from the present onwards, without double counting.
    anchor1970 (SENSITIVITY, D29)
        future = baseline 1970-1999 + (F_2051 - N). Equivalent to replacing the observed change since
        ~1985 by the GCM change; shown in the first run to re-introduce the wetter 1970-1999 regime.
    published (SENSITIVITY)
        WorldClim CMIP6 2041-2060 layers as published (approach of the original manuscript).
    Both models of a product (calibrated on 1970-1999 and 2000-2024) are projected onto the same future.

Checks
    1. Implementation: WorldClim 'anchor1970' future vs published layers (must be nearly identical).
    2. Interpolated GCM change 1985->2012 vs observed change in each product (temperature should agree;
       precipitation differences reveal observed variability not represented by the GCMs).
    3. Share of precipitation ratios reaching the cap; cells with tmax < tmin.
    4. Projected change at occurrence records and spread among the five GCMs (R1 minor 6).
    5. Product-consistent future vs published layers (size of the R1-6 issue); 'present' vs 'anchor1970'.
    6. Changes of the wettest quarter in the future (bio8 decision).

Set names  {WC|TC}_future_{present|anchor1970}_{gcm}_{ssp}_2041_2060_{original|cleaned}
           CMIP6_future_published_{gcm}_{ssp}_2041_2060
Ensembles  set name without the GCM, e.g. TC_future_present_ssp585_2041_2060_original_ensemble
Outputs  data_processed/05_future_delta/{set}_bioclim.nc, monthly/{set}.nc, ensemble/*.nc
QA       results/qa/05_future_delta/  05_report.txt, 05_run_info.json, 05_*.csv, 05_fig_*.png
Manuscript results/manuscript/tables/T05_*.csv, results/manuscript/notes/05_future_delta.md
Run      python scripts/05_future_delta.py     (delete data_processed/05_future_delta first if it
                                                 contains files from an earlier naming scheme)
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemiscorpius.config import load_config, project_path, qa_dir, CONFIG_FILE  # noqa: E402
from hemiscorpius.report import Report, section, ok, warn, fail, write_run_info  # noqa: E402
from hemiscorpius.manuscript import ManuscriptNotes  # noqa: E402
from hemiscorpius.grid import ref_grid, to_dataarray, nc_encoding, write_geotiff  # noqa: E402
from hemiscorpius.occurrences import load_occurrences, cell_indices  # noqa: E402
from hemiscorpius.delta import apply_delta  # noqa: E402
from hemiscorpius.precip import wettest_quarter_start  # noqa: E402

STEP = "05_future_delta"
VARS = ("tmin", "tmax", "prec")
PRODUCTS = ("WC", "TC")
VERSIONS = ("original", "cleaned")
BIO = [f"bio{i}" for i in range(1, 20)]
KEY_BIO = ["bio1", "bio4", "bio12", "bio14", "bio15", "bio17"]
DEFAULTS = {"precip_offset_mm": 1.0, "precip_ratio_cap": 3.0, "precip_ratio_cap_sensitivity": 5.0,
            "cap_trigger_share": 0.01}
ISSUES = []


def note(level, msg):
    {"OK": ok, "WARN": warn, "FAIL": fail}[level](msg)
    if level != "OK":
        ISSUES.append((level, msg))


def load_nc(path, names):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return {v: ds[v].values.astype("float32") for v in names}


def save_nc(path, data, grid, attrs, monthly):
    import xarray as xr
    if monthly:
        ds = xr.Dataset({v: to_dataarray(data[v], grid, v, "month", np.arange(1, 13),
                                         "mm/month" if v == "prec" else "degC") for v in VARS})
    else:
        ds = xr.Dataset({v: to_dataarray(data[v], grid, v) for v in data})
    ds.attrs.update(attrs)
    ds.to_netcdf(path, encoding=nc_encoding(ds))


def mid_year(period):
    a, b = (int(x) for x in str(period).replace("_", "-").split("-"))
    return (a + b + 1) / 2.0


def parse(name):
    """Split a set name into its parts (product, method, gcm, ssp, version)."""
    p = name.split("_")
    product, method, gcm, ssp = p[0], p[2], p[3], p[4]
    version = "_".join(p[7:]) if len(p) > 7 else "published"
    return dict(product=product, method=method, gcm=gcm, ssp=ssp, version=version)


# --------------------------------------------------------------------------- futures
def build_and_write(cfg, proc, out, grid, mask, rule, cap, keep):
    """Build all future sets, write R inputs, keep in memory only the sets needed for QA."""
    fut = cfg["future"]
    per, per_i = fut["period"].replace("-", "_"), fut["interpolation_period"].replace("-", "_")
    w = (mid_year("2000-2024") - mid_year(cfg["reference_normals"]["period"])) / \
        (mid_year(fut["interpolation_period"]) - mid_year(cfg["reference_normals"]["period"]))
    off = rule["precip_offset_mm"]
    suffix = "" if cap == rule["precip_ratio_cap"] else f"_cap{cap:g}"
    methods = [fut["delta"].get("main_method", "present")] + list(fut["delta"].get("sensitivity_methods", []))
    normals = load_nc(proc / "02_crop_and_units" / "wc21_normals_1970_2000.nc", VARS)
    base = {(p, per_, v): load_nc(proc / "03_monthly_climatologies" / f"{p}_{per_}_{v}.nc", VARS)
            for p in PRODUCTS for per_ in ("1970_1999", "2000_2024") for v in VERSIONS}
    rin = out / "r_inputs"
    rin.mkdir(parents=True, exist_ok=True)
    (out / "monthly").mkdir(exist_ok=True)
    names, kept, cap_rows, gcm_change = [], {}, [], {}
    inside3 = None

    def emit(name, clim, attrs):
        clim = {k: np.asarray(a, dtype="float32") for k, a in clim.items()}
        for v in VARS:
            write_geotiff(rin / f"{name}_{v}.tif", clim[v], grid)
        save_nc(out / "monthly" / f"{name}.nc", clim, grid, attrs, monthly=True)
        names.append(name)
        if keep(name):
            kept[name] = clim

    for gcm in fut["gcms"]:
        for ssp in fut["ssps"]:
            f50 = load_nc(proc / "02_crop_and_units" / "cmip6" / f"{gcm}_{ssp}_{per}.nc", VARS)
            f30 = load_nc(proc / "02_crop_and_units" / "cmip6" / f"{gcm}_{ssp}_{per_i}.nc", VARS)
            g12 = {v: normals[v] + w * (f30[v] - normals[v]) for v in VARS}
            g12["prec"] = np.maximum(g12["prec"], 0.0)
            gcm_change[(gcm, ssp)] = {v: g12[v] - normals[v] for v in VARS}
            if inside3 is None:
                inside3 = np.broadcast_to(mask, f50["prec"].shape)
            if not suffix:
                emit(f"CMIP6_future_published_{gcm}_{ssp}_{per}", f50,
                     dict(method="WorldClim CMIP6 as published", gcm=gcm, ssp=ssp))
            for method in methods:
                ref, bper = (g12, "2000_2024") if method == "present" else (normals, "1970_1999")
                raw = (f50["prec"] + off) / (ref["prec"] + off)
                capped = np.isfinite(raw) & (raw > cap)
                cap_rows.append(dict(method=method, gcm=gcm, ssp=ssp, cap=cap,
                                     share_cell_months_capped_iran=float(capped[inside3].mean()),
                                     max_raw_ratio_iran=float(np.nanmax(np.where(inside3, raw, np.nan)))))
                for p in PRODUCTS:
                    for v in VERSIONS:
                        if method != "present" and v != "original":
                            continue                               # sensitivity methods: original only
                        clim, _ = apply_delta(base[(p, bper, v)], ref, f50, off, cap)
                        emit(f"{p}_future_{method}_{gcm}_{ssp}_{per}_{v}{suffix}", clim,
                             dict(method=method, product=p, gcm=gcm, ssp=ssp, version=v,
                                  baseline=bper, interpolation_weight=w if method == "present" else 0.0,
                                  precip_offset_mm=off, precip_ratio_cap=cap))
    (rin / "sets.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return names, kept, pd.DataFrame(cap_rows), gcm_change, w, base


def run_biovars(cfg, out, names):
    import rasterio
    rin, rout = out / "r_inputs", out / "r_outputs"
    rout.mkdir(exist_ok=True)
    res = subprocess.run([cfg["paths"]["rscript"], str(ROOT / "scripts" / "04_bioclim_biovars.R"),
                          str(rin), str(rout)], capture_output=True, text=True)
    lines = [ln for ln in (res.stdout + res.stderr).splitlines() if ln.strip()]
    for ln in lines[:2] + ["..."] + lines[-2:]:
        print(f"    R> {ln}")
    if res.returncode != 0:
        raise RuntimeError(f"Rscript failed (exit {res.returncode})")
    bio = {}
    for s in names:
        with rasterio.open(rout / f"{s}_bio.tif") as src:
            a = src.read(masked=True).astype("float32").filled(np.nan)
        bio[s] = {f"bio{i + 1}": a[i] for i in range(19)}
    return bio


def save_bioclim_and_ensembles(bio, out, grid):
    (out / "ensemble").mkdir(exist_ok=True)
    groups = {}
    for s, b in bio.items():
        save_nc(out / f"{s}_bioclim.nc", b, grid, dict(set=s), monthly=False)
        p = s.split("_")
        groups.setdefault("_".join(p[:3] + p[4:]), []).append(s)
    for g, members in groups.items():
        ens = {}
        for v in BIO:
            st = np.stack([bio[m][v] for m in members])
            ens[f"{v}_mean"], ens[f"{v}_sd"] = st.mean(axis=0), st.std(axis=0, ddof=1)
        save_nc(out / "ensemble" / f"{g}_ensemble_bioclim.nc", ens, grid,
                dict(group=g, members=",".join(parse(m)["gcm"] for m in members)), monthly=False)
    print(f"  saved {len(bio)} bioclim sets and {len(groups)} GCM ensembles")


# --------------------------------------------------------------------------- QA
def check_implementation(kept, bio, mask, per, qa):
    section("3. Implementation check: WorldClim anchor1970 future vs published WorldClim CMIP6")
    rows = []
    for s in kept:
        if not (s.startswith("WC_future_anchor1970") and s.endswith("_original")):
            continue
        d = parse(s)
        direct = f"CMIP6_future_published_{d['gcm']}_{d['ssp']}_{per}"
        for v in VARS:
            a, b = kept[s][v], kept[direct][v]
            ok_ = np.broadcast_to(mask, a.shape) & np.isfinite(a) & np.isfinite(b)
            rows.append(dict(set=s, variable=v, mae=float(np.abs(a - b)[ok_].mean()),
                             r=float(np.corrcoef(a[ok_], b[ok_])[0, 1])))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "05_check_anchor1970_vs_published.csv", index=False)
    t, p = df[df.variable != "prec"], df[df.variable == "prec"]
    print(f"  temperature MAE <= {t.mae.max():.3f} degC; precipitation r >= {p.r.min():.4f}")
    note("OK" if t.mae.max() < 0.2 and p.r.min() > 0.98 else "WARN",
         "delta implementation reproduces the published WorldClim CMIP6 layers")
    return df


def gcm_vs_observed_change(gcm_change, base, mask, occ, qa, grid):
    section("4. Interpolated GCM change (~1985 -> ~2012) vs observed change (1970-1999 -> 2000-2024)")
    r, c = occ["row"].values, occ["col"].values
    w = np.cos(np.deg2rad(grid.lats))[:, None] * mask
    rows = []
    for p in PRODUCTS:
        obs = {v: base[(p, "2000_2024", "original")][v] - base[(p, "1970_1999", "original")][v] for v in VARS}
        for (gcm, ssp), ch in gcm_change.items():
            for v in VARS:
                if v == "prec":
                    o, g = obs[v].sum(axis=0), ch[v].sum(axis=0)          # annual totals
                else:
                    o, g = obs[v].mean(axis=0), ch[v].mean(axis=0)        # annual means
                ok_ = np.isfinite(o) & np.isfinite(g) & (w > 0)
                rows.append(dict(product=p, gcm=gcm, ssp=ssp, variable=v,
                                 observed_iran=float((o * w)[ok_].sum() / w[ok_].sum()),
                                 gcm_iran=float((g * w)[ok_].sum() / w[ok_].sum()),
                                 observed_occ=float(np.nanmean(o[r, c])), gcm_occ=float(np.nanmean(g[r, c]))))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "05_gcm_vs_observed_change_1985_2012.csv", index=False)
    summ = df.groupby(["product", "variable"])[["observed_iran", "gcm_iran", "observed_occ", "gcm_occ"]].mean()
    print("   " + summ.round(2).to_string().replace("\n", "\n   "))
    return df


def projected_changes(bio, base_bio, occ, qa, tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    section("6. Projected change at occurrence records (vs 2000-2024 of the same product) and GCM spread")
    r, c = occ["row"].values, occ["col"].values
    rows = []
    for s, b in bio.items():
        d = parse(s)
        prod = "WC" if d["product"] == "CMIP6" else d["product"]
        ver = "original" if d["version"] == "published" else d["version"].split("_")[0]
        ref = base_bio[f"{prod}_2000_2024_{ver}"]
        for v in BIO:
            rows.append(dict(set=s, **d, variable=v,
                             change_vs_2000_2024_occ_mean=float(np.nanmean(b[v][r, c] - ref[v][r, c]))))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "05_future_change_by_gcm.csv", index=False)
    df.round(4).to_csv(tables / "T05_future_change_by_gcm.csv", index=False)
    sub = df[df.version.isin(["original", "published"])]
    spread = (sub.groupby(["product", "method", "ssp", "variable"])["change_vs_2000_2024_occ_mean"]
              .agg(["mean", "std", "min", "max"]).reset_index())
    spread.to_csv(qa / "05_gcm_spread_at_occurrences.csv", index=False)
    spread.round(4).to_csv(tables / "T05_gcm_spread_at_occurrences.csv", index=False)
    show = spread[spread.variable.isin(KEY_BIO)].round(2)
    print("   " + show.to_string(index=False).replace("\n", "\n   "))

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    combos = [(p, m) for m in ("present", "anchor1970", "published") for p in ("WC", "TC", "CMIP6")
              if (sub["product"].eq(p) & sub.method.eq(m)).any()]
    for ax, v in zip(axes.ravel(), KEY_BIO):
        groups, labels = [], []
        for p, m in combos:
            for ssp in sorted(sub.ssp.unique()):
                vals = sub[(sub["product"] == p) & (sub.method == m) & (sub.ssp == ssp) & (sub.variable == v)]
                groups.append(vals["change_vs_2000_2024_occ_mean"].values)
                labels.append(f"{p}\n{m}\n{ssp}")
        try:
            ax.boxplot(groups, tick_labels=labels, showmeans=True)
        except TypeError:
            ax.boxplot(groups, labels=labels, showmeans=True)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{v}: change vs 2000-2024 at occurrences (5 GCMs)", fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(qa / "05_fig_gcm_spread_key_bioclim.png", dpi=200)
    plt.close(fig)
    return df, spread


def method_comparisons(bio, occ, mask, per, qa, tables, sd_ref):
    section("7. Size of the R1-6 issue and of the baseline-anchoring choice (at occurrence records)")
    r, c = occ["row"].values, occ["col"].values
    rows = []
    for s in bio:
        d = parse(s)
        if d["product"] == "CMIP6" or d["version"] != "original":
            continue
        published = f"CMIP6_future_published_{d['gcm']}_{d['ssp']}_{per}"
        other = s.replace("_present_", "_anchor1970_") if d["method"] == "present" else None
        for v in BIO:
            x = bio[s][v][r, c]
            rec = dict(product=d["product"], method=d["method"], gcm=d["gcm"], ssp=d["ssp"], variable=v,
                       std_abs_diff_vs_published=float(np.nanmean(np.abs(x - bio[published][v][r, c]))) / sd_ref[v])
            if other and other in bio:
                rec["std_abs_diff_present_vs_anchor1970"] = \
                    float(np.nanmean(np.abs(x - bio[other][v][r, c]))) / sd_ref[v]
            rows.append(rec)
    df = pd.DataFrame(rows)
    summ = df.groupby(["product", "method", "variable"]).mean(numeric_only=True).reset_index()
    summ.to_csv(qa / "05_method_comparisons.csv", index=False)
    summ.round(4).to_csv(tables / "T05_method_comparisons.csv", index=False)
    top = summ[(summ["product"] == "TC") & (summ.method == "present")].sort_values(
        "std_abs_diff_vs_published", ascending=False).head(6)
    print("  TC 'present' future vs published WorldClim CMIP6 (mean |difference| / Iran SD), largest:")
    print("   " + top.round(3).to_string(index=False).replace("\n", "\n   "))
    return summ


def future_quarters(kept, base, occ, mask, qa):
    section("8. Wettest quarter in the future vs 2000-2024 (bio8 decision)")
    r, c = occ["row"].values, occ["col"].values
    rows = []
    for s, clim in kept.items():
        d = parse(s)
        if d["method"] != "present" or d["version"] != "original":
            continue
        wq_f = wettest_quarter_start(clim["prec"])
        wq_b = wettest_quarter_start(base[(d["product"], "2000_2024", "original")]["prec"])
        diff = mask & np.isfinite(wq_f) & (wq_f != wq_b)
        rows.append(dict(set=s, iran_share_changed=float(diff.sum() / mask.sum()),
                         occurrence_records_changed=int(diff[r, c].sum())))
    df = pd.DataFrame(rows)
    df.to_csv(qa / "05_future_wettest_quarter_changes.csv", index=False)
    if len(df):
        print(f"  wettest quarter differs from 2000-2024: Iran share {df.iran_share_changed.min():.2f}-"
              f"{df.iran_share_changed.max():.2f}; occurrence records {df.occurrence_records_changed.min()}-"
              f"{df.occurrence_records_changed.max()}")
    return df


def notes_05(cfg, rule, w, cap_df, chk, gvo, spread, comp, fq):
    n = ManuscriptNotes(cfg, STEP, "Product-consistent future climate (delta method)")
    n.number("interpolation_weight", round(w, 4), "", "weight of the 2021-2040 GCM change at the baseline mid-year", "D36")
    n.number("cap_share_max", float(cap_df.share_cell_months_capped_iran.max()), "fraction",
             "largest share of Iran cell-months with capped precipitation ratio", "D30")
    n.number("check_temp_mae_max", float(chk[chk.variable != "prec"].mae.max()), "degC",
             "anchor1970 WorldClim future vs published layers: largest temperature MAE", "QA")
    n.number("check_prec_r_min", float(chk[chk.variable == "prec"].r.min()), "",
             "anchor1970 WorldClim future vs published layers: lowest precipitation r", "QA")
    g = gvo.groupby(["product", "variable"])[["observed_occ", "gcm_occ"]].mean()
    for (p, v), row in g.iterrows():
        n.number(f"observed_change_occ_{p}_{v}", round(float(row.observed_occ), 3), "",
                 f"{p} {v}: observed change 1970-1999 -> 2000-2024 at occurrences", "D36")
        n.number(f"gcm_change_1985_2012_occ_{p}_{v}", round(float(row.gcm_occ), 3), "",
                 f"{v}: interpolated GCM change ~1985 -> ~2012 at occurrences (mean of GCMs/SSPs)", "D36")
    for _, row in spread[spread.variable.isin(KEY_BIO)].iterrows():
        k = f"{row['product']}_{row.method}_{row.ssp}_{row.variable}"
        n.number(f"change_occ_mean_{k}", round(float(row["mean"]), 4), "",
                 f"{k}: mean change vs 2000-2024 at occurrences (5 GCMs)", "R1 minor 6")
        n.number(f"change_occ_gcm_sd_{k}", round(float(row["std"]), 4), "", f"{k}: SD among GCMs", "R1 minor 6")
    for _, row in comp[comp.method == "present"].iterrows():
        n.number(f"present_vs_published_std_{row['product']}_{row.variable}",
                 round(float(row.std_abs_diff_vs_published), 4), "SD units",
                 f"{row['product']} {row.variable}: product-consistent future vs published layers", "R1-6")
    if len(fq):
        n.number("future_wettest_quarter_occ_changed_max", int(fq.occurrence_records_changed.max()), "records",
                 "records whose wettest quarter changes from 2000-2024 to 2041-2060", "bio8 decision")
    n.text("Methods (draft)", f"""
Future climate (2041-2060; SSP2-4.5 and SSP5-8.5; five CMIP6 GCMs) was built separately for each product
by adding to its 2000-2024 baseline the change projected by each GCM from the baseline mid-year to the
future mid-year. Because the WorldClim CMIP6 layers are bias-corrected to the WorldClim 2.1 normals
(1970-2000), the GCM state at the baseline mid-year was interpolated linearly between the normals and the
2021-2040 GCM layers (weight {w:.3f}). Temperature changes were additive; precipitation changes were ratios
(P_future + {rule['precip_offset_mm']:g} mm) / (P_reference + {rule['precip_offset_mm']:g} mm), capped at {rule['precip_ratio_cap']:g}.
This anchors projections on the most recently observed climate and avoids double counting of the change
already observed. Two sensitivity futures were built: changes added to the 1970-1999 baseline (which
replaces the observed change since ~1985 by the GCM change) and the published WorldClim CMIP6 layers
(approach of the original analysis). Applied to WorldClim, the 1970-1999-anchored procedure reproduced the
published layers, confirming the implementation.""")
    n.save()


# --------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    grid = ref_grid(cfg)
    proc = project_path(cfg, "processed")
    out = proc / STEP
    out.mkdir(parents=True, exist_ok=True)
    qa = qa_dir(cfg, STEP)
    tables = project_path(cfg, "results") / "manuscript" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_run_info(qa / "05_run_info.json", CONFIG_FILE)
    rule = {**DEFAULTS, **{k: v for k, v in cfg["future"].get("delta", {}).items() if k in DEFAULTS}}
    rule = {k: float(v) for k, v in rule.items()}
    per = cfg["future"]["period"].replace("-", "_")

    with Report(qa / "05_report.txt"):
        print(f"Step {STEP}")
        import rasterio
        with rasterio.open(proc / "02_crop_and_units" / "reference_grid.tif") as r_:
            mask = r_.read(1) == 1
        occ = cell_indices(load_occurrences(cfg), grid)

        def keep(name):          # sets kept in memory for QA
            d = parse(name)
            return (d["version"] in ("original", "published")) and \
                   (d["product"] in ("CMIP6",) or d["method"] in ("present", "anchor1970"))

        section("1. Building futures")
        names, kept, cap_df, gcm_change, w, base = build_and_write(
            cfg, proc, out, grid, mask, rule, rule["precip_ratio_cap"], keep)
        print(f"  {len(names)} sets; interpolation weight w = {w:.3f}")
        share = float(cap_df.share_cell_months_capped_iran.max())
        if share > rule["cap_trigger_share"]:
            note("WARN", f"cap reached in {share:.2%} of cell-months: sensitivity cap "
                         f"{rule['precip_ratio_cap_sensitivity']:g} built")
            n2, k2, c2, _, _, _ = build_and_write(cfg, proc, out, grid, mask, rule,
                                                  rule["precip_ratio_cap_sensitivity"], lambda s: False)
            names += n2
            cap_df = pd.concat([cap_df, c2], ignore_index=True)
            (out / "r_inputs" / "sets.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        else:
            note("OK", f"precipitation cap reached in at most {share:.3%} of Iran cell-months")
        cap_df.to_csv(qa / "05_precip_ratio_cap_statistics.csv", index=False)
        tmax_lt = sum(int((np.broadcast_to(mask, k["tmax"].shape) & (k["tmax"] < k["tmin"])).sum())
                      for k in kept.values())
        note("OK" if tmax_lt == 0 else "WARN", f"cell-months with tmax < tmin (sets kept for QA): {tmax_lt}")

        section("2. dismo::biovars")
        bio = run_biovars(cfg, out, names)
        save_bioclim_and_ensembles(bio, out, grid)

        base_bio = {f"{p}_{pp}_{v}": load_nc(proc / "04_bioclim" / f"{p}_{pp}_{v}_bioclim.nc", BIO)
                    for p in PRODUCTS for pp in ("1970_1999", "2000_2024") for v in VERSIONS}
        sd_ref = {v: float(np.nanstd(base_bio["WC_1970_1999_original"][v][mask])) for v in BIO}

        res = {}
        for key, fn, args in (("chk", check_implementation, (kept, bio, mask, per, qa)),
                              ("gvo", gcm_vs_observed_change, (gcm_change, base, mask, occ, qa, grid)),
                              ("chg", projected_changes, (bio, base_bio, occ, qa, tables)),
                              ("comp", method_comparisons, (bio, occ, mask, per, qa, tables, sd_ref)),
                              ("fq", future_quarters, (kept, base, occ, mask, qa))):
            try:
                res[key] = fn(*args)
            except Exception as e:
                note("FAIL", f"{fn.__name__} crashed: {type(e).__name__}: {e}")
        try:
            notes_05(cfg, rule, w, cap_df, res["chk"], res["gvo"], res["chg"][1], res["comp"], res["fq"])
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
            print("\n  Step 05 completed. Review figures and tables in results/qa/05_future_delta/.")


if __name__ == "__main__":
    main()
