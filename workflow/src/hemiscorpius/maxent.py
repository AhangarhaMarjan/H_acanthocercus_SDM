"""Thin wrapper around maxent.jar 3.4.4 (samples-with-data mode)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SPECIES = "Hacanthocercus"
PRED_DECIMALS = 6          # common precision of MaxEnt predictions before computing metrics (D55)
FEATURE_FLAGS = {"L": "linear", "Q": "quadratic", "H": "hinge", "P": "product", "T": "threshold"}


def resolve_java(java="java"):
    """Return a usable java executable: the configured one, the one on PATH, or a standard install."""
    import glob
    if java and java != "java" and Path(java).exists():
        return str(java)
    found = shutil.which("java")
    if found:
        return found
    patterns = ["C:/Program Files/Java/*/bin/java.exe", "C:/Program Files (x86)/Java/*/bin/java.exe",
                "C:/Program Files/Eclipse Adoptium/*/bin/java.exe", "/usr/bin/java", "/usr/local/bin/java"]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError("Java not found: set models.java in config.yaml to the full path of java.exe")


def java_version(java):
    res = subprocess.run([java, "-version"], capture_output=True, text=True)
    return (res.stderr or res.stdout).splitlines()[0] if (res.stderr or res.stdout) else "unknown"


def feature_args(fc):
    fc = fc.upper()
    unknown = set(fc) - set(FEATURE_FLAGS)
    if unknown:
        raise ValueError(f"unknown feature letters {unknown} in {fc}")
    return [f"{name}={'true' if letter in fc else 'false'}" for letter, name in FEATURE_FLAGS.items()]


def run_maxent(jar, outdir, samples, background, fc, beta, test=None, output="cloglog",
               java="java", memory_mb=1024, extra=None, analysis=False, replicates=1,
               replicatetype="bootstrap", jackknife=None):
    """Run one MaxEnt model in SWD mode; returns the output directory. Raises on failure.

    analysis=True switches on response curves (with plot data) and jackknife (final models, step 14).
    replicates > 1 runs MaxEnt's own replicates (e.g. bootstrap) with its fixed default random seed."""
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    args = [java, f"-mx{memory_mb}m", "-jar", str(jar), "nowarnings", "noprefixes",
            f"environmentallayers={background}", f"samplesfile={samples}", f"outputdirectory={outdir}",
            "autorun", "visible=false", "redoifexists", "askoverwrite=false", "autofeature=false",
            *feature_args(fc), f"betamultiplier={beta}", f"outputformat={output}",
            "writebackgroundpredictions=true", "outputgrids=false", "pictures=false", "plots=false",
            f"responsecurves={'true' if analysis else 'false'}",
            f"jackknife={'true' if (analysis if jackknife is None else jackknife) else 'false'}",
            f"writeplotdata={'true' if analysis else 'false'}", "warnings=false",
            "addsamplestobackground=true", "extrapolate=true", "doclamp=true", f"replicates={replicates}"]
    if replicates > 1:
        args.append(f"replicatetype={replicatetype}")
    if test is not None:
        args.append(f"testsamplesfile={test}")
    if extra:
        args += list(extra)
    res = subprocess.run(args, capture_output=True, text=True)
    ok_file = (outdir / f"{SPECIES}.lambdas").exists() or (outdir / f"{SPECIES}_0.lambdas").exists()
    if res.returncode != 0 or not ok_file:
        raise RuntimeError(f"MaxEnt failed in {outdir}: {res.stderr[-500:] or res.stdout[-500:]}")
    return outdir


def _pred_column(df, output):
    col = [c for c in df.columns if output.lower() in c.lower()]
    if not col:
        raise KeyError(f"no {output} column in {list(df.columns)}")
    return col[0]


def read_predictions(outdir, output="cloglog"):
    """Return dict with arrays: train, test (may be empty), background."""
    outdir = Path(outdir)
    sp = pd.read_csv(outdir / f"{SPECIES}_samplePredictions.csv")
    c = _pred_column(sp, output)
    kind = sp[[k for k in sp.columns if "test or train" in k.lower()][0]].astype(str).str.lower()
    bg = pd.read_csv(outdir / f"{SPECIES}_backgroundPredictions.csv")
    # MaxEnt writes sample and background predictions with different numbers of digits; rounding to a
    # common precision prevents formatting differences from breaking ties (D55)
    rnd = lambda a: np.round(np.asarray(a, dtype=float), PRED_DECIMALS)  # noqa: E731
    return {"train": rnd(sp.loc[kind == "train", c].values),
            "test": rnd(sp.loc[kind == "test", c].values),
            "background": rnd(bg[_pred_column(bg, output)].values),
            "bg_x": bg.iloc[:, 0].values.astype(float), "bg_y": bg.iloc[:, 1].values.astype(float)}


def n_parameters(outdir):
    """Number of non-zero lambdas (feature weights) in the .lambdas file."""
    k = 0
    for line in (Path(outdir) / f"{SPECIES}.lambdas").read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                if float(parts[1]) != 0.0:
                    k += 1
            except ValueError:
                pass
    return k


def write_swd(path, label, lon, lat, values: pd.DataFrame):
    df = pd.DataFrame({"species": label, "lon": np.round(lon, 6), "lat": np.round(lat, 6)})
    df = pd.concat([df, values.reset_index(drop=True)], axis=1)
    df.to_csv(path, index=False)


def read_results(outdir):
    """maxentResults.csv as a dict (last row = summary for replicated runs)."""
    df = pd.read_csv(Path(outdir) / "maxentResults.csv")
    return df.iloc[-1].to_dict(), df


def results_value(res, *keywords):
    """First value whose column name contains all keywords (case-insensitive)."""
    for k, v in res.items():
        if all(w.lower() in str(k).lower() for w in keywords):
            return v
    return np.nan


def read_plot_data(outdir, variables, replicate=None):
    """Response-curve plot data written by MaxEnt (writeplotdata=true).

    Returns DataFrame with variable, kind ('marginal' or 'only'), x, y."""
    plots = Path(outdir) / "plots"
    rows = []
    tag = "" if replicate is None else f"_{replicate}"
    for v in variables:
        for kind, suffix in (("marginal", ""), ("only", "_only")):
            f = plots / f"{SPECIES}{tag}_{v}{suffix}.dat"
            if f.exists():
                d = pd.read_csv(f)
                rows.append(pd.DataFrame({"variable": v, "kind": kind, "x": d.iloc[:, 1].values,
                                          "y": d.iloc[:, 2].values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["variable", "kind", "x", "y"])
