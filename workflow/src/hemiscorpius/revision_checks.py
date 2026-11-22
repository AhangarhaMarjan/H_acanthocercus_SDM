"""Numerical corrections introduced by codex_20261112; no model fitting."""
import numpy as np


def bootstrap_interval(values, confidence=0.95):
    """Empirical percentile interval of estimates, not a CI of their Monte Carlo mean.

    The standard deviation of replicate estimates is the bootstrap SE. Never divide
    it by sqrt(number of bootstrap draws). Small ensembles are labelled exploratory
    by the caller; quantiles do not make a small ensemble precise.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite bootstrap estimates are required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    alpha = (1 - confidence) / 2
    low, high = np.quantile(x, [alpha, 1 - alpha])
    return float(low), float(high), float(x.std(ddof=1))


def null_tail_summary(values, empirical, better="higher"):
    """Do not silently count undefined statistics as non-exceedances.

    A tail probability is reported only if every null statistic is defined.
    Otherwise the valid distribution remains descriptive: filtering undefined
    (often constant) models would change the null hypothesis.
    """
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    out = {"n_valid": int(valid.sum()), "n_undefined": int((~valid).sum()), "p": np.nan,
           "status": "undefined_null_statistics"}
    if not np.isfinite(empirical):
        out["status"] = "undefined_empirical_statistic"
    elif len(x) and valid.all():
        exceed = x >= empirical if better == "higher" else x <= empirical
        out.update(p=float((exceed.sum() + 1) / (len(x) + 1)), status="complete_null_distribution")
    return out


def read_masked_raster(path):
    """Convert the raster's declared NoData to NaN before packing/export."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype("float32").filled(np.nan)


def exact_classification(values, threshold=0.0, operation="ge"):
    """Preserve original threshold decisions before display rounding."""
    a = np.asarray(values, float)
    if operation not in ("ge", "lt"):
        raise ValueError(operation)
    hit = a >= threshold if operation == "ge" else a < threshold
    return np.where(np.isfinite(a), hit.astype(float), np.nan)


def importance_summary(frame):
    """Keep full-model importance separate from bootstrap means and SDs."""
    metrics = ["contribution", "permutation_importance"]
    keys = ["baseline", "variable"]
    reps = frame[frame.source.str.startswith("boot")]
    out = reps.groupby(keys)[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_bootstrap_{b}" for a, b in out.columns]
    out["n_bootstrap"] = reps.groupby(keys).size()
    full = frame[frame.source == "full"].set_index(keys)[metrics].add_suffix("_full_model")
    return out.join(full, how="outer").reset_index()
