"""
Independent Python implementation of the 19 bioclimatic variables of dismo::biovars.

Used ONLY to verify the R output (step 04) and in automated tests. The analysis itself uses
dismo::biovars (R), exactly as in the original manuscript.

Definitions follow dismo::biovars (Hijmans et al.; O'Donnell & Ignizio 2012):
    tavg = (tmin + tmax) / 2
    quarters = circular 3-month windows starting at each month (Jan-Mar, ..., Dec-Feb)
    standard deviations use n - 1 (R 'sd'); bio15 is the CV of (prec + 1) in percent
    ties in which.max / which.min resolve to the first quarter (as numpy argmax/argmin)
Inputs are arrays of shape (12, ...) in degC and mm. Output: dict bio1..bio19.
"""
import numpy as np


def _quarters(x):
    """Circular 3-month sums: q[k] = x[k] + x[k+1] + x[k+2] (indices mod 12)."""
    return np.stack([x[k] + x[(k + 1) % 12] + x[(k + 2) % 12] for k in range(12)])


def _pick(values, index):
    """values (12, ...) indexed by index (...) along axis 0."""
    return np.take_along_axis(values, index[None, ...], axis=0)[0]


def biovars(prec, tmin, tmax):
    prec, tmin, tmax = (np.asarray(a, dtype="float64") for a in (prec, tmin, tmax))
    tavg = (tmin + tmax) / 2.0
    b = {}
    b["bio1"] = tavg.mean(axis=0)
    b["bio2"] = (tmax - tmin).mean(axis=0)
    b["bio4"] = 100.0 * tavg.std(axis=0, ddof=1)
    b["bio5"] = tmax.max(axis=0)
    b["bio6"] = tmin.min(axis=0)
    b["bio7"] = b["bio5"] - b["bio6"]
    with np.errstate(invalid="ignore", divide="ignore"):
        b["bio3"] = 100.0 * b["bio2"] / b["bio7"]
    b["bio12"] = prec.sum(axis=0)
    b["bio13"] = prec.max(axis=0)
    b["bio14"] = prec.min(axis=0)
    p1 = prec + 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        b["bio15"] = 100.0 * p1.std(axis=0, ddof=1) / p1.mean(axis=0)

    wet = _quarters(prec)
    tq = _quarters(tavg) / 3.0
    nan = ~np.isfinite(wet).all(axis=0) | ~np.isfinite(tq).all(axis=0)
    wet_f = np.where(np.isfinite(wet), wet, -np.inf)
    tq_f = np.where(np.isfinite(tq), tq, -np.inf)
    i_wet = np.argmax(wet_f, axis=0)
    i_dry = np.argmin(np.where(np.isfinite(wet), wet, np.inf), axis=0)
    i_warm = np.argmax(tq_f, axis=0)
    i_cold = np.argmin(np.where(np.isfinite(tq), tq, np.inf), axis=0)

    b["bio8"] = _pick(tq, i_wet)
    b["bio9"] = _pick(tq, i_dry)
    b["bio10"] = _pick(tq, i_warm)
    b["bio11"] = _pick(tq, i_cold)
    b["bio16"] = _pick(wet, i_wet)
    b["bio17"] = _pick(wet, i_dry)
    b["bio18"] = _pick(wet, i_warm)
    b["bio19"] = _pick(wet, i_cold)
    for k in b:
        b[k] = np.where(nan, np.nan, b[k]) if b[k].ndim else b[k]
    out = {f"bio{i}": b[f"bio{i}"] for i in range(1, 20)}
    out["_wettest_quarter_start"] = np.where(nan, np.nan, i_wet.astype("float64"))
    out["_driest_quarter_start"] = np.where(nan, np.nan, i_dry.astype("float64"))
    return out
