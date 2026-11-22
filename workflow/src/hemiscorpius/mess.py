"""Multivariate environmental similarity surface (MESS) and most dissimilar variable (MoD), Elith et al. (2010).

For each variable with reference values r (training data) and projection value p:
    f = 100 * share of r < p   (percentile)
    f == 0          -> 100 * (p - min) / (max - min)          (negative below the range)
    0 < f <= 50     -> 2 f
    50 < f < 100    -> 2 (100 - f)
    f == 100        -> 100 * (max - p) / (max - min)          (negative above the range)
MESS = minimum over variables; MoD = variable giving the minimum. MESS < 0: at least one variable outside the
reference range (novel climate; extrapolation).
"""
import numpy as np


def similarity(ref, p):
    ref = np.sort(np.asarray(ref, float))
    ref = ref[np.isfinite(ref)]
    p = np.asarray(p, float)
    lo, hi = ref[0], ref[-1]
    span = (hi - lo) if hi > lo else 1.0
    f = 100.0 * np.searchsorted(ref, p, side="left") / len(ref)
    s = np.where(f <= 50, 2 * f, 2 * (100 - f))
    s = np.where(p <= lo, 100 * (p - lo) / span, s)
    s = np.where(p >= hi, 100 * (hi - p) / span, s)
    return np.where(np.isfinite(p), s, np.nan)


def mess(ref: dict, env: dict):
    """ref, env: dict variable -> 1-D array. Returns (mess, mod_index, variable names)."""
    names = list(ref)
    sims = np.stack([similarity(ref[v], env[v]) for v in names])
    allnan = np.all(np.isnan(sims), axis=0)
    filled = np.where(np.isnan(sims), np.inf, sims)
    mod = np.argmin(filled, axis=0)
    m = np.min(filled, axis=0)
    m[allnan] = np.nan
    return m, mod, names
