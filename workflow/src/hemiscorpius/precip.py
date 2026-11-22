"""Precipitation helpers shared by several steps (decisions D18-D19)."""
import numpy as np


def flag_extremes(prec, min_mm=200.0, factor=10.0, offset_mm=5.0):
    """Boolean flags for monthly values that are extreme relative to the cell's own climate.

    prec: (n_months, rows, cols) starting in January, n_months a multiple of 12.
    A value is flagged if value >= min_mm AND value >= factor x (same-month cell median + offset_mm).
    Returns (flags, medians[12, rows, cols]).
    """
    flags = np.zeros(prec.shape, dtype=bool)
    med = np.empty((12,) + prec.shape[1:], dtype="float32")
    for m in range(12):
        sub = prec[m::12]
        med[m] = np.nanmedian(sub, axis=0)
        flags[m::12] = (sub >= min_mm) & (sub >= factor * (med[m] + offset_mm))
    return flags, med


def clean_precip(prec, rule):
    """Replace flagged values by the same-month cell median (sensitivity version, D19).

    rule: dict with min_mm, factor, offset_mm. Returns (cleaned, flags).
    """
    flags, med = flag_extremes(prec, rule["min_mm"], rule["factor"], rule["offset_mm"])
    cleaned = prec.copy()
    for m in range(12):
        cleaned[m::12] = np.where(flags[m::12], med[m][None], prec[m::12])
    return cleaned, flags


def wettest_quarter_start(prec12):
    """First month index (0-11) of the wettest circular 3-month window; NaN where data are missing."""
    q = np.stack([prec12[k] + prec12[(k + 1) % 12] + prec12[(k + 2) % 12] for k in range(12)])
    valid = np.isfinite(q).all(axis=0)
    out = np.argmax(np.where(np.isfinite(q), q, -np.inf), axis=0).astype("float32")
    out[~valid] = np.nan
    return out
