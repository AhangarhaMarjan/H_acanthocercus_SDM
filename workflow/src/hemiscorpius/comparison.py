"""Comparison helpers for step 17 (decision D63)."""
import itertools

import numpy as np


def schoener_d(a, b):
    """Schoener's D between two suitability surfaces (normalised to sum 1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    pa, pb = a[ok] / a[ok].sum(), b[ok] / b[ok].sum()
    return float(1 - 0.5 * np.abs(pa - pb).sum())


def jaccard(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def anova_partition(y, factors):
    """Variance partitioning of a balanced full-factorial design.

    y: array (n_cells, n_levels_f1, ..., n_levels_fk, [n_rep]); factors: names of the first k level axes
    (the optional last axis = replicates, the residual). Returns dict effect -> share of total SS, averaged
    as sum of SS over cells / total SS over cells. Effects: every main effect and interaction.
    """
    y = np.asarray(y, float)
    k = len(factors)
    has_rep = y.ndim == k + 2
    grand = y.reshape(y.shape[0], -1).mean(axis=1)
    total = ((y - grand.reshape((-1,) + (1,) * (y.ndim - 1))) ** 2).reshape(y.shape[0], -1).sum(axis=1)
    axes_f = list(range(1, k + 1))
    cell_means = y.mean(axis=-1) if has_rep else y               # mean over replicates
    n_rep = y.shape[-1] if has_rep else 1
    # marginal means for every subset of factors
    means = {}
    for r in range(0, k + 1):
        for sub in itertools.combinations(range(k), r):
            drop = tuple(a for a in axes_f if (a - 1) not in sub)
            means[sub] = cell_means.mean(axis=drop, keepdims=True) if drop else cell_means
    # effect = inclusion-exclusion over subsets
    ss = {}
    for r in range(1, k + 1):
        for sub in itertools.combinations(range(k), r):
            eff = np.zeros_like(cell_means)
            for rr in range(0, r + 1):
                for s2 in itertools.combinations(sub, rr):
                    eff = eff + ((-1) ** (r - rr)) * np.broadcast_to(means[s2], cell_means.shape)
            ss[":".join(factors[i] for i in sub)] = (eff ** 2).reshape(eff.shape[0], -1).sum(axis=1) * n_rep
    if has_rep:
        ss["replicate (residual)"] = ((y - cell_means[..., None]) ** 2).reshape(y.shape[0], -1).sum(axis=1)
    tot = total.sum()
    return {k_: float(v.sum() / tot) for k_, v in ss.items()} if tot > 0 else {}


def binary_agreement(ref, test, weights=None):
    """Agreement metrics of a binary map against a reference (area-weighted if weights are given)."""
    ref, test = np.asarray(ref, bool), np.asarray(test, bool)
    w = np.ones(ref.shape) if weights is None else np.asarray(weights, float)
    tp, tn = w[ref & test].sum(), w[~ref & ~test].sum()
    fp, fn = w[~ref & test].sum(), w[ref & ~test].sum()
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    oa = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(overall_accuracy=oa, sensitivity=sens, specificity=spec, precision=prec,
                f1=2 * prec * sens / (prec + sens) if prec + sens else np.nan,
                jaccard=tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                kappa=(oa - pe) / (1 - pe) if pe < 1 else np.nan, tss=sens + spec - 1,
                mcc=(tp * tn - fp * fn) / mcc_den if mcc_den else np.nan,
                underestimation_km2=fn, overestimation_km2=fp)


def spherical_centroid(lon, lat, weights):
    """Area-weighted centroid on the sphere (mean of unit vectors). Returns lon, lat in degrees."""
    lo, la, w = np.deg2rad(lon), np.deg2rad(lat), np.asarray(weights, float)
    x, y, z = (w * np.cos(la) * np.cos(lo)).sum(), (w * np.cos(la) * np.sin(lo)).sum(), (w * np.sin(la)).sum()
    return float(np.rad2deg(np.arctan2(y, x))), float(np.rad2deg(np.arctan2(z, np.hypot(x, y))))


def distance_azimuth(lon1, lat1, lon2, lat2, r=6371.0088):
    """Great-circle distance (km) and initial azimuth (degrees from north, clockwise) from point 1 to 2."""
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(a))
    az = np.rad2deg(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)))
    return float(d), float((az + 360) % 360)
