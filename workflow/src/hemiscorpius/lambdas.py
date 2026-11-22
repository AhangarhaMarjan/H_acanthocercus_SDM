"""Evaluate a MaxEnt (3.4.x) .lambdas file in Python (projections, step 15; decision D60).

Feature syntax in .lambdas (name, lambda, min, max):
    x          linear            f = (x - min) / (max - min)
    x^2        quadratic         f = (x^2 - min) / (max - min)
    x*y        product           f = (x*y - min) / (max - min)
    'x         forward hinge     f = (x - min) / (max - min) if x > min else 0     (min = knot)
    `x         reverse hinge     f = (max - x) / (max - min) if x < max else 0     (max = knot)
    (t<x)      threshold         f = 1 if x > t else 0
Prediction: S = sum(lambda * f); raw = exp(S - linearPredictorNormalizer) / densityNormalizer;
cloglog = 1 - exp(-exp(entropy) * raw). With clamping (MaxEnt default) every variable is first limited to
the range of its linear feature (the training range). Validated against MaxEnt's own background
predictions in step 15 before any projection is used.
"""
import re
from pathlib import Path

import numpy as np

_THRESH = re.compile(r"^\((?P<t>[-+0-9.eE]+)<(?P<v>.+)\)$")


def read_lambdas(path):
    feats, consts = [], {}
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            feats.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        elif len(parts) == 2:
            consts[parts[0]] = float(parts[1])
    return feats, consts


def variable_ranges(feats):
    """Training range of each variable = min/max of its linear feature."""
    rng = {}
    for name, _, lo, hi in feats:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            rng[name] = (lo, hi)
    return rng


def _feature_values(name, lo, hi, env):
    span = (hi - lo) if hi != lo else 1.0
    m = _THRESH.match(name)
    if m:
        return (env[m.group("v")] > float(m.group("t"))).astype(float)
    if name.startswith("'"):
        x = env[name[1:]]
        return np.where(x > lo, (x - lo) / span, 0.0)
    if name.startswith("`"):
        x = env[name[1:]]
        return np.where(x < hi, (hi - x) / span, 0.0)
    if name.endswith("^2"):
        x = env[name[:-2]]
        return (x * x - lo) / span
    if "*" in name:
        a, b = name.split("*")
        return (env[a] * env[b] - lo) / span
    return (env[name] - lo) / span


def predict(lambdas_path, env, output="cloglog", clamp=True):
    """env: dict variable -> 1-D array (same length). Returns predictions (NaN where any input is NaN)."""
    feats, c = read_lambdas(lambdas_path)
    env = {k: np.asarray(v, dtype="float64") for k, v in env.items()}
    if clamp:
        for v, (lo, hi) in variable_ranges(feats).items():
            if v in env:
                env[v] = np.clip(env[v], lo, hi)
    n = len(next(iter(env.values())))
    s = np.zeros(n)
    for name, lam, lo, hi in feats:
        if lam != 0.0:
            s += lam * _feature_values(name, lo, hi, env)
    raw = np.exp(s - c["linearPredictorNormalizer"]) / c["densityNormalizer"]
    bad = np.zeros(n, bool)
    for v in env.values():
        bad |= ~np.isfinite(v)
    if output == "raw":
        out = raw
    elif output == "cloglog":
        out = 1.0 - np.exp(-np.exp(c["entropy"]) * raw)
    elif output == "logistic":
        out = raw * np.exp(c["entropy"]) / (1 + raw * np.exp(c["entropy"]))
    else:
        raise ValueError(output)
    out[bad] = np.nan
    return out


def outside_range(lambdas_path, env):
    """Boolean array: True where any variable is outside its training range (clamped cells)."""
    feats, _ = read_lambdas(lambdas_path)
    n = len(next(iter(env.values())))
    o = np.zeros(n, bool)
    for v, (lo, hi) in variable_ranges(feats).items():
        if v in env:
            x = np.asarray(env[v], float)
            o |= (x < lo) | (x > hi)
    return o
