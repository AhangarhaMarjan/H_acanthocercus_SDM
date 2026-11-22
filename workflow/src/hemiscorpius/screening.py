"""Predictor screening helpers (step 11, decision D49)."""
import numpy as np


def vif(X):
    """Variance inflation factor of every column of X (n x p): VIF_j = 1 / (1 - R2_j)."""
    X = np.asarray(X, dtype="float64")
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    out = []
    for j in range(X.shape[1]):
        y, others = X[:, j], np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            out.append(1.0)
            continue
        A = np.column_stack([np.ones(len(y)), others])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        r2 = 1 - resid.var() / y.var()
        out.append(np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2))
    return np.array(out)


def greedy_by_priority(priority, max_abs_r, threshold):
    """Walk the priority list; keep a variable if |r| <= threshold with every variable kept so far.

    max_abs_r: dict-like (a, b) -> max |Spearman r| over baselines. Returns (kept, log).
    """
    kept, log = [], []
    for v in priority:
        clash = [(k, max_abs_r[(v, k)]) for k in kept if max_abs_r[(v, k)] > threshold]
        if clash:
            k, r = max(clash, key=lambda t: t[1])
            log.append(dict(variable=v, decision="removed", reason=f"|r| = {r:.2f} with higher-priority {k}"))
        else:
            kept.append(v)
            log.append(dict(variable=v, decision="kept", reason=""))
    return kept, log
