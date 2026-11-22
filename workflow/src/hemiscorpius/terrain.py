"""Terrain helpers (step 07, decision D42)."""
import numpy as np


def vrm(slope_deg, northness, eastness, window=3):
    """Vector Ruggedness Measure (Sappington et al. 2007) on a 2-D block.

    Each cell's surface normal is decomposed from slope and aspect (here given as
    northness = cos(aspect) and eastness = sin(aspect)):
        x = sin(slope) * eastness,  y = sin(slope) * northness,  z = cos(slope)
    The vectors are summed over a window x window neighbourhood; VRM = 1 - |sum| / n.
    Returns NaN where any cell of the window is missing. Edges use the nearest values;
    pass a block with a halo of window // 2 cells and crop it afterwards.
    """
    from scipy.ndimage import uniform_filter
    s = np.deg2rad(np.asarray(slope_deg, dtype="float32"))
    n_ = np.asarray(northness, dtype="float32")
    e_ = np.asarray(eastness, dtype="float32")
    valid = np.isfinite(s) & np.isfinite(n_) & np.isfinite(e_)
    s = np.where(valid, s, np.float32(0))
    sin_s = np.where(valid, np.sin(s), np.float32(0)).astype("float32")
    x = sin_s * np.where(valid, e_, np.float32(0))
    y = sin_s * np.where(valid, n_, np.float32(0))
    z = np.where(valid, np.cos(s), np.float32(0)).astype("float32")
    del sin_s, s
    k = window * window
    sx = uniform_filter(x, size=window, mode="nearest") * k
    sy = uniform_filter(y, size=window, mode="nearest") * k
    sz = uniform_filter(z, size=window, mode="nearest") * k
    cnt = uniform_filter(valid.astype("float32"), size=window, mode="nearest") * k
    out = 1.0 - np.sqrt(sx ** 2 + sy ** 2 + sz ** 2) / k
    out[cnt < k - 1e-3] = np.nan
    return np.clip(out, 0.0, 1.0).astype("float32") if out.size else out.astype("float32")
