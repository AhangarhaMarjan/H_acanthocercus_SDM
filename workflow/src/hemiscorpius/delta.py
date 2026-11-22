"""Delta method used for future climate (step 05, decisions D29-D30)."""
import numpy as np


def precip_ratio(p_future, p_normal, offset=1.0, cap=3.0):
    """(P_future + offset) / (P_normal + offset), capped at 'cap'. Returns (ratio, capped_mask)."""
    raw = (np.asarray(p_future, dtype="float64") + offset) / (np.asarray(p_normal, dtype="float64") + offset)
    capped = np.isfinite(raw) & (raw > cap)
    return np.minimum(raw, cap), capped


def apply_delta(base, normal, future, offset=1.0, cap=3.0):
    """Future climatology from a baseline and a GCM change relative to the normals.

    base, normal, future: dicts with 'tmin', 'tmax', 'prec' arrays of identical shape.
    Temperature: base + (future - normal). Precipitation: base * capped ratio (never negative).
    """
    ratio, capped = precip_ratio(future["prec"], normal["prec"], offset, cap)
    out = {"tmin": base["tmin"] + (future["tmin"] - normal["tmin"]),
           "tmax": base["tmax"] + (future["tmax"] - normal["tmax"]),
           "prec": np.maximum(base["prec"] * ratio, 0.0)}
    return out, capped
