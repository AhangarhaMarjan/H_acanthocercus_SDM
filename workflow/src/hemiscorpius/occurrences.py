"""Load occurrence records and locate them on the reference grid."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import raw_dir

LON_NAMES = ("lon", "long", "longitude", "x", "decimallongitude")
LAT_NAMES = ("lat", "latitude", "y", "decimallatitude")


def load_occurrences(cfg: dict) -> pd.DataFrame:
    """Return a DataFrame with columns lon, lat (valid numeric rows only)."""
    name = cfg["paths"].get("static_files", {}).get("occurrences")
    static = raw_dir(cfg, "static")
    hits = list(Path(static).rglob(name)) if name else list(Path(static).rglob("*.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"occurrence file: found {len(hits)} candidates in {static}")
    df = pd.read_csv(hits[0])
    low = {c.lower().strip(): c for c in df.columns}
    lon = next(low[c] for c in LON_NAMES if c in low)
    lat = next(low[c] for c in LAT_NAMES if c in low)
    out = pd.DataFrame({"lon": pd.to_numeric(df[lon], errors="coerce"),
                        "lat": pd.to_numeric(df[lat], errors="coerce")}).dropna()
    return out.reset_index(drop=True)


def cell_indices(occ: pd.DataFrame, grid) -> pd.DataFrame:
    """Add row/col indices of the reference grid cell containing each record."""
    cell = 1.0 / 24.0
    occ = occ.copy()
    occ["row"] = np.floor((grid.north - occ["lat"]) / cell).astype(int)
    occ["col"] = np.floor((occ["lon"] - grid.west) / cell).astype(int)
    inside = occ["row"].between(0, grid.nrows - 1) & occ["col"].between(0, grid.ncols - 1)
    return occ[inside].reset_index(drop=True)


# --------------------------------------------------------------------------- thinning and clustering
SPTHIN_EARTH_RADIUS_KM = 6378.388      # radius used by fields::rdist.earth, i.e. by spThin


def great_circle_km(lon, lat, radius=SPTHIN_EARTH_RADIUS_KM):
    """Pairwise great-circle distances (km) between points given in degrees."""
    lon, lat = np.deg2rad(np.asarray(lon, float)), np.deg2rad(np.asarray(lat, float))
    dlon = lon[:, None] - lon[None, :]
    dlat = lat[:, None] - lat[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def thin_greedy(lon, lat, min_km, reps=100, seed=0):
    """Independent re-implementation of the spThin algorithm (Aiello-Lammens et al. 2015).

    In each repetition the record with the most neighbours closer than min_km is removed (ties at
    random) until no pair is closer than min_km. Returns the indices of the largest retained set over
    all repetitions and the list of retained-set sizes.
    """
    d = great_circle_km(lon, lat)
    close = (d < min_km) & ~np.eye(len(d), dtype=bool)
    rng = np.random.default_rng(seed)
    best, sizes = None, []
    for _ in range(reps):
        keep = np.ones(len(d), dtype=bool)
        while True:
            counts = (close & keep[None, :] & keep[:, None]).sum(axis=1)
            if counts.max() == 0:
                break
            cand = np.flatnonzero(counts == counts.max())
            keep[rng.choice(cand)] = False
        sizes.append(int(keep.sum()))
        if best is None or keep.sum() > best.sum():
            best = keep.copy()
    return np.flatnonzero(best), sizes


def clark_evans(nn_dist_km, area_km2):
    """Clark-Evans aggregation index R (no edge correction) and z statistic.

    R = mean observed nearest-neighbour distance / (0.5 / sqrt(n / A)); R < 1 clustered,
    R ~ 1 random, R > 1 regular. z uses SE = 0.26136 / sqrt(n * n / A).
    """
    nn = np.asarray(nn_dist_km, float)
    n = nn.size
    density = n / area_km2
    expected = 0.5 / np.sqrt(density)
    r = nn.mean() / expected
    se = 0.26136 / np.sqrt(n * density)
    return float(r), float((nn.mean() - expected) / se)


def nearest_neighbour_km(lon, lat):
    d = great_circle_km(lon, lat)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def great_circle_km_between(lon1, lat1, lon2, lat2, radius=SPTHIN_EARTH_RADIUS_KM):
    """Great-circle distances (km) between two point sets: matrix of shape (len1, len2)."""
    lon1, lat1 = np.deg2rad(np.asarray(lon1, float))[:, None], np.deg2rad(np.asarray(lat1, float))[:, None]
    lon2, lat2 = np.deg2rad(np.asarray(lon2, float))[None, :], np.deg2rad(np.asarray(lat2, float))[None, :]
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
