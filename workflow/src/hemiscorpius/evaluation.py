"""Evaluation and partition helpers for MaxEnt tuning (step 13, decision D51)."""
import numpy as np
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- partitions
def block_partition(occ_lon, occ_lat, bg_lon, bg_lat):
    """ENMeval 'block' partition (Muscarella et al. 2014): split occurrences at the longitude
    median, then each half at its latitude median -> 4 groups with ~equal occurrences.
    Background points are assigned with the same boundaries. Returns (occ_groups, bg_groups) in 1..4."""
    occ_lon, occ_lat = np.asarray(occ_lon, float), np.asarray(occ_lat, float)
    n = len(occ_lon)
    s = np.sort(occ_lon)
    lon_cut = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
    west = occ_lon <= lon_cut if n % 2 == 0 else occ_lon < lon_cut
    if n % 2 == 1:                                    # odd n: put the median record in the west half
        west = west | (occ_lon == lon_cut)
    cuts = {}
    for half, sel in (("w", west), ("e", ~west)):
        la = np.sort(occ_lat[sel])
        m = len(la)
        cuts[half] = (la[m // 2 - 1] + la[m // 2]) / 2 if m % 2 == 0 else la[m // 2]

    def assign(lon, lat, is_occ):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        w = (lon <= lon_cut) if not is_occ else west
        g = np.where(w, np.where(lat <= cuts["w"], 1, 2), np.where(lat <= cuts["e"], 3, 4))
        return g
    return assign(occ_lon, occ_lat, True), assign(bg_lon, bg_lat, False)


def random_kfold(n, k=4, seed=0):
    rng = np.random.default_rng(seed)
    g = np.tile(np.arange(1, k + 1), int(np.ceil(n / k)))[:n]
    return rng.permutation(g)


# --------------------------------------------------------------------------- metrics
def auc(presence, background):
    """AUC = probability that a presence scores higher than a background point (ties = 0.5)."""
    p, b = np.asarray(presence, float), np.asarray(background, float)
    allv = np.concatenate([p, b])
    ranks = _rankdata(allv)
    rp = ranks[: len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(b)))


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def p10_threshold(train_pred):
    """Prediction value below which at most 10 % of the training presences fall."""
    s = np.sort(np.asarray(train_pred, float))
    return float(s[int(np.floor(0.1 * len(s)))])


def omission(test_pred, threshold):
    return float(np.mean(np.asarray(test_pred, float) < threshold))


def boyce(presence, background, n_windows=100, width=None):
    """Continuous Boyce index (Hirzel et al. 2006), moving window as in ecospat.boyce.

    Window width defaults to 1/10 of the prediction range; returns Spearman r between the
    predicted-to-expected ratio and the window mid-points (NaN windows dropped)."""
    p, b = np.asarray(presence, float), np.asarray(background, float)
    lo, hi = min(p.min(), b.min()), max(p.max(), b.max())
    width = (hi - lo) / 10 if width is None else width
    starts = np.linspace(lo, hi - width, n_windows)
    pe, mids = [], []
    for s0 in starts:
        fp = np.mean((p >= s0) & (p <= s0 + width))
        fb = np.mean((b >= s0) & (b <= s0 + width))
        if fb > 0:
            pe.append(fp / fb)
            mids.append(s0 + width / 2)
    pe = np.array(pe)
    if len(pe) < 3 or np.all(pe == pe[0]):
        return np.nan
    keep = np.r_[True, np.diff(pe) != 0]           # drop successive duplicates (ecospat behaviour)
    return float(spearmanr(np.array(mids)[keep], pe[keep]).correlation)


def aicc(raw_occ, raw_bg, k):
    """AICc of a MaxEnt model (Warren & Seifert 2011): raw output normalised over the background."""
    n = len(raw_occ)
    if k >= n - 1 or k == 0:
        return np.inf
    ll = float(np.sum(np.log(np.asarray(raw_occ, float) / np.sum(raw_bg))))
    return 2 * k - 2 * ll + 2 * k * (k + 1) / (n - k - 1)


def tss_at(threshold, presence, background):
    """True skill statistic with background as pseudo-absence (for reporting only)."""
    sens = np.mean(np.asarray(presence) >= threshold)
    spec = np.mean(np.asarray(background) < threshold)
    return float(sens + spec - 1)


def validation_metrics(pred, fold, bg_group, min_bg=50, decimals=5):
    """Metrics of one cross-validation fold (D53).

    pred: dict with 'train', 'test', 'background' predictions; bg_group: block label of every background
    prediction (0 = unpartitioned, e.g. random CV). Validation AUC and CBI use the background of the
    withheld block (ENMeval validation.bg = 'partition'); full-background values are reported as well.
    """
    rnd = lambda a: np.round(np.asarray(a, dtype=float), decimals)  # noqa: E731
    pred = dict(pred, train=rnd(pred["train"]), test=rnd(pred["test"]), background=rnd(pred["background"]))
    thr10, thrmtp = p10_threshold(pred["train"]), float(np.min(pred["train"]))
    test, bg = pred["test"], pred["background"]
    in_block = np.asarray(bg_group) == fold
    if not in_block.any() and np.any(bg_group):
        raise ValueError("Spatial fold contains no matched background")
    bg_val = bg[in_block] if in_block.any() else bg
    r = dict(auc_train=auc(pred["train"], bg), auc_val=auc(test, bg_val), auc_val_fullbg=auc(test, bg),
             or10=omission(test, thr10), ormtp=omission(test, thrmtp),
             cbi_val=boyce(test, bg_val) if len(test) >= 3 else np.nan,
             cbi_val_fullbg=boyce(test, bg) if len(test) >= 3 else np.nan, n_bg_val=int(len(bg_val)),
             bg_pred_sd=float(np.std(bg)))
    r["auc_diff"] = r["auc_train"] - r["auc_val"]
    return r


def select_setting(agg, or10_tolerance=0.0):
    """Sequential selection (D51, D55): settings whose mean OR10 is within 'or10_tolerance' of the lowest
    (differences smaller than one omitted record are treated as ties) -> highest mean validation AUC ->
    fewest parameters. Only settings with 4 complete folds and >= 1 parameter are eligible."""
    a = agg[(agg.n_folds == 4) & (agg.n_params > 0)].copy()
    best_or = a.or10_mean.min()
    elig = a[a.or10_mean <= best_or + or10_tolerance + 1e-9]
    return elig.sort_values(["auc_val_mean", "n_params"], ascending=[False, True]).iloc[0]
