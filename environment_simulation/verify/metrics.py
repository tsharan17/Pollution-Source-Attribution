"""
Metrics used to verify the device against reference instruments and reference labels.

PM mass follows the US EPA performance targets for PM2.5 sensors (base testing); classification metrics
are the ones the project deck promises (per-class precision / recall, confusion matrix, abstention,
calibration). All have confidence intervals that respect the time correlation of the data
(blocks of whole days, never individual windows).
"""
import numpy as np

# EPA PM2.5 base-testing targets (precision, bias, linearity, error). Verify against the current EPA
# Air Sensor Performance Targets document before quoting.
EPA_PM25 = dict(slope=(0.65, 1.35), intercept=(-5.0, 5.0), r2_min=0.70, rmse_max=7.0, nrmse_max=0.30,
                precision_sd_max=5.0, precision_cv_max=0.30)


def ols_epa(ref, dev):
    """Regress device on reference. Returns slope, intercept, R^2 and the EPA error metrics."""
    ref, dev = np.asarray(ref, float), np.asarray(dev, float)
    ok = np.isfinite(ref) & np.isfinite(dev)
    x, y = ref[ok], dev[ok]
    slope, icpt = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    return dict(slope=float(slope), intercept=float(icpt), r2=float(r * r), rmse=rmse,
                nrmse=rmse / float(np.mean(x)), n=int(ok.sum()), mean_ref=float(np.mean(x)))


def epa_verdict(m, precision):
    lo, hi = EPA_PM25["slope"]
    checks = {
        "slope": lo <= m["slope"] <= hi,
        "intercept": EPA_PM25["intercept"][0] <= m["intercept"] <= EPA_PM25["intercept"][1],
        "R2": m["r2"] >= EPA_PM25["r2_min"],
        "error": m["rmse"] <= EPA_PM25["rmse_max"] or m["nrmse"] <= EPA_PM25["nrmse_max"],
        "precision": precision["sd"] <= EPA_PM25["precision_sd_max"] or precision["cv"] <= EPA_PM25["precision_cv_max"],
    }
    return checks


def precision_units(mat, min_mean=5.0):
    """mat: (n_times, n_units) co-located readings. Mean across-unit SD, and CV over times where the
    mean is at least `min_mean` (a CV at near-zero concentration is meaningless)."""
    mat = np.asarray(mat, float)
    mat = mat[np.all(np.isfinite(mat), 1)]
    sd = mat.std(1, ddof=1)
    mean = mat.mean(1)
    k = mean >= min_mean
    return dict(sd=float(sd.mean()), cv=float((sd[k] / mean[k]).mean()) if k.any() else float("nan"))


def fit_rh_correction(pm_dev, rh, pm_ref):
    """ref ~ a*pm + b*rh + c  (the form used for low-cost optical counters). Returns (a, b, c)."""
    A = np.column_stack([pm_dev, rh, np.ones(len(pm_dev))])
    coef, *_ = np.linalg.lstsq(A, pm_ref, rcond=None)
    return coef


def apply_rh_correction(coef, pm_dev, rh):
    return np.clip(coef[0] * pm_dev + coef[1] * rh + coef[2], 0, None)


def fit_source_correction(pm_dev, rh, shares, pm_ref, alpha=1.0):
    """Source-aware correction: the device's own attributed source mix selects the gain.
    ref ~ sum_c a_c * share_c * pm + b*rh + c   (ridge, c over the five source classes)."""
    A = np.column_stack([shares * pm_dev[:, None], rh, np.ones(len(pm_dev))])
    reg = alpha * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ pm_ref)


def apply_source_correction(coef, pm_dev, rh, shares):
    A = np.column_stack([shares * pm_dev[:, None], rh, np.ones(len(pm_dev))])
    return np.clip(A @ coef, 0, None)


def macro_f1(y_true, y_pred, labels):
    f = []
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_class(y_true, y_pred, labels):
    out = {}
    for c in labels:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        out[c] = dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)), support=int(tp + fn))
    return out


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    tot, e = len(conf), 0.0
    rel = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.any():
            e += m.sum() / tot * abs(conf[m].mean() - correct[m].mean())
            rel.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(e), rel


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def block_bootstrap(stat_fn, blocks, n_boot=400, seed=0):
    """blocks: list of per-block data; stat_fn(list_of_blocks) -> float. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    point = stat_fn(blocks)
    vals = [stat_fn([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(n_boot)]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def n_needed_for_halfwidth(p, half, z=1.96):
    """Independent samples needed so that a proportion p has a normal-approximation CI half-width `half`."""
    return int(np.ceil(p * (1 - p) * (z / half) ** 2))
