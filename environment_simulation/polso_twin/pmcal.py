"""
On-device PM2.5 correction for the PMS7003, using only what the device itself measures.

    PM_corrected = exp(beta_c) * PM_cf1^alpha_c / (1 + kappa_c * h),   h = a / (1 - a),  a = min(RH_BME688, 95 %) / 100

c is the source the device has just attributed (or "unknown" when it abstains). The humidity term is the
kappa-Koehler growth form: particles swell in humid air and the optical counter over-reads them. The
per-source gain handles composition: the counter under-sees small soot and over-weights some industrial and
hygroscopic aerosols. Three numbers per class; a table lookup plus one division on the MCU.

Parameters are fitted on the simulator's TRAINING sessions here. On real hardware they must be re-fitted from
the co-location campaign (FEM reference + labelled sources); the fit code is the same.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import params as P
from .simulate import run_scenario, window_truth
from .firmware import run_device, decode_frames_fast
from .model import pmap

KEYS = P.CLASSES + ["unknown"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "pm_cal.json")
IDENTITY = {k: dict(alpha=1.0, beta=0.0, kappa=0.0, q16=-0.3, q84=0.3) for k in KEYS}


def _rows(args):
    sc, qm = args
    streams, truth = run_scenario(sc)
    out = run_device(streams, qm, calib=IDENTITY)
    rows = []
    for w, e in enumerate(out["t_end"]):
        if out["status"][w] != "ok":
            continue
        lab, _, total = window_truth(truth, int(e))
        rows.append(dict(pm=out["pm_raw"][w], rh=out["rh_win"][w], label=out["label"][w], true=total, truth=lab,
                         sess=sc.seed, w=w))
    return rows


def build(scenarios, qm):
    return pd.DataFrame([r for rows in pmap(_rows, [(s, qm) for s in scenarios]) for r in rows])


def _h(rh):
    a = np.clip(np.asarray(rh, float), 0, 95) / 100.0
    return a / (1 - a)


def fit(df, truth_only=False):
    """truth_only: keep windows whose true dominant source is a single class (controlled chamber data)."""
    df = df[(df["pm"] >= 3) & (df["true"] >= 3)].copy()
    if truth_only:
        df = df[df["truth"].isin(P.CLASSES)]
    df["key"] = np.where(df["label"].isin(P.CLASSES), df["label"], "unknown")
    out = {}
    for k in KEYS:
        d = df[df["key"] == k]
        if len(d) < 200:
            d = df                                            # too few windows: share the global fit
        x, h, y = np.log(d["pm"].values), _h(d["rh"].values), np.log(d["true"].values)

        def res(p):
            a, b, kap = p
            return a * x + b - np.log1p(kap * h) - y
        r = least_squares(res, [1.0, 0.0, 0.2], bounds=([0.5, -3, 0.0], [1.5, 3, 2.0]), loss="soft_l1", f_scale=0.15)
        e = res(r.x)
        out[k] = dict(alpha=float(r.x[0]), beta=float(r.x[1]), kappa=float(r.x[2]),
                      q16=float(np.quantile(-e, 0.16)), q84=float(np.quantile(-e, 0.84)), n=int(len(d)))
    return out


def calibrate_bands(cal, df, n=10, recentre=True):
    """Re-centre each class and set its band from HELD-OUT data, on 5-minute averages (what the device displays):
    beta moves by the median of log(true 5-min mean / corrected 5-min mean); q16 / q84 are the 16th / 84th
    percentiles around that median, so about 68 % of true values should fall inside the band."""
    df = df.sort_values(["sess", "w"]).copy()
    df["corr"] = [apply(cal, r.pm, r.rh, r.label)[0] for r in df.itertuples()]
    out = {k: dict(v) for k, v in cal.items()}
    res = {k: [] for k in KEYS}
    for _, g in df.groupby("sess"):
        c, t, lab, w = g["corr"].values, g["true"].values, g["label"].values, g["w"].values
        for i in range(len(g)):
            m = (w > w[i] - n) & (w <= w[i])
            c5, t5 = c[m].mean(), t[m].mean()
            if c5 > 3 and t5 > 3:
                res[lab[i] if lab[i] in P.CLASSES else "unknown"].append(np.log(t5 / c5))
    for k in KEYS:
        r = np.array(res[k]) if len(res[k]) >= 100 else np.concatenate([np.array(v) for v in res.values()])
        q16, q50, q84 = np.quantile(r, [0.16, 0.5, 0.84])
        if not recentre:
            q50 = 0.0                                         # keep the centre; the band may then be lopsided
        out[k]["beta"] = out[k]["beta"] + float(q50)          # re-centre: the shown value is the held-out median
        out[k]["q16"], out[k]["q84"] = float(q16 - q50), float(q84 - q50)
        out[k]["n_band"] = int(len(r))
    return out


def apply(cal, pm, rh, label):
    k = label if label in P.CLASSES else "unknown"
    c = cal[k]
    pm = max(float(pm), 0.0)
    corr = np.exp(c["beta"]) * (pm + 1e-6) ** c["alpha"] / (1 + c["kappa"] * float(_h(rh)))
    return float(corr), float(corr * np.exp(c["q16"])), float(corr * np.exp(c["q84"]))


def save(cal, path=PATH):
    json.dump(cal, open(path, "w"), indent=1)


def load(path=PATH):
    return json.load(open(path)) if os.path.exists(path) else IDENTITY


def export_c_header(cal, path):
    lines = ["/* AUTO-GENERATED by polso_twin.pmcal - PM2.5 correction per attributed source. */",
             "#ifndef PM_CAL_PARAMS_H\n#define PM_CAL_PARAMS_H",
             "/* rows: vehicle, biomass, dust, industrial, background, unknown; cols: alpha, beta, kappa, q16, q84 */",
             "static const float PMCAL[6][5] = {"]
    for k in KEYS:
        c = cal[k]
        lines.append(f"    {{{c['alpha']:.6f}f, {c['beta']:.6f}f, {c['kappa']:.6f}f, {c['q16']:.6f}f, {c['q84']:.6f}f}},  /* {k} */")
    lines += ["};", "#endif", ""]
    open(path, "w").write("\n".join(lines))


def coverage(cal, df, n=10):
    """Share of held-out 5-minute averages whose true value falls inside the displayed band, and the median error."""
    df = df.sort_values(["sess", "w"])
    inside, err = [], []
    for _, g in df.groupby("sess"):
        a = np.array([apply(cal, r.pm, r.rh, r.label) for r in g.itertuples()])
        t, w = g["true"].values, g["w"].values
        for i in range(len(g)):
            m = (w > w[i] - n) & (w <= w[i])
            c5, lo, hi, t5 = a[m, 0].mean(), a[m, 1].mean(), a[m, 2].mean(), t[m].mean()
            if t5 > 3:
                inside.append(lo <= t5 <= hi)
                err.append(abs(c5 - t5) / t5)
    return float(np.mean(inside)), float(np.median(err)), float(np.quantile(err, 0.9))


def inflate_bands(cal, df, target=0.68):
    """Widen every band by one common factor until `target` of 5-minute true values on a SECOND held-out set
    (different sites and days) fall inside - so the band also covers site-to-site differences."""
    def scaled(f):
        return {k: dict(v, q16=v["q16"] * f, q84=v["q84"] * f) for k, v in cal.items()}
    lo, hi = 1.0, 6.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if coverage(scaled(mid), df)[0] < target:
            lo = mid
        else:
            hi = mid
    out = scaled(hi)
    for v in out.values():
        v["band_scale"] = hi
    return out
