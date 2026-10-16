"""
Virtual co-location campaign: 3 prototypes next to FEM-class reference instruments for N days.

Rehearses the verification analysis on simulated data so that, when real hardware arrives, the
pipeline, the file formats and the acceptance criteria already exist and have been exercised -
and so the sample sizes needed for a believable result are known in advance.
"""
import dataclasses
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from polso_twin import params as P, scenarios as S
from polso_twin.simulate import run_scenario, window_truth
from polso_twin.firmware import run_device, decode_frames_fast
from polso_twin.model import pmap
from . import metrics as MT, refsim as RS, schema

UNITS = (100, 101, 102)
LABELS = P.CLASSES
ANSWER = set(P.CLASSES)


def site_days(n_days, hours, seed0):
    out = []
    for d in range(n_days):
        types = S.TRAIN_SUBTYPES if d % 3 else S.TRAIN_SUBTYPES + S.UNSEEN_SUBTYPES + S.CONFOUNDERS
        base = S.field_session(seed0 + d, subtypes=types, units=(UNITS[0],), age_range=(40.0, 40.0),
                               dur=int(hours * 3600))
        for u in UNITS:
            out.append(dataclasses.replace(base, unit_id=u, sensor_seed=(seed0 + d) * 10 + u, name=f"day{d}",
                                           tags=dict(day=d, unit=u, subtypes=base.tags["subtypes"])))
    return out


def _worker(args):
    sc, qm = args
    streams, truth = run_scenario(sc)
    out = run_device(streams, qm)
    cf1, _, ok = decode_frames_fast(streams["uart"], streams["uart_present"])
    rows = []
    for w, e in enumerate(out["t_end"]):
        a, b = int(e) - P.FW_WINDOW_S + 1, int(e) + 1
        lab, shares, total = window_truth(truth, int(e))
        pm = cf1[a:b][ok[a:b]].mean(0) if ok[a:b].any() else np.full(3, np.nan)
        good = out["status"][w] == "ok"
        unseen = any(st in S.UNSEEN_SUBTYPES or st in S.CONFOUNDERS for st in sc.tags["subtypes"])
        rows.append(dict(day=sc.tags["day"], unit=sc.unit_id, t_end=int(e), truth=lab, shares=shares, total=total, unseen=unseen,
                         pred=out["label"][w], status=out["status"][w],
                         conf=float(out["probs"][w].max()) if good else np.nan, probs=out["probs"][w],
                         pm1=pm[0], pm25=pm[1], pm10=pm[2],
                         temp_dev=float(streams["t_meas"][a:b].mean()), rh_dev=float(streams["rh_meas"][a:b].mean()),
                         feats=out["features"][w]))
    res = dict(rows=rows, day=sc.tags["day"], unit=sc.unit_id)
    if sc.unit_id == UNITS[0]:
        n = len(truth["t"])
        hrs = n // 3600
        comp = truth["comp_pm25"]
        res["pm_true_h"] = comp.sum(0)[:hrs * 3600].reshape(hrs, 3600).mean(1)
        res["gases_h"] = truth["gases"][:hrs * 3600].reshape(hrs, 3600, 6).mean(1)
        cls = {}
        for c, series in zip(truth["comp_cls"], comp):
            k = "confounder" if c is None else c
            cls[k] = cls.get(k, 0.0) + float(series.mean())
        res["cls_pm"] = cls
    return res


def _block_labels(df):
    """10-minute blocks: device label (majority of answers) and hidden-truth label."""
    out = []
    df = df.assign(blk=df["t_end"] // 600)
    for (day, unit, blk), g in df.groupby(["day", "unit", "blk"]):
        ans = g.loc[g["pred"].isin(ANSWER), "pred"]
        dev = ans.value_counts().idxmax() if len(ans) else ("uncertain" if (g["status"] == "ok").any() else "none")
        tr = g["truth"].value_counts()
        tru = tr.idxmax()
        # blocks with no clearly dominant source stay "mixed": an observer cannot call them, and they are not scored
        # (the same rule the 16-test campaign uses)
        out.append(dict(day=day, unit=unit, blk=blk, device=dev, truth=tru, conf=float(g["conf"].mean()),
                        unseen=bool(g["unseen"].any())))
    return pd.DataFrame(out)


def run(qm, n_days=12, hours=8, seed0=700_000, log=print):
    scs = site_days(n_days, hours, seed0)
    res = pmap(_worker, [(sc, qm) for sc in scs])
    rows = [r for x in res for r in x["rows"]]
    df = pd.DataFrame(rows)
    truth_days = {x["day"]: x for x in res if "pm_true_h" in x}
    log(f"  rehearsal: {len(scs)} unit-days ({n_days} days x {len(UNITS)} units x {hours} h), {len(df)} windows")
    return df, truth_days


def analyse(df, truth_days, n_train_days=4, seed=21, out_dir=None, log=print):
    rng = np.random.default_rng(seed)
    days = sorted(truth_days)
    R = dict(n_days=len(days), units=list(UNITS), n_train_days=n_train_days)

    # ---- reference instruments -------------------------------------------------------------
    ref = {}
    for d in days:
        t = truth_days[d]
        ref[d] = dict(pm=RS.fem_pm25(t["pm_true_h"], rng), gas=RS.ref_gases(t["gases_h"], rng),
                      shares=RS.tracer_shares(t["cls_pm"], rng))
    # ---- hourly device PM2.5 ----------------------------------------------------------------
    df = df.assign(hour=df["t_end"] // 3600)
    hourly = df.groupby(["day", "unit", "hour"]).agg(pm=("pm25", "mean"), rh=("rh_dev", "mean")).reset_index()
    hourly["ref"] = [ref[d]["pm"][h] for d, h in zip(hourly["day"], hourly["hour"])]
    hourly["true"] = [truth_days[d]["pm_true_h"][h] for d, h in zip(hourly["day"], hourly["hour"])]
    for c in LABELS:                                           # share of the hour the device attributed to each source
        sh = df.groupby(["day", "unit", "hour"])["pred"].apply(lambda x, c=c: float((x == c).mean())).rename(f"s_{c}")
        hourly = hourly.merge(sh.reset_index(), on=["day", "unit", "hour"])
    # corrections are scored by 4-fold cross-validation over whole days: every hour is predicted by a
    # correction that never saw its own day (the same design to use with the real 30+ day co-location)
    scols = [f"s_{c}" for c in LABELS]
    fold_of_day = {d: i % 4 for i, d in enumerate(days)}
    hourly["fold"] = hourly["day"].map(fold_of_day)
    hourly["pm_corr"] = np.nan
    hourly["pm_src"] = np.nan
    for f in range(4):
        tr, te = hourly["fold"] != f, hourly["fold"] == f
        c_rh = MT.fit_rh_correction(hourly.loc[tr, "pm"].values, hourly.loc[tr, "rh"].values, hourly.loc[tr, "ref"].values)
        hourly.loc[te, "pm_corr"] = MT.apply_rh_correction(c_rh, hourly.loc[te, "pm"].values, hourly.loc[te, "rh"].values)
        c_s = MT.fit_source_correction(hourly.loc[tr, "pm"].values, hourly.loc[tr, "rh"].values, hourly.loc[tr, scols].values,
                                       hourly.loc[tr, "ref"].values, alpha=100.0)
        hourly.loc[te, "pm_src"] = MT.apply_source_correction(c_s, hourly.loc[te, "pm"].values, hourly.loc[te, "rh"].values,
                                                              hourly.loc[te, scols].values)
    coef = MT.fit_rh_correction(hourly["pm"].values, hourly["rh"].values, hourly["ref"].values)
    coef_s = MT.fit_source_correction(hourly["pm"].values, hourly["rh"].values, hourly[scols].values, hourly["ref"].values, alpha=100.0)
    test = hourly
    R["rh_coef"] = [float(c) for c in coef]
    R["src_coef"] = [float(c) for c in coef_s]
    R["pm_hourly"] = hourly
    pm = {}
    for tag, col in (("raw", "pm"), ("corrected", "pm_corr"), ("source-aware", "pm_src")):
        pooled = MT.ols_epa(test["ref"], test[col])
        piv = test.pivot_table(index=["day", "hour"], columns="unit", values=col).values
        prec = MT.precision_units(piv)
        per_unit = {int(u): MT.ols_epa(g["ref"], g[col]) for u, g in test.groupby("unit")}
        daily = test.groupby(["day", "unit"]).agg(ref=("ref", "mean"), dev=(col, "mean")).reset_index()
        d_m = MT.ols_epa(daily["ref"], daily["dev"])
        pm[tag] = dict(pooled=pooled, precision=prec, per_unit=per_unit, daily=d_m,
                       verdict=MT.epa_verdict(pooled, prec), verdict_daily=MT.epa_verdict(d_m, prec))
    R["pm"] = pm
    R["ref_vs_truth_rmse"] = float(np.sqrt(np.mean((test["ref"] - test["true"]) ** 2)))

    # ---- gas channel against reference analysers (information only: MOX is not reference grade) ----
    g = df[df["status"] == "ok"].copy()
    g["hour"] = g["t_end"] // 3600
    F = [f"f{i}" for i in range(12, 19)]
    for i, c in enumerate(F):
        g[c] = [f[12 + i] for f in g["feats"]]
    gh = g.groupby(["day", "unit", "hour"])[F].mean().reset_index()
    R["gas"] = {}
    for gas in ("CO", "NO2", "SO2"):
        gh[gas] = [np.log1p(ref[d]["gas"][gas][h]) for d, h in zip(gh["day"], gh["hour"])]
        tr, te = gh[gh["day"] < n_train_days], gh[gh["day"] >= n_train_days]
        m = Ridge(alpha=1.0).fit(tr[F].values, tr[gas].values)
        pred = m.predict(te[F].values)
        ss = ((te[gas].values - te[gas].mean()) ** 2).sum()
        R["gas"][gas] = dict(r2=float(1 - ((te[gas].values - pred) ** 2).sum() / ss), n=int(len(te)))

    # ---- source attribution against 10-minute observer labels -------------------------------
    blk = _block_labels(df)
    obs_rng = np.random.default_rng(seed + 1)
    blk["observer"] = RS.observer_labels(list(blk["truth"]), obs_rng)
    blk = blk[blk["observer"].notna()]
    blk = blk[blk["observer"].isin(LABELS) | (blk["observer"] == "confounder")]
    ans = blk[blk["device"].isin(ANSWER)]
    clear = blk[blk["observer"].isin(LABELS)]

    def f1_vs(col):
        def stat(bl):
            d = pd.concat(bl)
            d = d[d[col].isin(LABELS)]
            return MT.macro_f1(d[col].values, np.where(d["device"].isin(ANSWER), d["device"], "none"), LABELS)
        return stat
    by_day = [g2 for _, g2 in blk.groupby("day")]
    R["f1_observer"] = MT.block_bootstrap(f1_vs("observer"), by_day, 300, seed)
    R["f1_truth"] = MT.block_bootstrap(f1_vs("truth"), by_day, 300, seed)
    sel = ans[ans["observer"].isin(LABELS)]
    R["sel_acc"] = float((sel["device"] == sel["observer"]).mean())
    # known source types only vs days that include source types never seen in training
    R["split"] = {}
    for name, flag in (("known source types", False), ("includes unseen source types or confounders", True)):
        sub_blocks = [g2 for _, g2 in blk[blk["unseen"] == flag].groupby("day")]
        if len(sub_blocks) >= 2:
            f, lo, hi = MT.block_bootstrap(f1_vs("observer"), sub_blocks, 300, seed)
            d = pd.concat(sub_blocks)
            d = d[d["observer"].isin(LABELS)]
            pc_ = MT.per_class(d["observer"].values, np.where(d["device"].isin(ANSWER), d["device"], "none"), LABELS)
            R["split"][name] = dict(f1=f, lo=lo, hi=hi, days=len(sub_blocks), blocks=int(len(d)),
                                    recall={c: v["recall"] for c, v in pc_.items()}, support={c: v["support"] for c, v in pc_.items()})
    R["coverage"] = float(clear["device"].isin(ANSWER).mean())
    R["per_class"] = MT.per_class(clear["observer"].values, np.where(clear["device"].isin(ANSWER), clear["device"], "none"), LABELS)
    labs_cols = LABELS + ["uncertain", "none"]
    cm = pd.crosstab(clear["observer"], clear["device"]).reindex(index=LABELS, columns=labs_cols, fill_value=0)
    R["confusion_observer"] = cm
    conf_blocks = blk[blk["observer"] == "confounder"]
    # independent episodes per class (runs of consecutive 10-min blocks with the same observer label, one unit)
    first = blk[blk["unit"] == UNITS[0]].sort_values(["day", "blk"])
    ep = {c: 0 for c in LABELS}
    prev = (None, None, None)
    for r_ in first.itertuples():
        if (r_.day, r_.observer) != prev[:2] or r_.blk != prev[2] + 1:
            if r_.observer in ep:
                ep[r_.observer] += 1
        prev = (r_.day, r_.observer, r_.blk)
    R["episodes"] = ep
    R["blocks_df"] = blk
    R["confounder_answer_rate"] = float(conf_blocks["device"].isin(ANSWER).mean()) if len(conf_blocks) else float("nan")
    R["n_blocks"] = int(len(blk))

    # ---- calibration on windows with a clear true label ----------------------------------------
    w = df[(df["status"] == "ok") & df["truth"].isin(LABELS) & df["pred"].isin(ANSWER)]
    e, rel = MT.ece(w["conf"].values, (w["pred"] == w["truth"]).values)
    R["ece"], R["reliability"] = e, rel

    # ---- daily apportionment vs tracer chemistry -------------------------------------------------
    cells = []
    for (d, u), gd in df[df["status"] == "ok"].groupby(["day", "unit"]):
        gd = gd[gd["pred"].isin(ANSWER)]
        tot = gd["pm25"].sum()
        for c in LABELS:
            cells.append(dict(day=d, unit=u, cls=c, device=gd.loc[gd["pred"] == c, "pm25"].sum() / max(tot, 1e-9),
                              reference=ref[d]["shares"][c]))
    cells = pd.DataFrame(cells).groupby(["day", "cls"]).mean(numeric_only=True).reset_index()
    from scipy.stats import spearmanr
    R["apportion"] = dict(mae=float((cells["device"] - cells["reference"]).abs().mean()),
                          rho=float(spearmanr(cells["device"], cells["reference"])[0]), cells=cells)

    # ---- agreement between co-located units ---------------------------------------------------------
    piv = df[df["status"] == "ok"].pivot_table(index=["day", "t_end"], columns="unit", values="pred", aggfunc="first")
    piv = piv.dropna()
    both = piv[piv.apply(lambda r: all(x in ANSWER for x in r), axis=1)]
    pairs = [(a, b) for i, a in enumerate(UNITS) for b in UNITS[i + 1:]]
    R["unit_agreement"] = float(np.mean([(both[a] == both[b]).mean() for a, b in pairs])) if len(both) else float("nan")

    # ---- how much data is enough: F1 interval width vs number of days ---------------------------------
    stat = f1_vs("observer")
    curve = []
    for k in (2, 3, 4, 6, 8, 12):
        if k > len(by_day):
            continue
        vals = [stat([by_day[i] for i in rng.integers(0, len(by_day), k)]) for _ in range(200)]
        lo, hi = np.quantile(vals, [0.025, 0.975])
        curve.append((k, float(hi - lo)))
    R["ci_curve"] = curve
    R["wilson"] = [(n, *MT.wilson(int(round(0.9 * n)), n)) for n in (10, 20, 30, 50, 100, 200)]

    # ---- pre-registered acceptance criteria -------------------------------------------------------------
    c = pm["corrected"]
    crit = [
        ("PM2.5 slope vs FEM (EPA 1.0 +/- 0.35), humidity-corrected, cross-validated by day", c["verdict"]["slope"], f"{c['pooled']['slope']:.2f}"),
        ("PM2.5 intercept (EPA -5 to +5 ug/m3)", c["verdict"]["intercept"], f"{c['pooled']['intercept']:.1f}"),
        ("PM2.5 linearity R2 >= 0.70 (EPA)", c["verdict"]["R2"], f"{c['pooled']['r2']:.2f}"),
        ("PM2.5 error RMSE <= 7 ug/m3 or NRMSE <= 30 % (EPA)", c["verdict"]["error"], f"{c['pooled']['rmse']:.1f} / {c['pooled']['nrmse']:.0%}"),
        ("PM2.5 unit-to-unit precision SD <= 5 or CV <= 30 % (EPA)", c["verdict"]["precision"], f"SD {c['precision']['sd']:.1f}, CV {c['precision']['cv']:.0%}"),
        ("Source macro-F1 vs observer log >= 0.75", R["f1_observer"][0] >= 0.75, f"{R['f1_observer'][0]:.2f} [{R['f1_observer'][1]:.2f}-{R['f1_observer'][2]:.2f}]"),
        ("Selective accuracy >= 0.90 at coverage >= 0.70", R["sel_acc"] >= 0.90 and R["coverage"] >= 0.70, f"{R['sel_acc']:.0%} @ {R['coverage']:.0%}"),
        ("Calibration error ECE <= 0.10", R["ece"] <= 0.10, f"{R['ece']:.3f}"),
        ("Co-located units agree on >= 85 % of windows", R["unit_agreement"] >= 0.85, f"{R['unit_agreement']:.0%}"),
    ]
    if np.isfinite(R["confounder_answer_rate"]):
        crit.append(("Confounder (cooking, incense) blocks answered <= 30 %", R["confounder_answer_rate"] <= 0.30,
                     f"{R['confounder_answer_rate']:.0%}"))
    R["criteria"] = [dict(name=n, passed=bool(p), value=v) for n, p, v in crit]

    # ---- templates for the real campaign -------------------------------------------------------------------
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        first = df[(df["unit"] == UNITS[0]) & (df["day"] < 2)]
        schema.to_log_frame(first.to_dict("records")).to_csv(os.path.join(out_dir, "device_log_example.csv"), index=False)
        rows = []
        for d in days[:2]:
            for h, v in enumerate(ref[d]["pm"]):
                rows.append(dict(ts_utc=(schema.EPOCH + pd.Timedelta(days=d, hours=h)).isoformat(), pm25_fem=round(float(v), 2),
                                 **{f"{k.lower()}_ppb": round(float(x[h]), 1) for k, x in ref[d]["gas"].items()}))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "reference_hourly_example.csv"), index=False)
        pd.DataFrame([dict(date=(schema.EPOCH + pd.Timedelta(days=d)).date().isoformat(),
                           **{f"{c}_share": round(v, 3) for c, v in ref[d]["shares"].items()}) for d in days]
                     ).to_csv(os.path.join(out_dir, "reference_daily_apportionment_example.csv"), index=False)
        b2 = blk[blk["day"] < 2]
        pd.DataFrame(dict(day=b2["day"], block_start_min=(b2["blk"] * 10), dominant_source=b2["observer"])
                     ).drop_duplicates(["day", "block_start_min"]).to_csv(os.path.join(out_dir, "observer_log_example.csv"), index=False)
    n_pass = sum(x["passed"] for x in R["criteria"])
    log(f"  verification rehearsal: {n_pass}/{len(R['criteria'])} criteria met; F1 vs observer {R['f1_observer'][0]:.2f}, "
        f"vs truth {R['f1_truth'][0]:.2f}")
    return R
