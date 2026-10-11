"""
Virtual test campaign for the PolSo device. Each test has an acceptance criterion that
was fixed *before* running it; results are reported as PASS / FAIL / INFO.
"""
import os
import subprocess
import struct
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from . import params as P
from . import scenarios as S
from . import model as M
from .environment import Scenario, Source
from .simulate import run_scenario, window_truth, CONFOUNDER
from .firmware import run_device, qmodel_forward

ANSWER = set(P.CLASSES)
DESIGNS = ["baseline", "hooded", "multi_hole"]


# ---------------------------------------------------------------------------
# Evaluation plumbing
# ---------------------------------------------------------------------------
def _eval_one(args):
    sc, qm, dev_kw = args
    streams, truth = run_scenario(sc)
    out = run_device(streams, qm, **dev_kw)
    rows = []
    for w, e in enumerate(out["t_end"]):
        lab, shares, total = window_truth(truth, int(e))
        rows.append(dict(scenario=sc.name, seed=sc.seed, unit=sc.unit_id, t_end=int(e),
                         truth=lab, shares=shares, total=total, pred=out["label"][w],
                         status=out["status"][w], probs=out["probs"][w],
                         probs_raw=out["probs_raw"][w], feat=out["features"][w], **sc.tags))
    extra = dict(pms_errors=out["pms_errors"], sd_lost=out["sd_lost"], sd_written=out["sd_written"],
                 sd_pending=out["sd_pending"])
    return rows, extra


def run_eval(scenarios, qm, return_extra=False, **dev_kw):
    res = M.pmap(_eval_one, [(sc, qm, dev_kw) for sc in scenarios])
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    return (df, [e for _, e in res]) if return_extra else df


def metrics(df, pred_col="pred"):
    """Scores on windows with a clear single-class truth. Abstentions count as errors in F1."""
    d = df[df["truth"].isin(P.CLASSES) & (df["status"] == "ok")]
    if len(d) == 0:
        return dict(n=0, coverage=np.nan, sel_acc=np.nan, acc=np.nan, macro_f1=np.nan, conf_wrong=np.nan)
    pred = d[pred_col].astype(str)
    answered = pred.isin(ANSWER)
    correct = pred == d["truth"]
    labels = P.CLASSES + [P.UNCERTAIN]
    return dict(n=int(len(d)), coverage=float(answered.mean()),
                sel_acc=float(correct[answered].mean()) if answered.any() else np.nan,
                acc=float(correct.mean()),
                macro_f1=float(f1_score(d["truth"], pred, labels=P.CLASSES, average="macro", zero_division=0)),
                conf_wrong=float((answered & ~correct).mean()))


def relabel(df, probs, thr):
    lab = np.where(probs.max(1) >= thr, np.array(P.CLASSES, object)[probs.argmax(1)], P.UNCERTAIN)
    return np.where(df["status"] == "ok", lab, df["status"])


def _result(tid, name, criterion, passed, summary, data=None, info=False):
    return dict(id=tid, name=name, criterion=criterion,
                verdict="INFO" if info else ("PASS" if passed else "FAIL"),
                summary=summary, data=data or {})


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_all(out_dir, quick=False, log=print):
    t0 = time.time()
    chamber, field = S.training_sets()
    val = S.validation_set()
    if quick:
        chamber, field, val = chamber[::4], field[::4], val[::3]
    Xc, yc, _ = M.build_dataset(chamber)
    Xf, yf, _ = M.build_dataset(field)
    Xv, yv, _ = M.build_dataset(val)
    log(f"  datasets: chamber {Xc.shape[0]} windows, field {Xf.shape[0]}, validation {Xv.shape[0]} "
        f"({time.time() - t0:.0f} s)")
    Xb, yb = np.vstack([Xc, Xf]), np.concatenate([yc, yf])
    models = dict(
        A_chamber=M.train_mlp(Xc, yc),
        B_fused=M.train_mlp(Xb, yb),
        B_pm_only=M.train_mlp(Xb, yb, P.PM_FEATURES),
        B_gas_only=M.train_mlp(Xb, yb, P.GAS_FEATURES),
        B_rf=M.train_rf(Xb, yb),
    )
    qm = M.quantize(models["B_fused"])
    pv, _ = qmodel_forward(qm, Xv)
    qm["conf_thr"] = M.choose_threshold(pv, yv)
    # chamber-only deployable model, threshold chosen the same way
    qa = M.quantize(models["A_chamber"])
    qa["conf_thr"] = M.choose_threshold(qmodel_forward(qa, Xv)[0], yv)
    M.save_qmodel(qm, os.path.join(out_dir, "qmodel_B.json"))
    wbytes = M.export_c_header(qm, os.path.join(os.path.dirname(out_dir), "firmware_c", "model_params.h"))
    log(f"  trained in {time.time() - t0:.0f} s; confidence gate = {qm['conf_thr']:.2f}")
    return dict(models=models, qm=qm, qa=qa, weights_bytes=wbytes,
                n_train=dict(chamber=int(M.trainable(yc).sum()), field=int(M.trainable(yf).sum())),
                Xv=Xv, yv=yv)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_generalisation(T, quick):
    """T00: chamber-only vs chamber+field model on unseen field sessions and units."""
    n = 60 if quick else 220
    scs = [S.field_session(400_000 + i, units=tuple(S.TEST_UNITS)) for i in range(n)]
    df = run_eval(scs, T["qm"])
    mB = metrics(df)
    dA = df.copy()
    dA["pred"] = relabel(df, qmodel_forward(T["qa"], np.stack(df["feat"]))[0], T["qa"]["conf_thr"])
    mA = metrics(dA)
    # float vs int8
    ok = df["status"] == "ok"
    fl = M.float_predict(T["models"]["B_fused"], np.stack(df["feat"]))
    dF = df.copy()
    dF["pred"] = relabel(df, fl, 0.0)
    dQ = df.copy()
    dQ["pred"] = relabel(df, np.stack(df["probs_raw"]), 0.0)
    q_loss = metrics(dF)["acc"] - metrics(dQ)["acc"]
    return [
        _result("T00", "Generalisation to unseen field sessions + unseen units",
                "chamber+field model macro-F1 >= 0.85 on held-out field data",
                mB["macro_f1"] >= 0.85,
                f"Chamber+field model: macro-F1 {mB['macro_f1']:.2f}, coverage {mB['coverage']:.0%}, "
                f"selective accuracy {mB['sel_acc']:.0%}. Chamber-only model: macro-F1 {mA['macro_f1']:.2f}, "
                f"selective accuracy {mA['sel_acc']:.0%}. int8 quantisation cost {q_loss * 100:+.1f} pts accuracy.",
                dict(B=mB, A=mA, quant_loss=q_loss, df=df)),
    ], df


def test_chamber(T, quick):
    scs = S.chamber_set(3 if quick else 8, seed0=410_000, units=tuple(S.TEST_UNITS))
    df = run_eval(scs, T["qm"])
    m = metrics(df)
    return _result("T01", "Chamber tests, single sources (in-distribution)",
                   "macro-F1 >= 0.90", m["macro_f1"] >= 0.90,
                   f"macro-F1 {m['macro_f1']:.2f}, coverage {m['coverage']:.0%}, selective accuracy {m['sel_acc']:.0%} "
                   f"over {m['n']} windows on the 10 unseen pilot units.", dict(m=m, df=df))


def test_detection_limit(T, quick):
    strengths = [3, 6, 12, 25, 50, 100, 200]
    reps = 2 if quick else 4
    scs, meta = [], []
    rng = np.random.default_rng(420_000)
    reps_sub = {c: [s for s in S.TRAIN_SUBTYPES if P.SUBTYPES[s]["cls"] == c] for c in P.CLASSES[:4]}
    for c, subs in reps_sub.items():
        for st in strengths:
            for r in range(reps):
                sub = subs[r % len(subs)]
                sc = S.chamber_session(sub, st, seed=420_000 + len(scs), unit=int(rng.choice(S.TEST_UNITS)),
                                       bg=30.0, dur=1200)
                sc.tags.update(cls=c, strength=st)
                scs.append(sc)
    df = run_eval(scs, T["qm"])
    t_on = P.FW_WARMUP_S + 60
    plateau = df[(df["t_end"] >= t_on + 180) & (df["t_end"] <= 1200 - 240)]
    tab = plateau.assign(hit=plateau["pred"] == plateau["cls"]).groupby(["cls", "strength"])["hit"].mean().unstack()
    lim = {}
    for c in tab.index:
        ok = [s for s in strengths if tab.loc[c, s] >= 0.8]
        lim[c] = min(ok) if ok else None
    passed = all(v is not None and v <= 50 for v in lim.values())
    txt = ", ".join(f"{c.replace('_', ' ')} {('%d' % v) if v else '>200'}" for c, v in lim.items())
    return _result("T02", "Detection limit (excess PM2.5 over a 30 ug/m3 background)",
                   ">= 80 % correct attribution at <= 50 ug/m3 excess for every class", passed,
                   f"Smallest excess PM2.5 (ug/m3) attributed correctly >= 80 % of the time: {txt}.",
                   dict(table=tab, limits=lim))


def _controlled(seed, subtype, unit, **kw):
    srcs = [] if subtype is None else [Source(subtype, kw.pop("strength", 80.0), bearing=0.0)]
    base = dict(name="ctl", duration_s=1500, seed=seed, start_hour=11.0, t_mean=25.0, t_amp=0.5,
                rh_mean=50.0, u_mean=1.5, dir_mean=0.0, dir_sd=20.0, bg_pm25=30.0, sources=srcs,
                unit_id=unit, age_days=20.0, tags=dict(subtype=subtype))
    base.update(kw)
    return Scenario(**base)


def _sweep(T, key, values, quick, seed0, extra=None):
    reps = 1 if quick else 2
    scs = []
    for v in values:
        for st in S.TRAIN_SUBTYPES + [None]:
            for r in range(reps):
                sc = _controlled(seed0 + len(scs), st, S.TEST_UNITS[len(scs) % 10], **{key: v}, **(extra or {}))
                sc.tags[key] = v
                scs.append(sc)
    df = run_eval(scs, T["qm"])
    df = df[df["t_end"] >= 600]
    tab = {v: metrics(df[df[key] == v]) for v in values}
    return tab, df


def test_humidity(T, quick):
    vals = [20, 40, 60, 75, 85, 92, 97]
    tab, df = _sweep(T, "rh_mean", vals, quick, 430_000)
    ref = np.mean([tab[40]["acc"], tab[60]["acc"]])
    drop85 = ref - tab[85]["acc"]
    fog = df[(df["rh_mean"] >= 92) & (df["subtype"].isna())]
    fog_false = float(fog["pred"].isin(set(P.CLASSES) - {"background"}).mean()) if len(fog) else np.nan
    return _result("T03", "Humidity sweep 20-97 % RH (incl. fog)",
                   "accuracy drop at 85 % RH <= 10 pts vs 40-60 % RH; false source alarms in fog <= 10 %",
                   drop85 <= 0.10 and fog_false <= 0.10,
                   "Accuracy by RH: " + ", ".join(f"{v}% {tab[v]['acc']:.0%}" for v in vals)
                   + f". Drop at 85 % RH: {drop85 * 100:.0f} pts. Fog (RH>=92 %, no source): {fog_false:.0%} of windows "
                     f"falsely blamed on a source.", dict(table=tab, fog_false=fog_false))


def test_temperature(T, quick):
    vals = [5, 15, 25, 35, 45]
    tab, _ = _sweep(T, "t_mean", vals, quick, 440_000)
    accs = [tab[v]["acc"] for v in vals]
    spread = max(accs) - min(accs)
    return _result("T04", "Temperature sweep 5-45 C",
                   "accuracy spread across temperatures <= 10 pts", spread <= 0.10,
                   "Accuracy by temperature: " + ", ".join(f"{v} C {tab[v]['acc']:.0%}" for v in vals)
                   + f" (spread {spread * 100:.0f} pts).", dict(table=tab))


def test_wind_enclosure(T, quick):
    speeds = [0.3, 2.0, 4.0, 6.0]
    dirs = [0.0, 90.0, 180.0]
    reps = 1 if quick else 2
    scs = []
    subs = ["road_dust", "construction_dust", "diesel_traffic", "biomass_flaming", "coal_combustion", None]
    for enc in DESIGNS:
        for u in speeds:
            for d in dirs:
                for st in subs:
                    for r in range(reps):
                        sc = _controlled(450_000 + len(scs), st, S.TEST_UNITS[len(scs) % 10], u_mean=u,
                                         dir_mean=d, dir_sd=8.0, enclosure=enc, u_gust=0.2)
                        if sc.sources:
                            sc.sources[0].bearing = d
                        sc.tags.update(enclosure=enc, u=u, wdir=d)
                        scs.append(sc)
    df = run_eval(scs, T["qm"])
    df = df[df["t_end"] >= 600]
    acc = {enc: {u: metrics(df[(df["enclosure"] == enc) & (df["u"] == u)])["acc"] for u in speeds}
           for enc in DESIGNS}
    # PM10 bias for coarse dust at 4 m/s, relative to the same air through a loss-free inlet
    bias = {}
    for enc in DESIGNS:
        bias[enc] = {}
        for d in dirs:
            pm10 = {}
            for e2 in (enc, "ideal"):
                sc = _controlled(459_000 + int(d), "road_dust", 100, u_mean=4.0, dir_mean=d, dir_sd=5.0,
                                 enclosure=e2, u_gust=0.1, strength=150.0)
                sc.sources[0].bearing = d
                _, tr = run_scenario(sc)
                pm10[e2] = tr["pms_cf1"][600:, 2].mean()
            bias[enc][d] = float(pm10[enc] / max(pm10["ideal"], 1e-6) - 1)
    hood_drop = acc["hooded"][0.3] - acc["hooded"][4.0]
    return _result("T05", "Wind speed x direction x inlet design (PolSo-19 enclosures)",
                   "hooded design: accuracy drop at 4 m/s <= 5 pts vs calm",
                   hood_drop <= 0.05,
                   "Accuracy at 0.3 / 2 / 4 / 6 m/s - " + "; ".join(
                       f"{enc}: " + " / ".join(f"{acc[enc][u]:.0%}" for u in speeds) for enc in DESIGNS)
                   + ". PM10 bias at 4 m/s for dust (0/90/180 deg): " + "; ".join(
                       f"{enc} " + "/".join(f"{bias[enc][d] * 100:+.0f}%" for d in dirs) for enc in DESIGNS)
                   + ". Model trained on hooded-enclosure data only.",
                   dict(acc=acc, bias=bias))


def test_mixtures(T, quick):
    reps = 1 if quick else 3
    pairs = [("diesel_traffic", "road_dust"), ("biomass_flaming", "diesel_traffic"),
             ("coal_combustion", "road_dust"), ("biomass_smoldering", "coal_combustion"),
             ("petrol_traffic", "solvent_industry"), ("construction_dust", "biomass_flaming")]
    scs = []
    for a, b in pairs:
        for ratio in (0.5, 0.75):
            for r in range(reps):
                sc = _controlled(460_000 + len(scs), a, S.TEST_UNITS[len(scs) % 10], strength=160 * ratio,
                                 bg_pm25=20.0)
                sc.sources.append(Source(b, 160 * (1 - ratio), bearing=10.0))
                sc.tags.update(pair=f"{a}+{b}", ratio=ratio)
                scs.append(sc)
    df = run_eval(scs, T["qm"])
    df = df[df["t_end"] >= 600]
    answered = df["pred"].isin(ANSWER)
    contrib = df.apply(lambda r: r["pred"] in r["shares"] and r["shares"][r["pred"]] >= 0.2, axis=1)
    hit = float(contrib[answered].mean())
    dom = df[(df["ratio"] == 0.75) & df["truth"].isin(P.CLASSES)]
    dom_acc = float((dom["pred"] == dom["truth"]).mean()) if len(dom) else np.nan
    return _result("T06", "Mixed sources (50/50 and 75/25 pairs)",
                   "answered label is a real contributor >= 85 %; dominant source found >= 70 % at 75/25",
                   hit >= 0.85 and dom_acc >= 0.70,
                   f"Answered labels naming a real contributor: {hit:.0%}. Abstained on {1 - answered.mean():.0%} "
                   f"of mixed windows. Dominant source correctly named at 75/25: {dom_acc:.0%}.",
                   dict(hit=hit, dom_acc=dom_acc, abstain=float(1 - answered.mean())))


def test_latency(T, quick):
    reps = 2 if quick else 5
    scs = []
    for st in S.TRAIN_SUBTYPES:
        for r in range(reps):
            sc = _controlled(470_000 + len(scs), st, S.TEST_UNITS[len(scs) % 10], strength=100.0)
            sc.sources[0].t_on = 900.0
            sc.tags.update(cls=P.SUBTYPES[st]["cls"])
            scs.append(sc)
    df = run_eval(scs, T["qm"])
    lat = []
    for seed, g in df.groupby("seed"):
        g = g[g["t_end"] >= 900]
        hit = g[g["pred"] == g["cls"].iloc[0]]
        lat.append(float(hit["t_end"].iloc[0] - 900) if len(hit) else np.nan)
    lat = np.array(lat)
    med = float(np.nanmedian(lat))
    missed = float(np.isnan(lat).mean())
    return _result("T07", "Response time to a new source",
                   "median time-to-correct-label <= 90 s and <= 10 % never detected",
                   med <= 90 and missed <= 0.10,
                   f"Median {med:.0f} s, 90th percentile {np.nanpercentile(lat, 90):.0f} s from source onset to first "
                   f"correct label (60 s window, 30 s hop, EMA smoothing); {missed:.0%} never detected.",
                   dict(latency=lat.tolist()))


def test_drift(T, quick):
    ages = [0, 30, 90, 180, 365]
    n = 25 if quick else 60
    out = {"tracking": {}, "frozen": {}}
    for age in ages:
        scs = [S.field_session(480_000 + i, units=tuple(S.TEST_UNITS), age_days=float(age)) for i in range(n)]
        for mode, trk in (("tracking", True), ("frozen", False)):
            out[mode][age] = metrics(run_eval(scs, T["qm"], track_baseline=trk))["macro_f1"]
    drop = out["tracking"][0] - out["tracking"][180]
    return _result("T08", "Sensor ageing / drift 0-365 days",
                   "macro-F1 drop at 180 days <= 10 pts with on-device baseline tracking", drop <= 0.10,
                   "macro-F1 with baseline tracking: " + ", ".join(f"day {a} {out['tracking'][a]:.2f}" for a in ages)
                   + ". With frozen factory baseline: " + ", ".join(f"day {a} {out['frozen'][a]:.2f}" for a in ages) + ".",
                   out)


def test_units(T, quick):
    n = 15 if quick else 40
    per = {}
    for u in S.TEST_UNITS:
        scs = [S.field_session(490_000 + i, units=(u,)) for i in range(n)]
        per[u] = metrics(run_eval(scs, T["qm"]))["macro_f1"]
    v = np.array(list(per.values()))
    return _result("T09", "Unit-to-unit variation (10 pilot prototypes)",
                   "std of macro-F1 across units <= 0.05 and worst unit >= 0.80",
                   v.std() <= 0.05 and v.min() >= 0.80,
                   f"macro-F1 mean {v.mean():.2f}, std {v.std():.3f}, worst unit {v.min():.2f}, best {v.max():.2f} "
                   f"(same scenarios replayed on every unit).", dict(per_unit=per))


def test_unseen(T, quick):
    reps = 3 if quick else 8
    scs = []
    for st in S.UNSEEN_SUBTYPES + S.CONFOUNDERS:
        for r in range(reps):
            sc = _controlled(500_000 + len(scs), st, S.TEST_UNITS[len(scs) % 10], strength=90.0,
                             rh_mean=float(40 + 10 * (r % 4)))
            sc.tags.update(kind="unseen", subtype=st)
            scs.append(sc)
    for r in range(reps):                                          # fog, no source
        sc = _controlled(505_000 + r, None, S.TEST_UNITS[r % 10], rh_mean=98.0, t_amp=0.3, bg_pm25=25.0)
        sc.tags.update(subtype="fog")
        scs.append(sc)
    df = run_eval(scs, T["qm"])
    df = df[(df["t_end"] >= 600) & (df["status"] == "ok")]
    tab = {}
    for st, g in df.groupby("subtype"):
        spec = P.SUBTYPES.get(st)
        correct = spec["cls"] if spec else "background"
        pred = g["pred"]
        tab[st] = dict(correct=float((pred == correct).mean()) if correct else 0.0,
                       uncertain=float((pred == P.UNCERTAIN).mean()),
                       wrong=float((pred.isin(ANSWER) & (pred != correct)).mean()),
                       top=pred.value_counts().idxmax())
    unseen_wrong = np.mean([tab[s]["wrong"] for s in S.UNSEEN_SUBTYPES])
    conf_wrong = np.mean([tab[s]["wrong"] for s in S.CONFOUNDERS])
    fog = tab["fog"]["wrong"]
    return _result("T10", "Unseen source subtypes and confounders",
                   "confident-wrong <= 25 % on unseen subtypes and <= 30 % on confounders (cooking, incense); fog <= 10 %",
                   unseen_wrong <= 0.25 and conf_wrong <= 0.30 and fog <= 0.10,
                   "; ".join(f"{s}: correct {v['correct']:.0%} / uncertain {v['uncertain']:.0%} / wrong {v['wrong']:.0%}"
                             for s, v in tab.items()) + ".", dict(table=tab))


def test_field_days(T, quick):
    days = S.field_days()
    if quick:
        for d in days:
            d.duration_s = 6 * 3600
    df = run_eval(days, T["qm"])
    per = {name: metrics(g) for name, g in df.groupby("scenario")}
    m = metrics(df)
    return _result("T11", "24-hour field days in six environments",
                   "overall selective accuracy >= 80 % at coverage >= 70 %",
                   m["sel_acc"] >= 0.80 and m["coverage"] >= 0.70,
                   f"All days: selective accuracy {m['sel_acc']:.0%}, coverage {m['coverage']:.0%}, macro-F1 {m['macro_f1']:.2f}. "
                   + "; ".join(f"{k}: {v['sel_acc']:.0%} @ {v['coverage']:.0%}" for k, v in per.items()) + ".",
                   dict(per_day=per, overall=m, df=df))


def test_faults(T, quick):
    base = dict(name="fault", duration_s=7200, seed=520_000, start_hour=9.0, t_mean=28, rh_mean=55,
                u_mean=1.5, dir_mean=0.0, bg_pm25=60.0, unit_id=101, age_days=40.0,
                sources=[Source("diesel_traffic", 70, bearing=0.0)])
    cases = {
        "none": {}, "uart_bit_errors_0.1%": dict(pms_byte_error=1e-3),
        "uart_bit_errors_1%+slips": dict(pms_byte_error=1e-2, pms_byte_slip=0.02),
        "pms_frames_dropped_30%": dict(pms_frame_drop=0.30), "pms_dead_after_1h": dict(pms_dead_after=3600),
        "pms_fan_stall_after_1h": dict(pms_fan_stall_after=3600), "bme_i2c_errors_40%": dict(bme_i2c_error=0.40),
        "bme_dead_after_1h": dict(bme_dead_after=3600), "sd_write_fail_20%": dict(sd_fail=0.20),
    }
    res, crashed = {}, []
    for name, f in cases.items():
        sc = Scenario(**base, faults=f)
        try:
            df, extra = run_eval([sc], T["qm"], return_extra=True)
            ex = extra[0]
        except Exception as e:               # a crash is a test failure, not a campaign abort
            crashed.append(f"{name}: {e!r}")
            continue
        after = df[df["t_end"] >= 3700]
        res[name] = dict(acc=metrics(df)["acc"],
                         fault_flag_after=float((after["pred"] == P.FAULT).mean()),
                         confident_after=float(after["pred"].isin(ANSWER).mean()),
                         pms_errors=int(ex["pms_errors"]) if ex else 0,
                         sd_lost=int(ex["sd_lost"]) if ex else 0)
    dead_ok = all(res[k]["fault_flag_after"] >= 0.95 for k in ("pms_dead_after_1h", "bme_dead_after_1h") if k in res)
    noise_ok = abs(res["none"]["acc"] - res["uart_bit_errors_0.1%"]["acc"]) <= 0.03
    stall = res.get("pms_fan_stall_after_1h", {})
    passed = not crashed and dead_ok and noise_ok and stall.get("confident_after", 1) <= 0.05
    return _result("T12", "Fault injection (UART, sensor death, fan stall, I2C, SD card)",
                   "no crashes; dead sensors flagged >= 95 %; 0.1 % UART errors cost <= 3 pts; "
                   "stalled-fan windows not reported as a confident source (<= 5 %)",
                   passed,
                   "; ".join(f"{k}: acc {v['acc']:.0%}, flagged-after {v['fault_flag_after']:.0%}, "
                             f"confident-after {v['confident_after']:.0%}" for k, v in res.items())
                   + (f". CRASHES: {crashed}" if crashed else ". No crashes."),
                   dict(cases=res, crashed=crashed))


# ---------------------------------------------------------------------------
# T13: software-in-the-loop, C firmware vs Python twin
# ---------------------------------------------------------------------------
def _write_sil_input(streams, path):
    n = len(streams["gas_r"])
    with open(path, "wb") as f:
        f.write(b"PSIL" + struct.pack("<I", n) + struct.pack("<10f", *map(float, streams.get("b_tracked", streams["b0"]))))
        for i in range(n):
            if "uart_bytes" in streams:
                b = streams["uart_bytes"][i]
            else:
                b = streams["uart"][i].tobytes() if streams["uart_present"][i] else b""
            f.write(struct.pack("<H", len(b)) + b)
            f.write(struct.pack("<fBBff", float(streams["gas_r"][i]), int(streams["gas_step"][i]),
                                int(bool(streams["gas_ok"][i])), float(streams["t_meas"][i]),
                                float(streams["rh_meas"][i])))


def test_sil(T, root, quick):
    fw = os.path.join(root, "firmware_c")
    exe = os.path.join(fw, "sil.exe")
    build = subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-o", exe, "sil_main.c",
                            "polso_fw.c", "pms7003.c", "-lm"], cwd=fw, capture_output=True, text=True)
    if build.returncode != 0:
        return _result("T13", "Firmware C vs Python twin (software-in-the-loop)", "builds", False,
                       "gcc build failed: " + build.stderr[:500])
    warnings = build.stderr.strip()
    scs = [Scenario("sil_road", 5400 if not quick else 2400, seed=530_000, start_hour=8, rh_mean=70, u_mean=1.0,
                    bg_pm25=80, unit_id=102, sources=[Source("diesel_traffic", 60), Source("road_dust", 30, bearing=40)]),
           Scenario("sil_burn", 5400 if not quick else 2400, seed=530_001, rh_mean=45, u_mean=2.0, bg_pm25=40,
                    unit_id=103, age_days=90, sources=[Source("biomass_smoldering", 120)]),
           Scenario("sil_faulty", 5400 if not quick else 2400, seed=530_002, rh_mean=60, bg_pm25=50, unit_id=104,
                    sources=[Source("coal_combustion", 90)],
                    faults=dict(pms_byte_error=5e-3, pms_byte_slip=0.02, bme_i2c_error=0.1))]
    agree, n_total, max_dp, max_df, stats = 0, 0, 0.0, 0.0, ""
    code_map = {i: c for i, c in enumerate(P.CLASSES + [P.UNCERTAIN, P.FAULT, P.WARMUP])}
    for sc in scs:
        streams, _ = run_scenario(sc)
        if "uart_bytes" not in streams:
            streams["uart_bytes"] = [streams["uart"][i].tobytes() if streams["uart_present"][i] else b""
                                     for i in range(len(streams["gas_r"]))]
        py = run_device(streams, T["qm"])
        inp = os.path.join(fw, f"_{sc.name}.bin")
        outp = os.path.join(fw, f"_{sc.name}.csv")
        _write_sil_input(streams, inp)
        r = subprocess.run([exe, inp, outp], capture_output=True, text=True)
        stats = r.stderr.strip()
        c = pd.read_csv(outp)
        c_lab = c["label"].map(code_map).to_numpy()
        n_total += len(c)
        agree += int((c_lab == py["label"]).sum())
        ok = py["status"] == "ok"
        if ok.any():
            max_dp = max(max_dp, float(np.abs(c[[f"p{k}" for k in range(5)]].to_numpy()[ok] - py["probs"][ok]).max()))
            cf = c[[f"f{k}" for k in range(P.N_FEATURES)]].to_numpy()[ok]
            pf = py["features"][ok]
            max_df = max(max_df, float((np.abs(cf - pf) / (np.abs(pf) + 1e-2)).max()))
        os.remove(inp)
        os.remove(outp)
    kv = dict(x.split("=") for x in stats.split())
    state_b, w_b, macs = int(kv["sizeof(fw_state_t)"]), int(kv["model_weight_bytes"]), int(kv["macs"])
    rate = agree / n_total
    cycles = macs * 4 + 25_000                     # plain C MACs + feature maths (estimate)
    return _result("T13", "Firmware C vs Python twin (software-in-the-loop)",
                   "label agreement >= 99.5 %, max probability diff <= 0.01, RAM <= 32 KB, inference <= 10 ms",
                   rate >= 0.995 and max_dp <= 0.01 and state_b < 32_768 and cycles / 170e6 < 0.01,
                   f"Labels agree on {rate:.2%} of {n_total} windows (incl. a stream with UART corruption and I2C errors); "
                   f"max probability difference {max_dp:.1e}; max relative feature difference {max_df:.1e}. "
                   f"Firmware state {state_b / 1024:.1f} KB RAM, model {w_b / 1024:.1f} KB flash, {macs} MACs per inference, "
                   f"est. ~{cycles / 170e6 * 1e3:.2f} ms per window at 170 MHz (measure with the DWT cycle counter on the board)."
                   + (f" gcc warnings: {warnings[:200]}" if warnings else " Builds with -Wall -Wextra and no warnings."),
                   dict(agree=rate, max_dp=max_dp, max_df=max_df, state_bytes=state_b, weight_bytes=w_b,
                        macs=macs, est_ms=cycles / 170e6 * 1e3))


# ---------------------------------------------------------------------------
# T14: power and duty cycling
# ---------------------------------------------------------------------------
def battery_hours(pms_duty, nucleo=True, gps_psm=False):
    p = P.POWER
    v = p["battery_v"]
    pms_w = (pms_duty * p["pms_active_ma"] + (1 - pms_duty) * p["pms_sleep_ma"]) * 5.0 / 1000
    i_3v3 = (p["bme_ma"] + 0.15 * p["mcu_run_ma"] + 0.85 * p["mcu_sleep_ma"]
             + (p["nucleo_overhead_ma"] if nucleo else 0)
             + (p["gps_psm_ma"] if gps_psm else p["gps_ma"])
             + p["sd_idle_ma"] + p["sd_write_ma"] * p["sd_write_ms"] / 1000 * p["sd_writes_per_min"] / 60)
    i_batt = pms_w / (p["boost_eff"] * v) * 1000 + i_3v3
    return p["battery_mah"] * p["usable_frac"] / i_batt, i_batt


def test_power(T, field_df):
    modes = {
        "NUCLEO board, continuous": (1.0, True, False),
        "Custom PCB, continuous": (1.0, False, True),
        "Custom PCB, PMS 90 s on / 90 s off": (0.5, False, True),
        "Custom PCB, PMS 90 s on / 270 s off": (0.25, False, True),
    }
    tab = {}
    for name, (duty, nuc, psm) in modes.items():
        h, i = battery_hours(duty, nuc, psm)
        tab[name] = dict(hours=h, ma=i)
    # attribution under duty cycling, replayed from the field-day windows (raw probs, no EMA across gaps)
    d = field_df[field_df["status"] == "ok"].copy()
    thr = T["qm"]["conf_thr"]
    dc = {}
    for period in (180, 360):
        sel = d[(d["t_end"] - 89) % period == 0]
        sel = sel.assign(pred=relabel(sel, np.stack(sel["probs_raw"]), thr))
        m = metrics(sel)
        dc[period] = dict(sel_acc=m["sel_acc"], coverage_time=60 / period)
    cont = tab["Custom PCB, continuous"]["hours"]
    return _result("T14", "Battery life (2000 mAh Li-ion) and duty cycling",
                   ">= 24 h continuous sampling in the configuration as specified (NUCLEO + 2000 mAh)",
                   tab["NUCLEO board, continuous"]["hours"] >= 24,
                   "; ".join(f"{k}: {v['hours']:.1f} h ({v['ma']:.0f} mA avg)" for k, v in tab.items())
                   + f". Duty cycling keeps selective accuracy at {dc[180]['sel_acc']:.0%} (1/2 duty) and "
                     f"{dc[360]['sel_acc']:.0%} (1/4 duty) but only samples {dc[180]['coverage_time']:.0%} / "
                     f"{dc[360]['coverage_time']:.0%} of the time, so short events are missed.",
                   dict(modes=tab, duty=dc))


# ---------------------------------------------------------------------------
# T15: research question - does fusion help?
# ---------------------------------------------------------------------------
def test_fusion(T, gen_df, field_df):
    df = pd.concat([gen_df, field_df])
    df = df[df["status"] == "ok"]
    X = np.stack(df["feat"])
    out = {}
    for name in ("B_pm_only", "B_gas_only", "B_fused", "B_rf"):
        d = df.copy()
        d["pred"] = relabel(d, M.float_predict(T["models"][name], X), 0.0)
        out[name] = metrics(d)["macro_f1"]
    best_single = max(out["B_pm_only"], out["B_gas_only"])
    gain = out["B_fused"] - best_single
    return _result("T15", "Research question: PM-only vs gas-only vs fused",
                   "fused macro-F1 >= best single-sensor model + 0.05", gain >= 0.05,
                   f"macro-F1 (no abstention) on all held-out field windows: PM-only {out['B_pm_only']:.2f}, "
                   f"gas-only {out['B_gas_only']:.2f}, fused MLP {out['B_fused']:.2f} (+{gain:.2f}), "
                   f"fused random forest {out['B_rf']:.2f}.", out)
