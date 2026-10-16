"""
PMS7003 duty-cycling policies: how much battery can be saved, and how many source events are missed?

The PMS fan + laser draw ~80 mA at 5 V and dominate the battery budget, whereas the BME688 gas sensor
is cheap to keep running. Policies decide, from information the device itself has, when the PM sensor
is powered. Valid PM frames start 30 s after power-on (datasheet stabilisation time); a 60 s window
needs 60 s of valid frames, so one burst is 90 s long.

  continuous        fan always on (reference)
  periodic 1/3      90 s on every 270 s
  periodic 1/6      90 s on every 540 s
  adaptive          90 s every 15 min, plus the PM sensor is woken when the gas signal departs from its
                    recent behaviour (computed on the device from the BME688 features alone)
"""
import numpy as np

from polso_twin import params as P
from polso_twin.simulate import run_scenario, window_truth
from polso_twin.firmware import run_device
from polso_twin.model import pmap

WARM = 30
BURST = 90
ANSWER = set(P.CLASSES)


def gas_anomaly(feats, status, thr_jump, history=20):
    """Per-window flag from the gas features only: the mean ln-R deviation (f12) or the heater-profile
    shape (f13-f18) departs from its own median over the previous `history` windows (10 min)."""
    sig = np.column_stack([feats[:, 12], feats[:, 13:19]])
    flag = np.zeros(len(sig), bool)
    hist = []
    for w in range(len(sig)):
        if status[w] != "ok":
            continue
        if len(hist) >= 3:
            med = np.median(np.array(hist[-history:]), axis=0)
            flag[w] = (np.abs(sig[w] - med).max() > thr_jump)
        hist.append(sig[w])
    return flag


def bursts_periodic(n, period):
    return [(s, min(s + BURST, n)) for s in range(0, n, period)]


def bursts_adaptive(n, t_end, flag, base_period=900, hold=120):
    b = bursts_periodic(n, base_period)
    for te, fl in zip(t_end, flag):
        if fl:
            s = int(np.ceil((te + 1) / 30.0) * 30)
            b.append((s, min(s + max(BURST, hold), n)))
    # merge
    b.sort()
    out = [list(b[0])]
    for s, e in b[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def mask_from_bursts(n, bursts):
    valid = np.zeros(n, bool)
    on = 0
    for s, e in bursts:
        valid[min(s + WARM, e):e] = True
        on += e - s
    return valid, on / n


HOLD_S = 600          # a held answer is shown for at most 10 minutes


def _evaluate(out, truth):
    """Counts for (a) fresh answers, (b) the answer a user would see (last answer held <= 10 min),
    (c) episodes: runs of >= 3 windows with the same dominant source, and time to first correct answer."""
    t_end = out["t_end"]
    labs = [window_truth(truth, int(e))[0] for e in t_end]
    clear = answered = correct = 0
    held_ok = held_clear = 0
    last, t_last = None, -1e9
    ep = []                                   # (truth label, start index)
    for w, (e, lab) in enumerate(zip(t_end, labs)):
        pred = out["label"][w]
        if pred in ANSWER:
            last, t_last = pred, e
        if lab not in ANSWER:
            continue
        clear += 1
        if pred in ANSWER:
            answered += 1
            correct += pred == lab
        held_clear += 1
        held_ok += (last == lab) and (e - t_last <= HOLD_S)
    detected = n_ep = 0
    delays = []
    w = 0
    while w < len(labs):
        j = w
        while j + 1 < len(labs) and labs[j + 1] == labs[w]:
            j += 1
        if labs[w] in ANSWER and labs[w] != "background" and j - w + 1 >= 3:
            n_ep += 1
            hit = [k for k in range(w, j + 1) if out["label"][k] == labs[w]]
            if hit:
                detected += 1
                delays.append(float(t_end[hit[0]] - t_end[w]))
        w = j + 1
    return dict(clear=clear, answered=answered, correct=correct, held_ok=held_ok, held_clear=held_clear,
                episodes=n_ep, detected=detected, delays=delays)


def _worker(args):
    sc, qm, thr = args
    streams, truth = run_scenario(sc)
    n = len(streams["gas_r"])
    base = run_device(streams, qm)
    flag = gas_anomaly(base["features"], base["status"], thr)
    res = {}
    plans = {"continuous": [(0, n)], "periodic 1/3": bursts_periodic(n, 270),
             "periodic 1/6": bursts_periodic(n, 540),
             "adaptive": bursts_adaptive(n, base["t_end"], flag)}
    for name, bursts in plans.items():
        if name == "continuous":
            out, duty = base, 1.0
        else:
            valid, duty = mask_from_bursts(n, bursts)
            s2 = dict(streams)
            s2["uart_present"] = streams["uart_present"] & valid
            out = run_device(s2, qm)
        res[name] = dict(_evaluate(out, truth), duty=duty)
    # flag quality against truth: is a local source present in the window?
    tp = fp = fn = tn = 0
    for w, e in enumerate(base["t_end"]):
        lab, _, _ = window_truth(truth, int(e))
        local = lab != "background"
        tp += local and flag[w]
        fn += local and not flag[w]
        fp += (not local) and flag[w]
        tn += (not local) and not flag[w]
    res["flag"] = dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    return res


def run(qm, scenarios, thr, log=print):
    per = pmap(_worker, [(sc, qm, thr) for sc in scenarios])
    pol = {}
    for name in ("continuous", "periodic 1/3", "periodic 1/6", "adaptive"):
        g = lambda k: sum(r[name][k] for r in per)
        delays = [d for r in per for d in r[name]["delays"]]
        pol[name] = dict(
            duty=float(np.mean([r[name]["duty"] for r in per])),
            held_acc=g("held_ok") / max(g("held_clear"), 1),
            sel_acc=g("correct") / max(g("answered"), 1),
            fresh_frac=g("answered") / max(g("clear"), 1),
            ep_recall=g("detected") / max(g("episodes"), 1), episodes=g("episodes"),
            delay_med=float(np.median(delays)) if delays else float("nan"),
            delay_p90=float(np.quantile(delays, 0.9)) if delays else float("nan"))
    fl = {k: sum(r["flag"][k] for r in per) for k in ("tp", "fp", "fn", "tn")}
    fl["recall"] = fl["tp"] / max(fl["tp"] + fl["fn"], 1)
    fl["precision"] = fl["tp"] / max(fl["tp"] + fl["fp"], 1)
    fl["flag_rate"] = (fl["tp"] + fl["fp"]) / max(sum(fl.values()), 1)
    log("    " + "; ".join(f"{k}: duty {v['duty']:.0%}, held acc {v['held_acc']:.0%}, episodes {v['ep_recall']:.0%}, "
                           f"delay {v['delay_med']:.0f}s" for k, v in pol.items()))
    return pol, fl


def quiet_days(unit, seed0, n=6, hours=8):
    """Quiet site: low background, two short source episodes (40-80 min) per day."""
    from polso_twin.environment import Scenario, Source
    rng = np.random.default_rng(seed0)
    kinds = ["diesel_traffic", "biomass_flaming", "road_dust", "coal_combustion", "construction_dust"]
    out = []
    for k in range(n):
        srcs = []
        for _ in range(2):
            t0 = rng.uniform(0.5, hours - 1.5) * 3600
            srcs.append(Source(str(rng.choice(kinds)), float(rng.uniform(60, 160)), bearing=float(rng.uniform(0, 360)),
                               t_on=float(t0), t_off=float(t0 + rng.uniform(40, 80) * 60)))
        out.append(Scenario(f"quiet{k}", int(hours * 3600), seed=seed0 + k, start_hour=8.0,
                            t_mean=float(rng.uniform(15, 32)), rh_mean=float(rng.uniform(30, 75)),
                            u_mean=float(rng.uniform(1.0, 3.0)), dir_mean=float(rng.uniform(0, 360)),
                            bg_pm25=float(rng.uniform(12, 30)), sources=srcs, unit_id=unit, age_days=40.0,
                            tags=dict(kind="quiet")))
    return out
