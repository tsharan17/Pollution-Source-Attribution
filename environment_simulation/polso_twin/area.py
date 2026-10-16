"""
Area deployment: an array of PolSo devices across a 2 km x 2 km map, under six cases.

All devices share one environment per case (same weather, same emission events). Each
device's air is built from the map geometry: bearing and distance to every source decide
how strongly, and in which wind, that source reaches it. Each device then runs through
the full sensor models, firmware and int8 model like in the test campaign.
"""
import base64
import gzip
import json
import os
import pickle
import numpy as np

from . import params as P
from .environment import Scenario, Source
from .simulate import run_scenario, window_truth
from .firmware import run_device
from .model import pmap

D0 = 80.0          # m, distance at which a source's strength is defined
HOP = P.FW_HOP_S

# map sources: point (x, y) or line ((x1, y1), (x2, y2)); strength = PM2.5 excess at D0 (ug/m3)
MAP_SOURCES = [
    dict(id="highway_diesel", subtype="diesel_traffic", line=((-1000, -450), (1000, -450)), q=110, label="NH highway"),
    dict(id="highway_petrol", subtype="petrol_traffic", line=((-1000, -450), (1000, -450)), q=50, label="NH highway"),
    dict(id="highway_dust", subtype="road_dust", line=((-1000, -470), (1000, -470)), q=35, label="Highway shoulder dust"),
    dict(id="arterial", subtype="petrol_traffic", line=((150, -1000), (150, 1000)), q=35, label="Ring road"),
    dict(id="autos", subtype="two_stroke_traffic", line=((150, -1000), (150, 1000)), q=20, label="Ring road autos"),
    dict(id="coal_stack", subtype="coal_combustion", pt=(650, 620), q=140, label="Boiler stack"),
    dict(id="solvent", subtype="solvent_industry", pt=(820, 330), q=70, label="Paint shop"),
    dict(id="metal", subtype="metal_processing", pt=(480, 820), q=90, label="Foundry"),
    dict(id="farm_flaming", subtype="biomass_flaming", pt=(-760, 520), q=900, label="Stubble fire A"),
    dict(id="farm_smoulder", subtype="biomass_smoldering", pt=(-640, 160), q=600, label="Stubble fire B"),
    dict(id="construction", subtype="construction_dust", pt=(560, -760), q=130, label="Construction site"),
    dict(id="cooking", subtype="cooking", pt=(-420, -720), q=40, label="Street food stalls"),
    dict(id="trash", subtype="trash_burning", pt=(-220, -160), q=90, label="Waste dump"),
    dict(id="dust_storm", subtype="dust_storm", regional=True, q=300, label="Regional dust storm"),
]

GRID = [-750, -250, 250, 750]
DEVICES = [dict(id=f"P{k + 1:02d}", x=float(x), y=float(y), unit=100 + k)
           for k, (y, x) in enumerate((y, x) for y in GRID[::-1] for x in GRID)]

H = 3600
CASES = [
    dict(id="weekday", name="Weekday, clear", start=6.0, dur=6 * H, seed=700_001,
         met=dict(t_mean=29, t_amp=5, rh_mean=50, u_mean=2.0, dir_mean=250, dir_sd=35, bg_pm25=45),
         active={"highway_diesel": None, "highway_petrol": None, "highway_dust": None, "arterial": None,
                 "autos": None, "coal_stack": None, "solvent": (1 * H, 6 * H), "metal": None,
                 "construction": (2 * H, 6 * H), "cooking": (1 * H, 3 * H)},
         blurb="Morning rush hour, industry on, construction starts at 08:00."),
    dict(id="winter", name="Winter smog night", start=16.0, dur=6 * H, seed=700_002,
         met=dict(t_mean=13, t_amp=4, rh_mean=82, u_mean=0.7, dir_mean=300, dir_sd=40, bg_pm25=150),
         active={"highway_diesel": None, "highway_petrol": None, "arterial": None, "autos": None,
                 "coal_stack": None, "metal": None, "cooking": (2 * H, 4 * H), "trash": (2.5 * H, 5.5 * H)},
         blurb="Low wind, heavy regional haze (150 ug/m3), evening trash burning 18:30-21:30."),
    dict(id="crop", name="Crop-burning season", start=10.0, dur=6 * H, seed=700_003,
         met=dict(t_mean=25, t_amp=5, rh_mean=55, u_mean=2.5, dir_mean=280, dir_sd=25, bg_pm25=80),
         active={"highway_diesel": None, "highway_petrol": None, "arterial": None, "coal_stack": None,
                 "farm_flaming": (0.5 * H, 4.5 * H), "farm_smoulder": (3 * H, 6 * H)},
         blurb="Westerly wind carries stubble smoke from the farms across the town."),
    dict(id="duststorm", name="Pre-monsoon dust storm", start=12.0, dur=6 * H, seed=700_004,
         met=dict(t_mean=38, t_amp=4, rh_mean=20, u_mean=5.5, dir_mean=260, dir_sd=20, bg_pm25=70),
         active={"highway_diesel": None, "highway_petrol": None, "highway_dust": None, "arterial": None,
                 "construction": None, "dust_storm": (2 * H, 5 * H)},
         blurb="Hot, dry and windy; a regional dust front arrives at 14:00."),
    dict(id="fog", name="Monsoon fog morning", start=4.0, dur=6 * H, seed=700_005,
         met=dict(t_mean=22, t_amp=2, rh_mean=96, u_mean=0.6, dir_mean=200, dir_sd=50, bg_pm25=30),
         active={"highway_diesel": None, "highway_petrol": None, "arterial": None, "autos": None,
                 "cooking": (2.5 * H, 4.5 * H)},
         blurb="Near-saturated air: hygroscopic growth inflates particle counts."),
    dict(id="upset", name="Industrial upset + sensor faults", start=8.0, dur=6 * H, seed=700_006,
         met=dict(t_mean=30, t_amp=4, rh_mean=45, u_mean=1.8, dir_mean=45, dir_sd=25, bg_pm25=50),
         active={"highway_diesel": None, "highway_petrol": None, "arterial": None, "coal_stack": None,
                 "solvent": None, "metal": None},
         boost={"coal_stack": (2 * H, 4 * H, 3.0)},
         faults={"P06": dict(pms_dead_after=3 * H), "P10": dict(pms_fan_stall_after=2 * H),
                 "P13": dict(pms_byte_error=1e-2, pms_byte_slip=0.02), "P03": dict(bme_i2c_error=0.35)},
         blurb="North-easterly wind; the boiler stack triples its output 10:00-12:00. Four devices develop faults."),
]


def _geom(dev, src):
    """Distance (m) and bearing (deg, compass: 0 = north) from device to the source's nearest point."""
    x, y = dev["x"], dev["y"]
    if "pt" in src:
        sx, sy = src["pt"]
    else:
        (x1, y1), (x2, y2) = src["line"]
        vx, vy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * vx + (y - y1) * vy) / (vx * vx + vy * vy), 0, 1)
        sx, sy = x1 + t * vx, y1 + t * vy
    dx, dy = sx - x, sy - y
    return float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dx, dy)) % 360)


def _sources_for(case, dev):
    """Source list for one device. Same order for every device so emission events line up."""
    out = []
    for s in MAP_SOURCES:
        if s["id"] not in case["active"]:
            continue
        win = case["active"][s["id"]] or (0, 1e12)
        if s.get("regional"):
            q, brg = s["q"], case["met"]["dir_mean"]
        else:
            d, brg = _geom(dev, s)
            q = s["q"] * (D0 / max(d, D0)) ** 1.4
        out.append(Source(s["subtype"], float(q), bearing=brg, t_on=float(win[0]), t_off=float(win[1])))
    return out


def _run_device(args):
    case, k = args
    dev = DEVICES[k]
    srcs = _sources_for(case, dev)
    if case.get("boost"):
        # a temporary output increase is modelled as a second, co-located copy of the source
        ids = [s["id"] for s in MAP_SOURCES if s["id"] in case["active"]]
        for sid, (a, b, factor) in case["boost"].items():
            base = srcs[ids.index(sid)]
            srcs.append(Source(base.subtype, base.strength * (factor - 1), bearing=base.bearing, t_on=a, t_off=b))
    sc = Scenario(f"{case['id']}_{dev['id']}", case["dur"], seed=case["seed"], start_hour=case["start"],
                  sources=srcs, unit_id=dev["unit"], age_days=45.0, sensor_seed=case["seed"] * 100 + k,
                  faults=case.get("faults", {}).get(dev["id"], {}), **case["met"])
    streams, truth = run_scenario(sc)
    out = run_device(streams, QM)
    cls_index = {c: i for i, c in enumerate(P.CLASSES + [P.UNCERTAIN, P.FAULT, P.WARMUP, P.MIXED, "confounder"])}
    pred, conf, tru, pm_dev, pm_true, share, lab10, pm5, pm5lo, pm5hi = ([] for _ in range(10))
    truth_labels = []
    for w, e in enumerate(out["t_end"]):
        lab, shares, total = window_truth(truth, int(e))
        truth_labels.append(lab)
        pred.append(cls_index[out["label"][w]])
        lab10.append(cls_index[out["label10"][w]])
        conf.append(int(round(100 * out["probs"][w].max())) if out["status"][w] == "ok" else 0)
        tru.append(cls_index[lab])
        pm_dev.append(int(round(out["pm_raw"][w])) if np.isfinite(out["pm_raw"][w]) else -1)
        for lst, key in ((pm5, "pm5"), (pm5lo, "pm5_lo"), (pm5hi, "pm5_hi")):
            lst.append(int(round(out[key][w])) if np.isfinite(out[key][w]) else -1)
        pm_true.append(int(round(total)))
        local = {c: v for c, v in shares.items() if c != "background"}
        share.append(int(round(100 * max(local.values()))) if local else 0)
    # model internals per window for the STM32 tab: 25 features + 5 probabilities, int16 x1000, gzip + base64
    blob = np.clip(np.round(np.hstack([out["features"], out["probs"]]) * 1000), -32767, 32767).astype("<i2")
    blob = base64.b64encode(gzip.compress(blob.tobytes(), 6)).decode()
    acc = _accuracy(out, truth_labels, np.array(pm_true, float))
    met = None
    if k == 0:
        idx = np.arange(P.FW_WINDOW_S - 1, case["dur"], HOP)
        met = dict(u=np.round(truth["u"][idx], 2).tolist(), wdir=np.round(truth["wdir"][idx]).astype(int).tolist(),
                   temp=np.round(truth["temp"][idx], 1).tolist(), rh=np.round(truth["rh"][idx]).astype(int).tolist(),
                   activity=np.round(truth["src_activity"][:, idx], 2).tolist())
    return k, dict(pred=pred, conf=conf, truth=tru, pm=pm_dev, pm_true=pm_true, share=share, label10=lab10,
                   pm5=pm5, pm5_lo=pm5lo, pm5_hi=pm5hi, internals=blob), met, acc


def _truth10(labels):
    """The same 10-minute majority rule applied to the true labels."""
    out = []
    for w in range(len(labels)):
        ans = [l for l in labels[max(0, w - 19):w + 1] if l in P.CLASSES]
        if len(ans) < 6:
            out.append(None)
            continue
        vals, cnt = np.unique(ans, return_counts=True)
        k = int(np.argmax(cnt))
        out.append(vals[k] if cnt[k] >= 0.5 * len(ans) else P.MIXED)
    return out


def _accuracy(out, truth_labels, pm_true):
    """Counts for the before/after summary: instant vs 10-min source, raw vs corrected 5-min PM2.5."""
    lab, lab10, valid = out["label"], out["label10"], out["status"] == "ok"
    t10 = _truth10(truth_labels)
    r = dict(n=0, ans=0, ok=0, n10=0, ans10=0, ok10=0, chg=0, chg10=0, chg_t=0, chg_t10=0,
             err_raw=[], err5=[], inband=0, nband=0)
    for w in range(len(lab)):
        if truth_labels[w] in P.CLASSES and valid[w]:
            r["n"] += 1
            if lab[w] in P.CLASSES:
                r["ans"] += 1
                r["ok"] += lab[w] == truth_labels[w]
        if t10[w] in P.CLASSES and valid[w]:
            r["n10"] += 1
            if lab10[w] in P.CLASSES:
                r["ans10"] += 1
                r["ok10"] += lab10[w] == t10[w]
        if w:
            r["chg"] += lab[w] != lab[w - 1]
            r["chg10"] += lab10[w] != lab10[w - 1]
            r["chg_t"] += truth_labels[w] != truth_labels[w - 1]
            r["chg_t10"] += t10[w] != t10[w - 1]
        if valid[w] and pm_true[w] >= 5:
            r["err_raw"].append(float(abs(out["pm_raw"][w] - pm_true[w]) / pm_true[w]))
            lo = max(0, w - 9)
            m = valid[lo:w + 1]
            true5 = pm_true[lo:w + 1][m].mean()
            if np.isfinite(out["pm5"][w]) and true5 >= 5:
                r["err5"].append(float(abs(out["pm5"][w] - true5) / true5))
                r["nband"] += 1
                r["inband"] += out["pm5_lo"][w] <= true5 <= out["pm5_hi"][w]
    r["hours"] = len(lab) * HOP / 3600
    return r


def _summarise(accs):
    keys = ("n", "ans", "ok", "n10", "ans10", "ok10", "chg", "chg10", "chg_t", "chg_t10", "inband", "nband", "hours")
    tot = {k: sum(a[k] for a in accs) for k in keys}
    er = np.concatenate([a["err_raw"] for a in accs]) if accs else np.array([])
    e5 = np.concatenate([a["err5"] for a in accs]) if accs else np.array([])
    f = lambda a, b: float(a / b) if b else float("nan")
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else float("nan")
    return dict(instant_correct=f(tot["ok"], tot["ans"]), instant_answered=f(tot["ans"], tot["n"]),
                ten_correct=f(tot["ok10"], tot["ans10"]), ten_answered=f(tot["ans10"], tot["n10"]),
                changes_per_h=f(tot["chg"], tot["hours"]), changes10_per_h=f(tot["chg10"], tot["hours"]),
                truth_changes_per_h=f(tot["chg_t"], tot["hours"]), truth10_changes_per_h=f(tot["chg_t10"], tot["hours"]),
                pm_raw_med=q(er, 0.5), pm_raw_p90=q(er, 0.9), pm5_med=q(e5, 0.5), pm5_p90=q(e5, 0.9),
                band_coverage=f(tot["inband"], tot["nband"]))


def stm32_block():
    """What the STM32 tab shows: schedule pattern, power states, cycles, memory and power budget (from run_mcu.py)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "mcu_results.pkl")
    if not os.path.exists(path):
        return None
    from stm32_twin import schedule as SC, spec
    m = pickle.load(open(path, "rb"))
    archs = {}
    for key, rec in (("A0", m["archs"][0]), ("A4", m["archs"][-1])):
        n, a = rec["nom"], rec["arch"]
        t0 = 60.0
        segs = [(round(x - t0, 6), round(y - t0, 6), int(k)) for x, y, k in SC.state_segments(a, n["cpu"], n["per"], t0, t0 + 30)]
        ev = []
        for lane, st, en in n["events"]:
            mk = (en > t0) & (st < t0 + 30)
            ev += [(lane, round(float(x - t0), 6), round(float(y - t0), 6)) for x, y in zip(st[mk], en[mk])]
        archs[key] = dict(name=a.name.strip(), tick=a.tick, sd=a.sd, gps=a.gps, isr=a.isr_overhead_cycles,
                          avg_ma=[rec[w]["avg_ma"] for w in ("low", "nom", "high")], run=n["run"], sleep=n["sleep"],
                          stop=n["stop"], segments=segs, events=ev, tasks_us=n["tasks_us"],
                          i_run=n["i_run"], i_sleep=n["i_sleep"], i_stop=n["i_stop"])
    cyc = {v: {w: m["variants"][v]["cycles"][w] for w in ("low", "nom", "high")} for v in ("v0", "v3")}
    mem = m["memory"]
    return dict(archs=archs, cycles=cyc, macs=1392, state_bytes=m["variants"]["v3"]["state_bytes"],
                memory=dict(flash=[(k, v[1]) for k, v in mem["flash"].items()], ram=[(k, v[1]) for k, v in mem["ram"].items()],
                            flash_total=list(mem["flash_total"]), ram_total=list(mem["ram_total"]),
                            flash_cap=mem["flash_cap"], ram_cap=mem["ram_cap"]),
                levers=[(name, b["total_ma"], b["hours"], b["parts"]) for name, b in m["levers"]],
                spec=dict(f_mhz=spec.F_MAX_MHZ, i_run=spec.I_RUN_170_MA, i_sleep=spec.I_SLEEP_170_MA, i_stop=spec.I_STOP_MA),
                features=list(P.FEATURE_NAMES), heater=P.HEATER_PROFILE.tolist(), groups=P.HEATER_GROUPS.tolist())


QM = None


def _init(qm):
    global QM
    QM = qm


def simulate_area(qm, path):
    from concurrent.futures import ProcessPoolExecutor
    import os
    data = dict(classes=P.CLASSES + [P.UNCERTAIN, P.FAULT, P.WARMUP, P.MIXED, "confounder"],
                hop=HOP, window=P.FW_WINDOW_S, conf_thr=qm["conf_thr"],
                devices=DEVICES, sources=[], cases=[])
    for s in MAP_SOURCES:
        e = dict(id=s["id"], label=s["label"], subtype=s["subtype"], cls=P.SUBTYPES[s["subtype"]]["cls"] or "confounder",
                 q=s["q"])
        e.update({k: s[k] for k in ("pt", "line", "regional") if k in s})
        data["sources"].append(e)
    jobs = [(c, k) for c in CASES for k in range(len(DEVICES))]
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1), initializer=_init, initargs=(qm,)) as ex:
        res = list(ex.map(_run_device, jobs, chunksize=2))
    all_acc = []
    for ci, c in enumerate(CASES):
        rows = res[ci * len(DEVICES):(ci + 1) * len(DEVICES)]
        accs = [r[3] for r in rows]
        all_acc += accs
        met = rows[0][2]
        active_ids = [s["id"] for s in MAP_SOURCES if s["id"] in c["active"]]
        act = {sid: a for sid, a in zip(active_ids, met.pop("activity"))}
        for sid, (a, b, factor) in c.get("boost", {}).items():         # show the boost in the plume
            t = np.arange(len(act[sid])) * HOP + P.FW_WINDOW_S - 1
            act[sid] = [round(v * (factor if a <= tt < b else 1), 2) for v, tt in zip(act[sid], t)]
        data["cases"].append(dict(id=c["id"], name=c["name"], start=c["start"], blurb=c["blurb"],
                                  met=c["met"], faults=c.get("faults", {}), weather=met, activity=act,
                                  devices=[r[1] for r in rows], accuracy=_summarise(accs)))
    data["accuracy"] = _summarise(all_acc)
    data["stm32"] = stm32_block()
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return data
