"""
STM32 NUCLEO-G474RE twin:  python run_mcu.py [--quick]

  1. operation counts of the real firmware core -> Cortex-M4F cycles, for each computation variant
  2. real-time schedule + power-state simulation for several firmware architectures
  3. accuracy-vs-compute comparison of model families, and a feature-fragility ablation
  4. PMS7003 duty-cycling policies and the resulting battery life
  5. memory budget, peripheral plan, power levers  -> results/mcu_results.pkl / .json

Run run_all.py first (it trains the deployed model). The report is built by build_report.py.
"""
import argparse
import dataclasses
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from polso_twin import scenarios as S                              # noqa: E402
from polso_twin.model import load_qmodel                           # noqa: E402
from stm32_twin import profile as PR, schedule as SC, spec, power, platform_map, pareto, duty   # noqa: E402
from stm32_twin.spec import Arch                                   # noqa: E402

RES = os.path.join(ROOT, "results")

BASE = Arch("A0  Typical starter project", tick="hal1k", isr_overhead_cycles=1800, sd="blocking", gps="default",
            gps_parse_cycles=12000)
ARCHS = [
    BASE,
    dataclasses.replace(BASE, name="A1  + tickless idle (Stop 1)", tick="tickless"),
    dataclasses.replace(BASE, name="A2  + DMA-driven SD, RMC-only GPS, lean parser", tick="tickless", sd="async",
                        gps="rmc", gps_parse_cycles=1500),
    dataclasses.replace(BASE, name="A3  + register-level (LL) drivers", tick="tickless", sd="async", gps="rmc",
                        gps_parse_cycles=1500, isr_overhead_cycles=250),
    dataclasses.replace(BASE, name="A4  + optimised computation (v3)", tick="tickless", sd="async", gps="rmc",
                        gps_parse_cycles=1500, isr_overhead_cycles=250, variant="v3"),
]
A4 = ARCHS[-1]


def weight_bytes():
    d = json.load(open(os.path.join(RES, "qmodel_B.json")))
    return sum(len(np.ravel(L["W"])) + 4 * len(L["b"]) for L in d["layers"]) + 8 * len(d["mu"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = {}

    print("[1/5] operation counts and cycle model")
    out["variants"] = PR.analyse(1800 if a.quick else 3600)

    print("[2/5] schedule and power-state simulation")
    arch_res = []
    for arch in ARCHS:
        v = out["variants"][arch.variant]
        rec = {}
        for which in ("low", "nom", "high"):
            rec[which] = SC.simulate(arch, v["cycles"][which], T=1800 if a.quick else 3600, which=which,
                                     with_events=(which == "nom" and arch in (ARCHS[0], A4)))
        rec["arch"] = arch
        arch_res.append(rec)
        n = rec["nom"]
        print(f"  {arch.name:52s} avg {n['avg_ma']:5.2f} mA (range {rec['low']['avg_ma']:.2f}-{rec['high']['avg_ma']:.2f})  "
              f"run {n['run']:.3%} sleep {n['sleep']:.1%} stop {n['stop']:.1%}")
    out["archs"] = arch_res
    sweep = []
    for f in (170, 120, 85, 48, 24, 16):
        ar = dataclasses.replace(A4, f_mhz=float(f), name=f"A4 @ {f} MHz")
        r = SC.simulate(ar, out["variants"]["v3"]["cycles"]["nom"], T=1800 if a.quick else 3600)
        sweep.append(dict(f=f, avg_ma=r["avg_ma"], run=r["run"], stop=r["stop"], window_us=r["tasks_us"]["window"]))
    out["freq_sweep"] = sweep

    print("[3/5] memory budget and platform plan")
    st_bytes = out["variants"]["v3"]["state_bytes"]
    out["memory"] = platform_map.memory_budget(st_bytes, weight_bytes())
    out["tasks"] = platform_map.task_catalogue(arch_res[-1]["nom"], out["variants"]["v3"]["cycles"]["nom"], A4)

    print("[4/5] model families on the MCU")
    df, abl, meta = pareto.run(quick=a.quick)
    out["pareto"], out["ablation"], out["pareto_meta"] = df, abl, meta

    print("[5/5] PMS duty-cycling policies")
    qm = load_qmodel(os.path.join(RES, "qmodel_B.json"))
    hours = 4 if a.quick else 8
    scans = {}
    for site, dev, grid in (("busy", S.field_days(unit=7, seed0=800_000), (0.30, 0.10)),
                            ("quiet", duty.quiet_days(7, 810_000, hours=hours), (0.18, 0.10, 0.06))):
        for d in dev:
            d.duration_s = hours * 3600
        for thr in grid:
            pol, fl = duty.run(qm, dev, thr, log=lambda *x: None)
            scans[(site, thr)] = dict(policies=pol, flag=fl)
    thr = 0.10
    test = {}
    busy = S.field_days(unit=100, seed0=900_000)
    for d in busy:
        d.duration_s = hours * 3600
    test["busy"] = duty.run(qm, busy, thr, log=print)
    test["quiet"] = duty.run(qm, duty.quiet_days(100, 910_000, hours=hours), thr, log=print)
    out["duty"] = dict(scans=scans, test=test, thr=thr, hours=hours)

    # system power levers
    n0, n4 = arch_res[0]["nom"]["avg_ma"], arch_res[-1]["nom"]["avg_ma"]
    busy_d = test["busy"][0]["periodic 1/3"]["duty"]
    quiet_d = test["quiet"][0]["periodic 1/6"]["duty"]
    levers = [
        ("As in the project deck: NUCLEO board, PMS always on, continuous GPS, starter firmware",
         power.budget(1.0, n0, "continuous NMEA", nucleo=True)),
        ("Custom board (no ST-LINK / Nucleo overhead)", power.budget(1.0, n0, "continuous NMEA")),
        ("+ efficient firmware architecture (A4)", power.budget(1.0, n4, "continuous NMEA")),
        ("+ GPS: one fix every 10 minutes", power.budget(1.0, n4, "one fix every 10 min")),
        ("+ PMS 1/3 duty (busy site)", power.budget(busy_d, n4, "one fix every 10 min")),
        ("+ PMS 1/6 duty (quiet site)", power.budget(quiet_d, n4, "one fix every 10 min")),
    ]
    out["levers"] = levers
    out["meta"] = dict(runtime_min=(time.time() - t0) / 60, quick=a.quick)

    with open(os.path.join(RES, "mcu_results.pkl"), "wb") as f:
        pickle.dump(out, f)
    slim = dict(variants={k: dict(label=v["label"], cycles=v["cycles"], hop_cycles=v["hop_cycles"],
                                  equivalence=v["equivalence"], state_bytes=v["state_bytes"])
                          for k, v in out["variants"].items()})
    with open(os.path.join(RES, "mcu_results.json"), "w") as f:
        json.dump(slim, f, indent=1, default=float)
    print(f"done in {out['meta']['runtime_min']:.1f} min -> results/mcu_results.pkl")


if __name__ == "__main__":
    main()
