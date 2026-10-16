"""
Execution-cost model of the firmware on the STM32G474 Cortex-M4F.

1. Build the firmware core, unchanged, with the operation counter (firmware_c/prof) for each
   computation variant and run it on simulated sensor streams -> exact FP / libm operation counts
   per processing stage.
2. Convert counts to cycles with the Cortex-M4F cost table (spec.COST), low / nominal / high.
3. Check every variant still produces the same labels as the reference firmware (the plain C build
   is run on the same streams and compared window by window).
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

from . import spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW = os.path.join(ROOT, "firmware_c")
BUILD = os.path.join(ROOT, "results", "mcu_build")

OPT = {"log": "-DFW_OPT_LOGCACHE", "ah": "-DFW_OPT_AH_DECIMATE", "fast": "-DFW_OPT_FASTMATH"}
VARIANTS = {
    "v0": dict(label="Reference firmware", flags=[], mlp="scalar C"),
    "v1": dict(label="+ log10 cached per second", flags=[OPT["log"]], mlp="scalar C"),
    "v2": dict(label="+ humidity term every 10 s", flags=[OPT["log"], OPT["ah"]], mlp="scalar C"),
    "v3": dict(label="+ polynomial log / exp", flags=[OPT["log"], OPT["ah"], OPT["fast"]], mlp="scalar C"),
    "v4": dict(label="+ SIMD int8 MLP (estimated)", flags=[OPT["log"], OPT["ah"], OPT["fast"]],
               mlp="SMLAD / CMSIS-NN"),
}
MACS = 25 * 32 + 32 * 16 + 16 * 5      # int8 MLP 25-32-16-5
HIDDEN_NEURONS = 32 + 16


def _run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr[:2000]}")
    return r


def build_all():
    os.makedirs(BUILD, exist_ok=True)
    for key, v in VARIANTS.items():
        if key == "v4":
            continue
        flags = v["flags"]
        _run(["g++", "-O2", "-std=gnu++14", "-fpermissive", "-Wno-class-memaccess", *flags, "-o",
              os.path.join(BUILD, f"prof_{key}.exe"), os.path.join(FW, "prof", "prof_build.cpp")])
        _run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", *flags, "-o", os.path.join(BUILD, f"sil_{key}.exe"),
              "sil_main.c", "polso_fw.c", "pms7003.c", "-lm"], cwd=FW)


def make_streams(n_seconds=3600):
    """Three realistic streams (road, smouldering biomass, faulty coal plume) as PSIL files."""
    from polso_twin.environment import Scenario, Source
    from polso_twin.simulate import run_scenario
    from polso_twin import campaign as C
    scs = [
        Scenario("road", n_seconds, seed=930_001, start_hour=8, rh_mean=65, u_mean=1.2, bg_pm25=70, unit_id=102,
                 sources=[Source("diesel_traffic", 70), Source("road_dust", 30, bearing=40)]),
        Scenario("burn", n_seconds, seed=930_002, rh_mean=45, u_mean=2.0, bg_pm25=40, unit_id=103, age_days=90,
                 sources=[Source("biomass_smoldering", 120)]),
        Scenario("faulty", n_seconds, seed=930_003, rh_mean=60, bg_pm25=50, unit_id=104,
                 sources=[Source("coal_combustion", 90)],
                 faults=dict(pms_byte_error=5e-3, pms_byte_slip=0.02, bme_i2c_error=0.1)),
    ]
    paths = []
    for sc in scs:
        streams, _ = run_scenario(sc)
        if "uart_bytes" not in streams:
            streams["uart_bytes"] = [streams["uart"][i].tobytes() if streams["uart_present"][i] else b""
                                     for i in range(len(streams["gas_r"]))]
        p = os.path.join(BUILD, f"stream_{sc.name}.bin")
        C._write_sil_input(streams, p)
        paths.append((sc.name, p))
    return paths


def run_prof(key, streams):
    """Operation counts for one variant, summed over the clean streams."""
    tot, meta = None, dict(n_ticks=0, n_windows=0, n_feature_windows=0, frames_ok=0, state_bytes=0)
    for name, path in streams:
        if name == "faulty":
            continue
        csv = os.path.join(BUILD, f"prof_{key}_{name}.csv")
        r = _run([os.path.join(BUILD, f"prof_{key}.exe"), path, csv])
        d = json.loads(r.stdout)
        for k in ("n_ticks", "n_windows", "n_feature_windows", "frames_ok"):
            meta[k] += d[k]
        meta["state_bytes"] = d["state_bytes"]
        if tot is None:
            tot = {ph: dict.fromkeys(ops, 0) for ph, ops in d["ops"].items()}
        for ph, ops in d["ops"].items():
            for k, v in ops.items():
                tot[ph][k] += v
    return tot, meta


def per_call(tot, meta):
    n = dict(sample=meta["n_ticks"], wcheck=meta["n_windows"], features=meta["n_feature_windows"],
             mlp=meta["n_feature_windows"], post=meta["n_feature_windows"], uart=meta["frames_ok"])
    return {ph: {k: v / n[ph] for k, v in ops.items()} for ph, ops in tot.items()}


def stage_cycles(counts_pc, mlp_impl="scalar C", which="nom"):
    """Cycles per call of each stage. which: low | nom | high (applied to every cost)."""
    i = {"low": 0, "nom": 1, "high": 2}[which]
    art = spec.ART_INFLATION[i]
    out = {}
    for ph in ("sample", "wcheck", "features", "mlp", "post"):
        ops = counts_pc[ph]
        c = sum(n * spec.COST[k][i] for k, n in ops.items() if k in spec.COST)
        c += spec.FP_OVERHEAD[i] * sum(ops[k] for k in spec.FP_ARITH)
        out[ph] = c
    out["sample"] += spec.INT_WORK["sample_bookkeeping"][i]
    out["features"] += spec.INT_WORK["window_integer"][i]
    out["post"] += spec.INT_WORK["post"][i]
    out["mlp"] += MACS * spec.MLP_CYCLES_PER_MAC[mlp_impl][i] + HIDDEN_NEURONS * spec.MLP_REQUANT_PER_NEURON[i]
    out["uart_frame"] = spec.INT_WORK["uart_frame_parse"][i]
    return {k: v * art for k, v in out.items()}


def hop_cycles(cyc, hop_s=30):
    """CPU cycles of one 30 s hop for the application code (without interrupt entry overhead)."""
    return hop_s * (cyc["sample"] + cyc["uart_frame"]) + cyc["wcheck"] + cyc["features"] + cyc["mlp"] + cyc["post"]


def equivalence(streams):
    """Compare each variant's output to the reference build on all three streams."""
    rows = {}
    for key in VARIANTS:
        if key == "v4":
            continue
        agree, n, dp, dfeat, hard, soft = 0, 0, 0.0, 0.0, 0, 0
        for name, path in streams:
            for k in ("v0", key):
                _run([os.path.join(BUILD, f"sil_{k}.exe"), path, os.path.join(BUILD, f"sil_{k}_{name}.csv")])
            a = pd.read_csv(os.path.join(BUILD, f"sil_v0_{name}.csv"))
            b = pd.read_csv(os.path.join(BUILD, f"sil_{key}_{name}.csv"))
            ok = ~a.label.isin([6, 7])
            agree += int((a.label == b.label).sum())
            n += len(a)
            diff = a.label != b.label
            both_answer = a.label.between(0, 4) & b.label.between(0, 4)
            hard += int((diff & both_answer).sum())          # two different source answers
            soft += int((diff & ~both_answer).sum())         # answer <-> uncertain flip at the confidence gate
            pc = [f"p{j}" for j in range(5)]
            fc = [f"f{j}" for j in range(25)]
            if ok.any():
                dp = max(dp, float(np.abs(a.loc[ok, pc].values - b.loc[ok, pc].values).max()))
                den = np.abs(a.loc[ok, fc].values) + 1e-2
                dfeat = max(dfeat, float((np.abs(a.loc[ok, fc].values - b.loc[ok, fc].values) / den).max()))
        rows[key] = dict(agree=agree / n, windows=n, hard_disagree=hard, gate_flips=soft, max_dprob=dp,
                         max_rel_dfeat=dfeat)
    rows["v4"] = dict(rows["v3"], note="integer dot products are bit-identical to scalar C")
    return rows


def analyse(n_seconds=3600, log=print):
    build_all()
    streams = make_streams(n_seconds)
    result = {}
    for key, v in VARIANTS.items():
        src = "v3" if key == "v4" else key
        tot, meta = run_prof(src, streams)
        pc = per_call(tot, meta)
        cyc = {w: stage_cycles(pc, v["mlp"], w) for w in ("low", "nom", "high")}
        result[key] = dict(label=v["label"], mlp=v["mlp"], counts_per_call=pc, cycles=cyc,
                           hop_cycles={w: hop_cycles(cyc[w]) for w in cyc}, state_bytes=meta["state_bytes"],
                           meta=meta)
        log(f"  {key} {v['label']:34s} hop cycles nominal {result[key]['hop_cycles']['nom']:>9,.0f} "
            f"(low {result[key]['hop_cycles']['low']:,.0f}, high {result[key]['hop_cycles']['high']:,.0f})")
    eq = equivalence(streams)
    for k in result:
        result[k]["equivalence"] = eq[k]
    return result
