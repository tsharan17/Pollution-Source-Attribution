"""
Compare cycle counts measured on the NUCLEO-G474RE (firmware_c/bench/dwt_bench.c output) with the twin's
low / nominal / high predictions.

    python -m stm32_twin.compare_measured measured.txt

Each line of measured.txt:  BENCH <name> <min> <mean> <max>
A measurement INSIDE [low, high] confirms the model; OUTSIDE tells you which cost to re-fit
(printed as a suggested replacement for the corresponding spec.COST entry or stage cost).
"""
import json
import os
import sys

from . import spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# which twin quantity each benchmark corresponds to
MAP = {
    "logf": ("cost", "log"), "log10f": ("cost", "log10"), "expf": ("cost", "exp"),
    "sqrtf": ("cost", "sqrt"), "floorf": ("cost", "floor"),
    "pms_frame_parse": ("stage", "uart_frame"),
    "fw_tick_sample": ("stage", "sample"),
    "mlp_forward": ("stage", "mlp"),
}


def parse(text):
    out = {}
    for ln in text.splitlines():
        p = ln.split()
        if len(p) == 5 and p[0] == "BENCH":
            out[p[1]] = tuple(int(x) for x in p[2:])
    return out


def predictions(variant="v0"):
    p = os.path.join(ROOT, "results", "mcu_results.json")
    if not os.path.exists(p):
        return None
    res = json.load(open(p))
    cyc = res["variants"][variant]["cycles"]
    out = {}
    for name, (kind, key) in MAP.items():
        if kind == "cost":
            out[name] = tuple(spec.COST[key][i] for i in range(3))
        else:
            out[name] = tuple(cyc[w][key] for w in ("low", "nom", "high"))
    w = [cyc[k]["sample"] + cyc[k]["wcheck"] + cyc[k]["features"] + cyc[k]["mlp"] + cyc[k]["post"]
         for k in ("low", "nom", "high")]
    out["fw_tick_window"] = tuple(w)
    return out


def compare(measured, pred):
    rows = []
    for name, m in measured.items():
        if name not in pred:
            continue
        lo, nom, hi = pred[name]
        mean = m[1]
        status = "inside range" if lo * 0.9 <= mean <= hi * 1.1 else ("below range" if mean < lo else "above range")
        rows.append(dict(name=name, measured=mean, low=lo, nominal=nom, high=hi, ratio=mean / nom, status=status))
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    meas = parse(open(sys.argv[1]).read())
    pred = predictions()
    if pred is None:
        sys.exit("run python run_mcu.py first (needs results/mcu_results.json)")
    print(f"{'item':18s} {'measured':>10s} {'low':>9s} {'nominal':>9s} {'high':>9s} {'x nominal':>9s}  status")
    for r in compare(meas, pred):
        print(f"{r['name']:18s} {r['measured']:>10,.0f} {r['low']:>9,.0f} {r['nominal']:>9,.0f} {r['high']:>9,.0f} "
              f"{r['ratio']:>9.2f}  {r['status']}")
