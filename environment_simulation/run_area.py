"""Simulate the 16-device array across the 2 km map for all six cases -> results/area_sim.json"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from polso_twin.model import load_qmodel        # noqa: E402
from polso_twin.area import simulate_area       # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    qm = load_qmodel(os.path.join(ROOT, "results", "qmodel_B.json"))
    d = simulate_area(qm, os.path.join(ROOT, "results", "area_sim.json"))
    n = sum(len(c["devices"]) for c in d["cases"])
    print(f"{n} device-runs, {len(d['cases'])} cases in {time.time() - t0:.0f} s -> results/area_sim.json")
