"""
Rehearse the verification campaign on simulated data:  python run_verification.py [--quick]

  3 co-located prototypes next to FEM-class PM2.5 / gas references, tracer-based daily apportionment
  and a 10-minute observer log, N days -> acceptance table, sample-size curves and file-format templates
  (results/verification/*.csv).
"""
import argparse
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from polso_twin.model import load_qmodel            # noqa: E402
from verify import rehearsal                          # noqa: E402

RES = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="6 days x 4 h (fast smoke test)")
    a = ap.parse_args()
    t0 = time.time()
    qm = load_qmodel(os.path.join(RES, "qmodel_B.json"))
    n_days, hours = (6, 4) if a.quick else (12, 8)
    df, truth_days = rehearsal.run(qm, n_days=n_days, hours=hours)
    R = rehearsal.analyse(df, truth_days, n_train_days=max(2, n_days // 3),
                          out_dir=os.path.join(RES, "verification"))
    R["win"] = df[["pm25", "total", "truth"]].copy()
    R["runtime_min"] = (time.time() - t0) / 60
    R["hours"] = hours
    with open(os.path.join(RES, "verification_results.pkl"), "wb") as f:
        pickle.dump(R, f)
    print(f"done in {R['runtime_min']:.1f} min -> results/verification_results.pkl")
    for c in R["criteria"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['value']}")


if __name__ == "__main__":
    main()
