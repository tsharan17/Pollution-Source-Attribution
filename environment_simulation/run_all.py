"""
Run the whole PolSo virtual test campaign:  python run_all.py [--quick] [--report-only]

  1. build training data (chamber protocol + diverse field sessions) and train the models
  2. quantise the deployed model to int8 and generate firmware_c/model_params.h
  3. run tests T00-T15 on held-out scenarios and the 10 unseen pilot units
  4. write results/results.pkl, results/summary.json and results/report.html
"""
import argparse
import datetime
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from polso_twin import params as P, campaign as C, report as REP   # noqa: E402
from findings import findings_and_recommendations                   # noqa: E402

RES = os.path.join(ROOT, "results")


def assumptions():
    return [
        ("PMS7003 detection efficiency", f"50 % at {P.PMS_CE_LOW_D50} um, falls to 50 % again at {P.PMS_CE_HIGH_D50} um",
         "Low-cost OPC evaluations; measure vs a reference OPC/APS in a chamber"),
        ("PMS7003 sizing blur / counting volume", f"sigma {P.PMS_SIZING_SIGMA} (ln), {P.PMS_SAMPLE_CM3} cm3/s",
         "Fit from 1 Hz count noise in clean air"),
        ("PMS7003 unit gain spread", f"{P.PMS_UNIT_GAIN_SIGMA:.0%} (lognormal)", "Co-locate the 10 units for 48 h"),
        ("Hygroscopic growth (kappa)", "soot 0.05, smoke 0.15-0.2, sulfate-rich 0.35-0.45, dust 0.03",
         "Petters & Kreidenweis 2007; check PM vs RH against a dried reference"),
        ("BME688 gas sensitivities", "Gaussian in heater T; CO@340 C, NO2@190 C (oxidising), oVOC@270 C, aromatics@360 C",
         "Bosch BME AI-Studio sessions with known gases"),
        ("BME688 humidity coefficient", "0.035 -> 0.027 ln(R) per g/m3 (150 -> 400 C)", "Humidity steps in clean air"),
        ("BME688 unit spread", f"R0 {P.UNIT_MOX_RREF_SIGMA}, sensitivity {P.UNIT_MOX_SENS_SIGMA} (lognormal sigma)", "Per-unit zero-air calibration"),
        ("BME688 drift", f"{P.MOX_DRIFT_PER_DAY} ln/day + random walk {P.MOX_RW_PER_SQRT_DAY}/sqrt(day)", "Long-term co-location"),
        ("Emission ratios", "e.g. diesel CO 25 / NO2 2.5 ppb per ug/m3; biomass CO 6-10, oVOC 0.8-1.5", "Akagi 2011; roadside studies"),
        ("Inlet losses", "PolSo-19 Run A (hooded coarse loss 4.4 % mean at 4 m/s)", "Wind-tunnel / fan test of the printed hood"),
        ("Power draw", "PMS 80 mA@5 V, BME 9 mA, MCU 6-30 mA, NUCLEO overhead 45 mA, GPS 45 mA",
         "Measure each rail with a USB power meter / INA219"),
        ("Firmware", f"{P.FW_WINDOW_S} s window, {P.FW_HOP_S} s hop, {P.FW_WARMUP_S} s warm-up, EMA {P.FW_PROB_EMA}", "Design choice"),
    ]


def make_figures(R, extra):
    figs = dict(fingerprints=REP.fig_fingerprints(R["T01"]["data"]["df"]),
                confusions=REP.fig_confusions(R["T00"]["data"]["df"], extra["dfA"]),
                robustness=REP.fig_robustness(R), detection=REP.fig_detection(R), bars=REP.fig_bars(R))
    fd = R["T11"]["data"]["df"]
    for k, name in enumerate(fd["scenario"].unique()):
        figs[f"day_{k}"] = REP.fig_day(fd, name)
    return figs


def strip(R):
    """Copy of results without dataframes, for JSON."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("df", "table") or not hasattr(v, "shape")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if hasattr(o, "to_dict"):
            return clean(o.to_dict())
        return o
    return clean(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller scenario counts (~5x faster)")
    ap.add_argument("--report-only", action="store_true", help="re-render report from results.pkl")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    pkl = os.path.join(RES, "results.pkl")
    if a.report_only:
        with open(pkl, "rb") as f:
            R, extra, meta = pickle.load(f)
    else:
        t0 = time.time()
        print("[1/3] training")
        T = C.train_all(RES, quick=a.quick)
        print("[2/3] tests")
        R = {}
        steps = [
            ("T00", lambda: C.test_generalisation(T, a.quick)),
            ("T01", lambda: C.test_chamber(T, a.quick)), ("T02", lambda: C.test_detection_limit(T, a.quick)),
            ("T03", lambda: C.test_humidity(T, a.quick)), ("T04", lambda: C.test_temperature(T, a.quick)),
            ("T05", lambda: C.test_wind_enclosure(T, a.quick)), ("T06", lambda: C.test_mixtures(T, a.quick)),
            ("T07", lambda: C.test_latency(T, a.quick)), ("T08", lambda: C.test_drift(T, a.quick)),
            ("T09", lambda: C.test_units(T, a.quick)), ("T10", lambda: C.test_unseen(T, a.quick)),
            ("T11", lambda: C.test_field_days(T, a.quick)), ("T12", lambda: C.test_faults(T, a.quick)),
            ("T13", lambda: C.test_sil(T, ROOT, a.quick)),
        ]
        extra = {}
        for tid, fn in steps:
            ts = time.time()
            out = fn()
            if tid == "T00":
                (res,), gen_df = out
                dA = gen_df.copy()
                from polso_twin.firmware import qmodel_forward
                dA["pred"] = C.relabel(gen_df, qmodel_forward(T["qa"], np.stack(gen_df["feat"]))[0], T["qa"]["conf_thr"])
                extra["dfA"] = dA
                out = res
            R[tid] = out
            print(f"  {tid} {out['verdict']:4s} {out['name']}  ({time.time() - ts:.0f} s)\n      {out['summary']}")
        R["T14"] = C.test_power(T, R["T11"]["data"]["df"])
        R["T15"] = C.test_fusion(T, R["T00"]["data"]["df"], R["T11"]["data"]["df"])
        for tid in ("T14", "T15"):
            print(f"  {tid} {R[tid]['verdict']:4s} {R[tid]['name']}\n      {R[tid]['summary']}")
        sim_s = sum(len(r["data"]["df"]) * P.FW_HOP_S for r in R.values() if "df" in r["data"])
        meta = dict(date=datetime.date.today().isoformat(), runtime=f"{(time.time() - t0) / 60:.0f} min",
                    sim_hours=f"{sim_s / 3600 + 1200:,.0f}", conf_thr=T["qm"]["conf_thr"],
                    weights_bytes=T["weights_bytes"], n_train=T["n_train"], quick=a.quick)
        with open(pkl, "wb") as f:
            pickle.dump((R, extra, meta), f)
    print("[3/3] report")
    figs = make_figures(R, extra)
    notes = findings_and_recommendations(R, meta)
    with open(os.path.join(RES, "report.html"), "w", encoding="utf-8") as f:
        f.write(REP.build_html(R, figs, notes, assumptions(), meta))
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(dict(meta=meta, results=strip(R)), f, indent=1, default=str)
    n_pass = sum(r["verdict"] == "PASS" for r in R.values())
    print(f"done: {n_pass}/{len(R)} PASS -> results/report.html")


if __name__ == "__main__":
    main()
