"""
File formats for the verification campaign.

DEVICE LOG v1 (one row per 30 s window, written to the microSD by the firmware):
    ts_utc, unit, fw_version, model_version, status, label, conf, p_<class> x5,
    pm1_cf1, pm25_cf1, pm10_cf1, temp_c, rh_pct, f00..f24 (the 25 model features)
RAW LOG (1 Hz, strongly recommended, one file per day): the 6 cumulative PMS7003 counts, PM1/2.5/10 (CF=1),
    BME688 gas resistance with its heater-step index, temperature, humidity, pressure, GPS fix, battery voltage.
    Logging raw data means labels can be recomputed offline with a better model without repeating the campaign.
REFERENCE FILES (one per instrument; UTC timestamps; local time zone conversions are the usual source of error):
    reference_hourly.csv      ts_utc, pm25_fem, co_ppb, no2_ppb, so2_ppb  (+ any other analyser)
    reference_daily.csv       date, <class>_share x5   (filter chemistry + PMF / CMB apportionment)
    observer_log.csv          start_utc, end_utc, dominant_source, confidence, notes   (10-min resolution)
"""
import numpy as np
import pandas as pd

from polso_twin import params as P

EPOCH = pd.Timestamp("2026-11-02 07:00:00", tz="UTC")
LOG_COLUMNS = (["ts_utc", "unit", "fw_version", "model_version", "status", "label", "conf"]
               + [f"p_{c}" for c in P.CLASSES] + ["pm1_cf1", "pm25_cf1", "pm10_cf1", "temp_c", "rh_pct"]
               + [f"f{i:02d}" for i in range(P.N_FEATURES)])


def to_log_frame(rows):
    """rows: list of dicts from the rehearsal (one per window) -> DEVICE LOG v1 DataFrame."""
    out = []
    for r in rows:
        d = dict(ts_utc=(EPOCH + pd.Timedelta(seconds=int(r["t_end"]) + int(r["day"]) * 86400)).isoformat(),
                 unit=r["unit"], fw_version="1.0-sim", model_version="B-int8", status=r["status"], label=r["pred"],
                 conf=round(float(r["conf"]), 3) if np.isfinite(r["conf"]) else "")
        for c, p in zip(P.CLASSES, r["probs"]):
            d[f"p_{c}"] = round(float(p), 4)
        d.update(pm1_cf1=round(float(r["pm1"]), 1), pm25_cf1=round(float(r["pm25"]), 1), pm10_cf1=round(float(r["pm10"]), 1),
                 temp_c=round(float(r["temp_dev"]), 2), rh_pct=round(float(r["rh_dev"]), 1))
        for i, f in enumerate(r["feats"]):
            d[f"f{i:02d}"] = round(float(f), 5)
        out.append(d)
    return pd.DataFrame(out, columns=LOG_COLUMNS)
