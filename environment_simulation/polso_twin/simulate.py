"""Scenario -> environment -> sensors -> transport (with fault injection) -> raw device streams."""
import numpy as np

from . import params as P
from .environment import Scenario, generate
from .sensors import Unit, simulate_pms, simulate_bme, pms_frames_matrix

CONFOUNDER = "confounder"
_UNITS = {}


def get_unit(uid):
    if uid not in _UNITS:
        _UNITS[uid] = Unit(uid)
    return _UNITS[uid]


def run_scenario(sc: Scenario):
    """Simulate one scenario. Returns (streams for the firmware, truth for scoring)."""
    env = generate(sc)
    unit = get_unit(sc.unit_id)
    rng = np.random.default_rng((sc.seed if sc.sensor_seed is None else sc.sensor_seed) + 7919)
    n = len(env["t"])
    f = sc.faults

    cum, cf1, atm = simulate_pms(env, unit, sc.age_days, sc.enclosure, rng)
    if "pms_fan_stall_after" in f:                      # fan slows: flow and counts collapse
        k = env["t"] >= f["pms_fan_stall_after"]
        cum[k] = (cum[k] * 0.05).astype(np.uint16)
        cf1[k] = (cf1[k] * 0.05).astype(np.uint16)
        atm[k] = (atm[k] * 0.05).astype(np.uint16)
    bme = simulate_bme(env, unit, sc.age_days, sc.enclosure, rng)

    # ---- transport layer ----
    frames = pms_frames_matrix(cf1, atm, cum)
    present = np.ones(n, bool)
    if "pms_frame_drop" in f:
        present &= rng.random(n) >= f["pms_frame_drop"]
    if "pms_dead_after" in f:
        present &= env["t"] < f["pms_dead_after"]
    streams = dict(uart=frames, uart_present=present)
    if f.get("pms_byte_error", 0) > 0 or f.get("pms_byte_slip", 0) > 0:
        # byte-level corruption: hand the firmware a raw byte stream per second
        chunks = []
        for i in range(n):
            if not present[i]:
                chunks.append(b"")
                continue
            b = bytearray(frames[i].tobytes())
            flips = rng.random(32) < f.get("pms_byte_error", 0)
            for j in np.nonzero(flips)[0]:
                b[j] ^= 1 << int(rng.integers(0, 8))
            if rng.random() < f.get("pms_byte_slip", 0):        # lost / duplicated byte
                j = int(rng.integers(0, 32))
                b = b[:j] + b[j + 1:] if rng.random() < 0.5 else b[:j] + b[j:j + 1] + b[j:]
            chunks.append(bytes(b))
        streams["uart_bytes"] = chunks

    gas_ok = np.ones(n, bool)
    if "bme_i2c_error" in f:
        gas_ok &= rng.random(n) >= f["bme_i2c_error"]
    if "bme_dead_after" in f:
        gas_ok &= env["t"] < f["bme_dead_after"]
    streams.update(gas_r=bme["gas_r"], gas_step=bme["gas_step"], gas_ok=gas_ok,
                   t_meas=bme["t_meas"], rh_meas=bme["rh_meas"], p_meas=bme["p_meas"],
                   gps_fix=rng.random(n) >= f.get("gps_outage", 0.02),
                   sd_fail=f.get("sd_fail", 0.0), b0=unit.factory_baseline(), seed=sc.seed,
                   # baseline a continuously running tracker would hold today (converged, small error)
                   b_tracked=unit.factory_baseline() + unit.mox_drift(sc.age_days) + rng.normal(0, 0.03, len(P.HEATER_PROFILE)))

    truth = dict(t=env["t"], comp_pm25=env["comp_pm25"], comp_cls=env["comp_cls"],
                 comp_names=env["comp_names"], temp=env["temp"], rh=env["rh"], u=env["u"],
                 wdir=env["wdir"], gases=env["gases"], pms_cum=cum, pms_cf1=cf1,
                 src_activity=env["src_activity"],
                 scenario=sc)
    return streams, truth


def window_truth(truth, t_end, window=P.FW_WINDOW_S):
    """Ground-truth label for the window ending at t_end (inclusive)."""
    a, b = max(0, t_end - window + 1), t_end + 1
    pm = truth["comp_pm25"][:, a:b].mean(1)
    total = pm.sum() + 1e-9
    by_cls = {}
    for c, v in zip(truth["comp_cls"], pm):
        key = CONFOUNDER if c is None else c
        by_cls[key] = by_cls.get(key, 0.0) + v
    local = {k: v for k, v in by_cls.items() if k != "background"}
    shares = {k: v / total for k, v in by_cls.items()}
    if local:
        best = max(local, key=local.get)
        if local[best] / total >= 0.4:
            return best, shares, total
        if sum(local.values()) / total >= 0.25:
            return P.MIXED, shares, total
    return "background", shares, total
