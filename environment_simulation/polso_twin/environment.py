"""
Environment engine: weather, regional background and time-varying local sources.

Produces, at 1 Hz, what the air *at the device* looks like: per-mode particle number
concentrations, gas concentrations, temperature, humidity and wind, plus the ground
truth (dry PM2.5 contributed by each source).
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import lfilter

from . import params as P
from .physics import (ou_process, abs_humidity, rel_humidity, subtype_number_per_pm25,
                      angle_diff)


@dataclass
class Source:
    subtype: str
    strength: float                 # typical PM2.5 excess (ug/m3) when plume reaches device
    bearing: float = 0.0            # direction (deg) the source lies in, seen from device
    t_on: float = 0.0               # s
    t_off: float = 1e12             # s
    pattern: str | None = None      # override SUBTYPES pattern (e.g. "chamber")


@dataclass
class Scenario:
    name: str
    duration_s: int
    seed: int = 0
    start_hour: float = 10.0
    # weather
    t_mean: float = 25.0
    t_amp: float = 5.0
    rh_mean: float = 50.0           # RH at t_mean; absolute humidity held ~constant
    u_mean: float = 1.5             # m/s
    u_gust: float = 0.4             # lognormal gustiness
    dir_mean: float = 0.0           # direction wind comes FROM (deg)
    dir_sd: float = 30.0
    # regional background
    bg_pm25: float = 40.0
    bg_diurnal: bool = True
    sources: list = field(default_factory=list)
    chamber: bool = False           # no wind, no direction effects, steady T/RH
    # device
    unit_id: int = 0
    age_days: float = 0.0
    enclosure: str = P.DEFAULT_ENCLOSURE
    faults: dict = field(default_factory=dict)
    emission_var: float = 0.3       # scenario-to-scenario emission-ratio spread
    sensor_seed: int | None = None  # separate sensor-noise seed (several devices sharing one environment)
    tags: dict = field(default_factory=dict)


def _pattern(kind, n, t, hours, u, rng):
    """Dimensionless activity multiplier (mean ~1 while active)."""
    if kind == "traffic":
        rush = 0.6 + 0.5 * np.exp(-0.5 * ((hours - 9.0) / 1.5) ** 2) \
                   + 0.6 * np.exp(-0.5 * ((hours - 19.0) / 1.8) ** 2)
        rush = np.where((hours % 24) < 5.5, 0.25, rush)
        rate = 0.12 * rush                          # vehicles per second (plume-relevant)
        events = rng.random(n) < rate
        amp = np.where(events, rng.lognormal(0.0, 0.9, n), 0.0)
        a = np.exp(-1.0 / 12.0)                     # passing-vehicle plume decays in ~12 s
        spikes = 0.5 * lfilter([1.0], [1.0, -a], amp)
        f = 0.35 * rush + spikes
        return f / max(np.mean(f), 1e-9)
    if kind == "burning":
        state, f = 1, np.empty(n)
        i = 0
        while i < n:
            dur = int(rng.exponential(700 if state else 300)) + 30
            f[i:i + dur] = state
            state ^= 1
            i += dur
        f = lfilter([1 - np.exp(-1 / 30.0)], [1, -np.exp(-1 / 30.0)], f)
        f = f * np.exp(ou_process(n, 60, 0.4, rng))
        return f / max(np.mean(f), 1e-9)
    if kind == "resuspension":
        wind = 0.3 + (np.clip(u, 0, None) / 4.0) ** 3
        traffic = _pattern("traffic", n, t, hours, u, rng)
        f = (0.6 * wind + 0.4 * traffic) * np.exp(ou_process(n, 90, 0.4, rng))
        return f / max(np.mean(f), 1e-9)
    if kind == "steady":
        return np.exp(ou_process(n, 90, 0.3, rng) - 0.045)
    if kind == "plume":
        return np.exp(ou_process(n, 120, 0.5, rng) - 0.125)
    if kind == "storm":
        ramp = 1.0 / (1.0 + np.exp(-(t - t[0] - 900.0) / 200.0))
        return ramp * np.exp(ou_process(n, 120, 0.3, rng))
    if kind == "chamber":
        return np.exp(ou_process(n, 60, 0.08, rng))
    raise ValueError(kind)


def generate(sc: Scenario):
    """Generate the ambient environment for a scenario. Returns a dict of arrays."""
    rng = np.random.default_rng(sc.seed)
    n = int(sc.duration_s)
    t = np.arange(n, dtype=float)
    hours = sc.start_hour + t / 3600.0
    hod = hours % 24

    # ---- weather ----
    if sc.chamber:
        temp = sc.t_mean + ou_process(n, 900, 0.15, rng)
        ah0 = abs_humidity(sc.t_mean, sc.rh_mean)
        ah = ah0 * (1 + ou_process(n, 900, 0.01, rng))
        u = np.zeros(n)
        wdir = np.zeros(n)
    else:
        temp = sc.t_mean + sc.t_amp * np.cos(2 * np.pi * (hod - 15.0) / 24.0) \
               + ou_process(n, 600, 0.5, rng)
        ah0 = abs_humidity(sc.t_mean, sc.rh_mean)
        ah = ah0 * (1 + ou_process(n, 3600, 0.05, rng))
        u = sc.u_mean * np.exp(ou_process(n, 60, sc.u_gust, rng) - 0.5 * sc.u_gust ** 2)
        wdir = (sc.dir_mean + ou_process(n, 300, sc.dir_sd, rng)) % 360
    rh = np.clip(rel_humidity(temp, ah), 1.0, 99.5)
    ah = abs_humidity(temp, rh)                 # re-derive after the RH clip (fog)

    # ---- regional background ----
    if sc.chamber:
        bg = sc.bg_pm25 * np.exp(ou_process(n, 1800, 0.05, rng))
    else:
        diurnal = 1.0 + (0.35 * np.cos(2 * np.pi * (hod - 6.0) / 24.0) if sc.bg_diurnal else 0.0)
        bg = sc.bg_pm25 * diurnal * np.exp(ou_process(n, 1800, 0.10, rng))

    comps = [("background", "background", P.BACKGROUND_MODES, bg)]
    activity = []
    gases = {g: P.BACKGROUND_GAS[g][0] + P.BACKGROUND_GAS[g][1] * bg for g in P.GASES}

    # ---- local sources ----
    for k, s in enumerate(sc.sources):
        spec = P.SUBTYPES[s.subtype]
        kind = s.pattern or spec["pattern"]
        f = _pattern(kind, n, t, hours, u, rng)
        if sc.chamber:
            geo = np.ones(n)
        else:
            dtheta = angle_diff(wdir, s.bearing)
            gauss = np.exp(-0.5 * (dtheta / 35.0) ** 2) + 0.05
            calm = np.clip(u / 1.0, 0, 1)
            direc = calm * gauss + (1 - calm) * 0.5
            dil = np.clip((2.0 / np.maximum(u, 0.3)) ** 0.8, 0, 4.0)
            if kind in ("resuspension", "storm"):
                dil = dil ** 0.3
            geo = direc * dil
        # soft on/off edges (chamber: 40 s fill, 120 s purge)
        tau_on, tau_off = (40.0, 120.0) if sc.chamber else (20.0, 60.0)
        on = np.where(t >= s.t_on, 1 - np.exp(-np.clip(t - s.t_on, 0, None) / tau_on), 0.0)
        off = np.exp(-np.clip(t - s.t_off, 0, None) / tau_off)
        pm25 = s.strength * f * geo * on * off
        activity.append(f * on * off)
        # per-scenario emission variability: mode sizes and gas ratios
        modes = [(m[0] * rng.lognormal(0, 0.10), m[1], m[2] * rng.lognormal(0, 0.15),
                  m[3], m[4], m[5]) for m in spec["modes"]]
        comps.append((s.subtype, spec["cls"], modes, pm25))
        for g in P.GASES:
            ratio = spec["gas"][g] * rng.lognormal(0, sc.emission_var)
            if ratio > 0:
                gases[g] = gases[g] + ratio * pm25 * np.exp(ou_process(n, 120, 0.15, rng))

    # per-mode number concentrations (#/cm3)
    mode_list, mode_n = [], []
    for name, cls, modes, pm25 in comps:
        npp = subtype_number_per_pm25(modes)
        for m, k in zip(modes, npp):
            mode_list.append(m)
            mode_n.append(pm25 * k)

    return dict(
        t=t, hours=hours, temp=temp, rh=rh, ah=ah, u=u, wdir=wdir,
        pressure=1005.0 + ou_process(n, 3600, 1.0, rng),
        comp_names=[c[0] for c in comps], comp_cls=[c[1] for c in comps],
        comp_pm25=np.stack([c[3] for c in comps]),
        modes=mode_list, mode_n=np.stack(mode_n),
        gases=np.stack([gases[g] for g in P.GASES], axis=1),
        src_activity=np.stack(activity) if activity else np.zeros((0, n)),
    )
