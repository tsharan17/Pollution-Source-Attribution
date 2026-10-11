"""
Scenario library: how training data would really be collected, and the test conditions.

Unit IDs:   0-2  chamber training devices        3-6  extra field-training devices
            7-8  validation devices              100-109  the 10 pilot prototypes (test only)
Seeds:      training < 200000 <= validation < 400000 <= test
"""
import numpy as np

from . import params as P
from .environment import Scenario, Source

TRAIN_SUBTYPES = [k for k, v in P.SUBTYPES.items() if v["train"]]
UNSEEN_SUBTYPES = [k for k, v in P.SUBTYPES.items() if not v["train"] and v["cls"] is not None]
CONFOUNDERS = [k for k, v in P.SUBTYPES.items() if v["cls"] is None]
TEST_UNITS = list(range(100, 110))


def chamber_session(subtype, strength, seed, unit=0, rh=50.0, temp=25.0, age=0.0, dur=1200,
                    bg=15.0, enclosure=P.DEFAULT_ENCLOSURE):
    """Lab protocol: zero/low air, then a single source fills the chamber, then purge."""
    srcs = [] if subtype is None else [Source(subtype, strength, pattern="chamber",
                                              t_on=P.FW_WARMUP_S + 60, t_off=dur - 240)]
    return Scenario(f"chamber_{subtype}", dur, seed=seed, chamber=True, t_mean=temp,
                    rh_mean=rh, bg_pm25=bg, sources=srcs, unit_id=unit, age_days=age,
                    enclosure=enclosure, tags=dict(kind="chamber", subtype=subtype))


def chamber_set(n_per_subtype, seed0, units=(0, 1, 2), subtypes=None):
    rng = np.random.default_rng(seed0)
    out = []
    for st in (subtypes or TRAIN_SUBTYPES) + [None]:
        for k in range(n_per_subtype):
            out.append(chamber_session(
                st, float(np.exp(rng.uniform(np.log(10), np.log(300)))),
                seed=seed0 + len(out), unit=int(rng.choice(units)),
                rh=float(rng.uniform(40, 60)), temp=float(rng.uniform(22, 28)),
                bg=float(rng.uniform(8, 25) if st else rng.uniform(8, 80))))
    return out


def field_session(seed, subtypes=None, n_sources=None, units=(0, 1, 2, 3, 4, 5, 6),
                  age_range=(0, 120), dur=1800, **over):
    """A randomised outdoor session (weather, background, wind, 0-2 local sources)."""
    rng = np.random.default_rng(seed)
    subtypes = subtypes or TRAIN_SUBTYPES
    if n_sources is None:
        n_sources = int(rng.choice([0, 1, 1, 1, 2]))
    srcs = []
    for _ in range(n_sources):
        st = str(rng.choice(subtypes))
        srcs.append(Source(st, float(np.exp(rng.uniform(np.log(10), np.log(300)))),
                           bearing=float(rng.uniform(0, 360)),
                           t_on=float(rng.uniform(0, dur * 0.4)),
                           t_off=float(rng.uniform(dur * 0.7, dur * 1.3))))
    kw = dict(name="field", duration_s=dur, seed=seed,
              start_hour=float(rng.uniform(0, 24)), t_mean=float(rng.uniform(8, 40)),
              t_amp=float(rng.uniform(2, 7)), rh_mean=float(rng.uniform(15, 92)),
              u_mean=float(np.exp(rng.uniform(np.log(0.3), np.log(6)))),
              dir_mean=float(rng.uniform(0, 360)), dir_sd=float(rng.uniform(10, 50)),
              bg_pm25=float(np.exp(rng.uniform(np.log(8), np.log(200)))),
              sources=srcs, unit_id=int(rng.choice(units)),
              age_days=float(rng.uniform(*age_range)),
              tags=dict(kind="field", subtypes=[s.subtype for s in srcs]))
    kw.update(over)
    return Scenario(**kw)


def training_sets():
    """A: chamber-only (what a lab campaign gives).  B: chamber + diverse labelled field data."""
    chamber = chamber_set(40, seed0=1000)
    field = [field_session(50_000 + i) for i in range(900)]
    return chamber, field


def validation_set():
    return ([field_session(200_000 + i, units=(7, 8)) for i in range(250)]
            + chamber_set(6, seed0=210_000, units=(7, 8)))


# ---------------------------------------------------------------------------
# 24-hour "field days" (T11): realistic named environments
# ---------------------------------------------------------------------------
def field_days(unit=100, age=30.0, enclosure=P.DEFAULT_ENCLOSURE, seed0=900_000):
    D = 86400
    days = {
        "Urban roadside, winter (Delhi-like)": dict(
            t_mean=16, t_amp=6, rh_mean=75, u_mean=0.8, dir_mean=300, bg_pm25=140,
            sources=[Source("diesel_traffic", 60, bearing=300), Source("petrol_traffic", 30, bearing=290),
                     Source("two_stroke_traffic", 20, bearing=310),
                     Source("trash_burning", 80, bearing=120, t_on=19 * 3600 - 7 * 3600, t_off=22 * 3600 - 7 * 3600)]),
        "Rural village, crop-burning season": dict(
            t_mean=22, t_amp=7, rh_mean=60, u_mean=1.5, dir_mean=320, bg_pm25=70,
            sources=[Source("biomass_flaming", 150, bearing=320, t_on=4 * 3600, t_off=12 * 3600),
                     Source("biomass_smoldering", 90, bearing=300, t_on=12 * 3600, t_off=20 * 3600),
                     Source("road_dust", 25, bearing=90)]),
        "Construction site, pre-monsoon summer": dict(
            t_mean=36, t_amp=6, rh_mean=30, u_mean=3.0, dir_mean=250, bg_pm25=60,
            sources=[Source("construction_dust", 120, bearing=250, t_on=2 * 3600, t_off=11 * 3600),
                     Source("diesel_traffic", 25, bearing=200),
                     Source("dust_storm", 300, bearing=270, t_on=8 * 3600, t_off=10 * 3600)]),
        "Industrial estate": dict(
            t_mean=28, t_amp=5, rh_mean=55, u_mean=2.0, dir_mean=180, dir_sd=60, bg_pm25=55,
            sources=[Source("coal_combustion", 120, bearing=170), Source("solvent_industry", 60, bearing=220),
                     Source("metal_processing", 80, bearing=140), Source("diesel_traffic", 20, bearing=10)]),
        "Residential park, monsoon": dict(
            t_mean=27, t_amp=3, rh_mean=90, u_mean=2.0, dir_mean=230, bg_pm25=20,
            sources=[Source("cooking", 40, bearing=60, t_on=12 * 3600 - 7 * 3600, t_off=13 * 3600 - 7 * 3600),
                     Source("petrol_traffic", 10, bearing=150)]),
        "Highway toll plaza": dict(
            t_mean=30, t_amp=5, rh_mean=45, u_mean=2.5, dir_mean=90, bg_pm25=50,
            sources=[Source("diesel_traffic", 140, bearing=90), Source("petrol_traffic", 50, bearing=100),
                     Source("road_dust", 40, bearing=80)]),
    }
    out = []
    for k, (name, kw) in enumerate(days.items()):
        out.append(Scenario(name, D, seed=seed0 + k, start_hour=7.0, unit_id=unit, age_days=age,
                            enclosure=enclosure, tags=dict(kind="day"), **kw))
    return out
