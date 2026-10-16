"""
Simulated reference instruments, each with its own error, built from the twin's ground truth.

The point is to rehearse the analysis honestly: real references are not perfect, so the toolkit is
exercised against references that carry typical errors.

  FEM-class PM2.5 monitor (BAM / TEOM / Fidas): hourly dry-mass, error 1.5 ug/m3 + 3 %
  Reference gas analysers (CO, NO2, SO2): hourly, 3 % + a small offset
  Tracer-based source apportionment (filter chemistry + PMF / CMB): daily class shares, ~25 % per factor
  Observer log (human / camera, 10-minute resolution): dominant source, ~10 % wrong, ~5 % missing
"""
import numpy as np

from polso_twin import params as P

GAS_IDX = {g: i for i, g in enumerate(P.GASES)}
LABELS = P.CLASSES


def fem_pm25(pm25_true_hourly, rng):
    return pm25_true_hourly * (1 + 0.03 * rng.standard_normal(len(pm25_true_hourly))) + 1.5 * rng.standard_normal(len(pm25_true_hourly))


def ref_gases(gases_hourly, rng):
    """gases_hourly: (hours, 6) ppb. Returns dict for CO, NO2, SO2."""
    off = dict(CO=20.0, NO2=1.5, SO2=0.5)
    return {g: np.clip(gases_hourly[:, GAS_IDX[g]] * (1 + 0.03 * rng.standard_normal(len(gases_hourly)))
                       + off[g] * rng.standard_normal(len(gases_hourly)), 0, None) for g in off}


def tracer_shares(class_pm_total, rng, sigma=0.25):
    """class_pm_total: dict class -> daily mean PM2.5. Lognormal factor error, renormalised."""
    v = np.array([max(class_pm_total.get(c, 0.0), 0.0) for c in LABELS], float)
    v = v * np.exp(sigma * rng.standard_normal(len(v)))
    return dict(zip(LABELS, v / v.sum()))


def observer_labels(true_labels, rng, p_wrong=0.10, p_missing=0.05):
    """true_labels: list of true dominant-source labels per 10-minute block."""
    out = []
    for t in true_labels:
        u = rng.random()
        if u < p_missing:
            out.append(None)
        elif u < p_missing + p_wrong and t in LABELS:
            out.append(str(rng.choice([c for c in LABELS if c != t])))
        else:
            out.append(t)
    return out
