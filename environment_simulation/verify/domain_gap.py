"""
How far is the simulator from reality?  (run this on the first week of real data)

Given feature windows from the simulator and from real hardware (the 25 model features, same definitions),
it answers two questions:

  1. Can a classifier tell them apart?  Cross-validated AUC of "simulated vs real". 0.5 = indistinguishable,
     1.0 = completely different. This is the single number to track as params.py is re-fitted.
  2. Which features differ most, and in which direction?  Per-feature standardised median shift and a
     two-sample KS statistic. Each feature maps to a parameter family in params.py (printed alongside),
     so the first fixes are obvious.

Use clean-air / background windows only for a first pass (no source labels needed), then each source class.
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

from polso_twin import params as P

FEATURE_TO_PARAMS = {
    "lfrac": "PMS size response: PMS_CE_LOW_D50 / PMS_SIZING_SIGMA / aerosol modes in SUBTYPES",
    "log_count": "PMS counting efficiency, PMS unit gain, emission strength, inlet loss",
    "pm25/pm10": "coarse-particle response: PMS_CE_HIGH_D50, enclosure losses",
    "pm1/pm25": "fine-mode size distribution, PMS sizing blur",
    "cv_count": "PMS counting noise (PMS_SAMPLE_CM3), source variability patterns",
    "skew_count": "source temporal patterns (environment._pattern)",
    "max_rise": "source temporal patterns, PMS smoothing (PMS_SMOOTH_TAU)",
    "gas_resp": "MOX baseline and sensitivity: MOX_GAS, UNIT_MOX_*, FW_BASE_* tracker",
    "gas_shape": "MOX heater-temperature selectivity: MOX_GAS[...]['Tpk','w'], humidity coefficient",
    "gas_slope": "MOX adsorption / desorption time constants tau_up / tau_dn",
    "pm_gas_corr": "co-variation of PM and gas in the source (emission ratios in SUBTYPES)",
    "gas_per_pm": "emission ratios (SUBTYPES gas), background gas offsets",
    "rh": "BME688 RH bias (UNIT, BME_RH_UNIT_SD), weather generator",
    "temp": "BME688 self-heating (BME_SELF_HEAT), weather generator",
}


def _family(name):
    for k in FEATURE_TO_PARAMS:
        if name.startswith(k):
            return FEATURE_TO_PARAMS[k]
    return ""


def compare(X_sim, X_real, seed=0):
    X_sim, X_real = np.asarray(X_sim, float), np.asarray(X_real, float)
    n = min(len(X_sim), len(X_real), 8000)
    rng = np.random.default_rng(seed)
    a = X_sim[rng.choice(len(X_sim), n, replace=False)]
    b = X_real[rng.choice(len(X_real), n, replace=False)]
    X = np.vstack([a, b])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=120, max_depth=4, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, p))
    rows = []
    for j, name in enumerate(P.FEATURE_NAMES):
        sd = np.std(np.r_[a[:, j], b[:, j]]) + 1e-9
        rows.append(dict(feature=name, shift_sd=float((np.median(b[:, j]) - np.median(a[:, j])) / sd),
                         ks=float(ks_2samp(a[:, j], b[:, j]).statistic), fix_first=_family(name)))
    rows.sort(key=lambda r: -r["ks"])
    return dict(auc=auc, features=rows)


if __name__ == "__main__":
    import sys
    import pandas as pd
    if len(sys.argv) < 3:
        sys.exit("usage: python -m verify.domain_gap sim_features.csv real_features.csv   (columns f00..f24)")
    cols = [f"f{i:02d}" for i in range(P.N_FEATURES)]
    r = compare(pd.read_csv(sys.argv[1])[cols].values, pd.read_csv(sys.argv[2])[cols].values)
    print(f"simulated-vs-real AUC = {r['auc']:.3f}  (0.5 = indistinguishable)")
    for f in r["features"][:10]:
        print(f"  {f['feature']:16s} KS {f['ks']:.2f}  shift {f['shift_sd']:+.2f} sd   fix first: {f['fix_first']}")
