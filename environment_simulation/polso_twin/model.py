"""Dataset building, training, int8 quantisation and C export of the edge classifier."""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from . import params as P
from .simulate import run_scenario, window_truth
from .firmware import run_device, qmodel_forward

CLS_IDX = {c: i for i, c in enumerate(P.CLASSES)}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def _windows_of(sc, qm=None, **dev_kw):
    streams, truth = run_scenario(sc)
    out = run_device(streams, qm, **dev_kw)
    rows = []
    for w, e in enumerate(out["t_end"]):
        lab, shares, total = window_truth(truth, int(e))
        rows.append((lab, shares, total))
    return sc, out, truth, rows


def _feature_rows(sc):
    sc, out, truth, rows = _windows_of(sc)
    ok = out["status"] == "ok"
    labels = [r[0] for r in rows]
    return out["features"][ok], [l for l, k in zip(labels, ok) if k], \
        [dict(scenario=sc.name, unit=sc.unit_id, t_end=int(e), shares=r[1], total=r[2])
         for e, r, k in zip(out["t_end"], rows, ok) if k]


def pmap(fn, items, workers=None):
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if len(items) < 8 or workers == 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, items, chunksize=4))


def build_dataset(scenarios):
    X, y, meta = [], [], []
    for f, l, m in pmap(_feature_rows, scenarios):
        X.append(f)
        y += l
        meta += m
    return np.vstack(X), np.array(y, object), meta


def trainable(y):
    """Windows with a single clear class (drops 'mixed' and confounder windows)."""
    return np.isin(y, P.CLASSES)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _balance(X, y, rng):
    idx = [np.nonzero(y == c)[0] for c in P.CLASSES]
    n = max(len(i) for i in idx)
    sel = np.concatenate([rng.choice(i, n, replace=len(i) < n) for i in idx if len(i)])
    return X[sel], y[sel]


def train_mlp(X, y, features=None, seed=0, hidden=(32, 16)):
    features = list(range(X.shape[1])) if features is None else list(features)
    rng = np.random.default_rng(seed)
    k = trainable(y)
    Xb, yb = _balance(X[k][:, features], y[k], rng)
    mu = Xb.mean(0)
    sd = Xb.std(0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=3e-3, max_iter=400, early_stopping=True,
                        n_iter_no_change=20, random_state=seed, learning_rate_init=2e-3)
    yi = np.array([CLS_IDX[c] for c in yb])
    clf.fit((Xb - mu) / sd, yi)
    return dict(clf=clf, mu=mu, sd=sd, features=features, Xcal=Xb)


def train_rf(X, y, features=None, seed=0):
    features = list(range(X.shape[1])) if features is None else list(features)
    k = trainable(y)
    Xb, yb = _balance(X[k][:, features], y[k], np.random.default_rng(seed))
    rf = RandomForestClassifier(150, max_depth=12, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rf.fit(Xb, [CLS_IDX[c] for c in yb])
    return dict(rf=rf, features=features)


def float_predict(m, X):
    Xf = X[:, m["features"]]
    if "rf" in m:
        p = m["rf"].predict_proba(Xf)
        full = np.zeros((len(X), len(P.CLASSES)))
        full[:, m["rf"].classes_] = p
        return full
    return m["clf"].predict_proba((Xf - m["mu"]) / m["sd"])


# ---------------------------------------------------------------------------
# int8 quantisation (integer-only hidden layers, TFLite-style requantisation)
# ---------------------------------------------------------------------------
def _fixed_multiplier(real):
    sh = 31 - int(np.ceil(np.log2(real)))
    m = int(round(real * (1 << sh)))
    if m >= (1 << 31):
        m //= 2
        sh -= 1
    return m, sh


def quantize(m, conf_thr=0.6):
    """Full-feature MLP only (the deployed model)."""
    clf, mu, sd = m["clf"], m["mu"].astype(np.float32), m["sd"].astype(np.float32)
    xs = (m["Xcal"] - mu) / sd
    s_in = float(np.percentile(np.abs(xs), 99.9)) / 127.0
    qm = dict(mu=mu, inv_sd=(1.0 / sd).astype(np.float32), inv_s0=np.float32(1.0 / s_in),
              layers=[], conf_thr=conf_thr, features=m["features"])
    a = xs
    n_layers = len(clf.coefs_)
    for li, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        W = W.T                                         # (out, in)
        s_w = float(np.max(np.abs(W))) / 127.0
        Wq = np.clip(np.round(W / s_w), -127, 127).astype(np.int8)
        bq = np.round(b / (s_in * s_w)).astype(np.int32)
        a = a @ W.T + b
        if li < n_layers - 1:
            a = np.maximum(a, 0)
            s_out = max(float(np.percentile(a, 99.9)), 1e-6) / 127.0
            mult, sh = _fixed_multiplier(s_in * s_w / s_out)
            qm["layers"].append(dict(W=Wq, b=bq, m=mult, sh=sh))
            s_in = s_out
        else:
            qm["layers"].append(dict(W=Wq, b=bq, scale=np.float32(s_in * s_w)))
    return qm


def choose_threshold(probs, y, coverage=0.90):
    """Confidence gate chosen on validation data: answer the most confident `coverage` share of
    clear windows and say 'uncertain' on the rest (standard selective classification).

    (An earlier rule tuned on mixed-source windows pushed the gate to 0.95 and made the device
    abstain far too often - see run_log_v2_gate095.txt.)
    """
    conf = probs.max(1)[trainable(y)]
    return float(np.clip(np.floor(np.quantile(conf, 1 - coverage) * 100) / 100, 0.30, 0.95))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def save_qmodel(qm, path):
    d = dict(mu=qm["mu"].tolist(), inv_sd=qm["inv_sd"].tolist(), inv_s0=float(qm["inv_s0"]),
             conf_thr=qm["conf_thr"], features=list(map(int, qm["features"])), layers=[])
    for L in qm["layers"]:
        e = dict(W=L["W"].tolist(), b=L["b"].tolist())
        e.update({k: (float(L[k]) if k == "scale" else int(L[k])) for k in ("m", "sh", "scale") if k in L})
        d["layers"].append(e)
    with open(path, "w") as f:
        json.dump(d, f)


def load_qmodel(path):
    d = json.load(open(path))
    qm = dict(mu=np.array(d["mu"], np.float32), inv_sd=np.array(d["inv_sd"], np.float32),
              inv_s0=np.float32(d["inv_s0"]), conf_thr=d["conf_thr"], features=d["features"],
              layers=[])
    for L in d["layers"]:
        e = dict(W=np.array(L["W"], np.int8), b=np.array(L["b"], np.int32))
        for k in ("m", "sh"):
            if k in L:
                e[k] = int(L[k])
        if "scale" in L:
            e["scale"] = np.float32(L["scale"])
        qm["layers"].append(e)
    return qm


def _carr(name, ctype, arr, fmt="{}"):
    flat = np.asarray(arr).ravel()
    body = ", ".join(fmt.format(v) for v in flat)
    return f"static const {ctype} {name}[{len(flat)}] = {{{body}}};\n"


def export_c_header(qm, path):
    """Generate firmware_c/model_params.h from the quantised model and firmware constants."""
    L = qm["layers"]
    dims = [L[0]["W"].shape[1]] + [l["W"].shape[0] for l in L]
    h = ["/* AUTO-GENERATED by polso_twin.model.export_c_header - do not edit. */\n",
         "#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n#include <stdint.h>\n\n",
         f"#define MP_N_IN {dims[0]}\n#define MP_H1 {dims[1]}\n#define MP_H2 {dims[2]}\n#define MP_N_OUT {dims[3]}\n",
         f"#define MP_CONF_THR {qm['conf_thr']:.6f}f\n",
         f"#define MP_INV_S0 {float(qm['inv_s0']):.9e}f\n",
         f"#define MP_L1_M {L[0]['m']}\n#define MP_L1_SH {L[0]['sh']}\n",
         f"#define MP_L2_M {L[1]['m']}\n#define MP_L2_SH {L[1]['sh']}\n",
         f"#define MP_L3_SCALE {float(L[2]['scale']):.9e}f\n\n",
         _carr("MP_MU", "float", qm["mu"], "{:.9e}f"),
         _carr("MP_INV_SD", "float", qm["inv_sd"], "{:.9e}f"),
         _carr("MP_W1", "int8_t", L[0]["W"]), _carr("MP_B1", "int32_t", L[0]["b"]),
         _carr("MP_W2", "int8_t", L[1]["W"]), _carr("MP_B2", "int32_t", L[1]["b"]),
         _carr("MP_W3", "int8_t", L[2]["W"]), _carr("MP_B3", "int32_t", L[2]["b"]),
         "\n/* firmware constants mirrored from params.py */\n",
         _carr("FW_HEATER_PROFILE", "uint16_t", P.HEATER_PROFILE),
         _carr("FW_GROUP_OF_STEP", "uint8_t",
               [int(np.where(P.HEATER_GROUPS == t)[0][0]) for t in P.HEATER_PROFILE]),
         _carr("FW_GAMMA_STEP", "float", P.mox_humidity_gamma(P.HEATER_PROFILE), "{:.9e}f"),
         f"#define FW_WINDOW_S {P.FW_WINDOW_S}\n#define FW_HOP_S {P.FW_HOP_S}\n#define FW_WARMUP_S {P.FW_WARMUP_S}\n",
         f"#define FW_PM_VALID_FRAC {P.FW_PM_VALID_FRAC}f\n#define FW_GAS_VALID_FRAC {P.FW_GAS_VALID_FRAC}f\n",
         f"#define FW_BASE_UP {P.FW_BASE_UP}f\n#define FW_BASE_DOWN {P.FW_BASE_DOWN}f\n",
         f"#define FW_PROB_EMA {P.FW_PROB_EMA}f\n#define FW_AH_REF {P.AH_REF}f\n",
         f"#define FW_AMB_TEMP_COEF {P.MOX_AMB_TEMP_COEF}f\n#define FW_SELF_HEAT {P.BME_SELF_HEAT}f\n",
         f"#define FW_R_MIN {P.MOX_R_MIN}f\n#define FW_R_MAX {P.MOX_R_MAX}f\n",
         f"#define FW_N_STEPS {len(P.HEATER_PROFILE)}\n#define FW_N_GROUPS {len(P.HEATER_GROUPS)}\n",
         "\n#endif\n"]
    with open(path, "w") as f:
        f.write("".join(h))
    weights_bytes = sum(l["W"].size + 4 * l["b"].size for l in L) + 8 * dims[0]
    return weights_bytes
