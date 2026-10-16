"""
Accuracy-vs-compute study: which model family should run on the STM32G474?

Every candidate is trained on the same simulated chamber + field data and scored on held-out field
sessions from the 10 unseen pilot units. Cost on the MCU is computed from the model structure:
multiply-accumulates (MLP, linear) or comparisons along the decision path (trees), converted to
Cortex-M4F cycles, microseconds, microjoules and flash / RAM bytes.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from polso_twin import params as P, scenarios as S, model as M
from polso_twin.firmware import qmodel_forward
from . import spec

F_MHZ = spec.F_MAX_MHZ
CYC_MAC = {"scalar": spec.MLP_CYCLES_PER_MAC["scalar C"][1], "simd": spec.MLP_CYCLES_PER_MAC["SMLAD / CMSIS-NN"][1]}
FIXED_IN = 25 * 20 + 5 * 140 + 200          # input quantisation + soft-max exp + argmax (cycles)
CYC_NODE, CYC_LEAF = 11.0, 15.0             # per tree comparison; per leaf vote accumulation (cycles)


_CTX = {"groups": None}


def datasets(quick=False, log=print):
    n_ch, n_fd, n_te = (6, 120, 50) if quick else (15, 300, 120)
    chamber = S.chamber_set(n_ch, seed0=1000)
    field = [S.field_session(50_000 + i) for i in range(n_fd)]
    test = [S.field_session(600_000 + i, units=tuple(S.TEST_UNITS)) for i in range(n_te)]
    Xc, yc, _ = M.build_dataset(chamber)
    Xf, yf, _ = M.build_dataset(field)
    Xt, yt, meta_t = M.build_dataset(test)
    t_end = np.array([m["t_end"] for m in meta_t])
    _CTX["groups"] = np.cumsum(np.r_[0, np.diff(t_end) < 0])         # new session whenever time restarts
    log(f"  pareto data: train {len(Xc) + len(Xf)} windows, test {len(Xt)}")
    return np.vstack([Xc, Xf]), np.concatenate([yc, yf]), Xt, yt


def _scores(probs, yt):
    k = M.trainable(yt)
    p, y = probs[k], np.array([M.CLS_IDX[c] for c in yt[k]])
    pred = p.argmax(1)
    conf = p.max(1)
    keep = conf >= np.quantile(conf, 0.10)
    lo = hi = float("nan")
    g = _CTX["groups"]
    if g is not None:                                  # cluster bootstrap over field sessions
        g = g[k]
        sess = [np.nonzero(g == u)[0] for u in np.unique(g)]
        rng = np.random.default_rng(5)
        bs = []
        for _ in range(150):
            idx = np.concatenate([sess[i] for i in rng.integers(0, len(sess), len(sess))])
            bs.append(f1_score(y[idx], pred[idx], average="macro"))
        lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(f1=float(f1_score(y, pred, average="macro")), f1_lo=float(lo), f1_hi=float(hi),
                acc=float((pred == y).mean()), sel_acc90=float((pred[keep] == y[keep]).mean()))


def _mlp_row(X, y, Xt, yt, hidden, features=None, seed=0):
    m = M.train_mlp(X, y, features, seed, hidden)
    feats = m["features"]
    fl = M.float_predict(m, Xt)
    sc = _scores(fl, yt)
    q = M.quantize(m)
    # input layer of the quantised model expects the selected features only
    pq = qmodel_forward(q, Xt[:, feats])[0]
    sq = _scores(pq, yt)
    dims = [len(feats), *hidden, 5]
    macs = sum(a * b for a, b in zip(dims[:-1], dims[1:]))
    neurons = sum(hidden)
    flash = sum(a * b for a, b in zip(dims[:-1], dims[1:])) + 4 * sum(dims[1:]) + 8 * len(feats)
    cyc = {k: macs * c + neurons * spec.MLP_REQUANT_PER_NEURON[1] + FIXED_IN for k, c in CYC_MAC.items()}
    return dict(f1=sq["f1"], f1_lo=sq["f1_lo"], f1_hi=sq["f1_hi"], f1_float=sc["f1"], sel_acc90=sq["sel_acc90"], acc=sq["acc"], macs=macs,
                flash=flash, ram=dims[1] + dims[2] + 64 if len(dims) > 3 else dims[1] + 64,
                cycles=cyc["scalar"], cycles_simd=cyc["simd"])


def _tree_cost(est_list, Xt_sel):
    """Average comparisons per sample and flash bytes of a list of fitted sklearn trees."""
    internal = leaves = 0
    for t in est_list:
        tr = t.tree_
        leaf = tr.children_left == -1
        internal += int((~leaf).sum())
        leaves += int(leaf.sum())
    path = 0.0
    for t in est_list:
        ind = t.decision_path(Xt_sel)
        path += float(ind.sum(1).mean()) - 1.0          # comparisons = nodes visited - 1
    return internal, leaves, path


def _rf_row(X, y, Xt, yt, n_trees, depth, features=None, seed=0):
    feats = list(range(X.shape[1])) if features is None else list(features)
    k = M.trainable(y)
    Xb, yb = M._balance(X[k][:, feats], y[k], np.random.default_rng(seed))
    rf = RandomForestClassifier(n_trees, max_depth=depth, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rf.fit(Xb, [M.CLS_IDX[c] for c in yb])
    pr = np.zeros((len(Xt), 5))
    pr[:, rf.classes_] = rf.predict_proba(Xt[:, feats])
    sc = _scores(pr, yt)
    internal, leaves, path = _tree_cost(rf.estimators_, Xt[:, feats])
    flash = internal * 8 + leaves * 5 + 4 * 0
    cyc = path * CYC_NODE + n_trees * CYC_LEAF + 200
    return dict(f1=sc["f1"], f1_lo=sc["f1_lo"], f1_hi=sc["f1_hi"], f1_float=sc["f1"], sel_acc90=sc["sel_acc90"], acc=sc["acc"], macs=0, flash=flash,
                ram=64, cycles=cyc, cycles_simd=cyc, nodes=internal + leaves, path=path)


def _tree_row(X, y, Xt, yt, depth, seed=0):
    k = M.trainable(y)
    Xb, yb = M._balance(X[k], y[k], np.random.default_rng(seed))
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, random_state=seed)
    dt.fit(Xb, [M.CLS_IDX[c] for c in yb])
    pr = np.zeros((len(Xt), 5))
    pr[:, dt.classes_] = dt.predict_proba(Xt)
    sc = _scores(pr, yt)
    internal, leaves, path = _tree_cost([dt], Xt)
    return dict(f1=sc["f1"], f1_lo=sc["f1_lo"], f1_hi=sc["f1_hi"], f1_float=sc["f1"], sel_acc90=sc["sel_acc90"], acc=sc["acc"], macs=0,
                flash=internal * 8 + leaves * 5, ram=64, cycles=path * CYC_NODE + CYC_LEAF + 200,
                cycles_simd=path * CYC_NODE + CYC_LEAF + 200)


def _logreg_row(X, y, Xt, yt, seed=0):
    k = M.trainable(y)
    Xb, yb = M._balance(X[k], y[k], np.random.default_rng(seed))
    mu, sd = Xb.mean(0), Xb.std(0) + 1e-6
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((Xb - mu) / sd, [M.CLS_IDX[c] for c in yb])
    pr = np.zeros((len(Xt), 5))
    pr[:, lr.classes_] = lr.predict_proba((Xt - mu) / sd)
    sc = _scores(pr, yt)
    macs = 25 * 5
    return dict(f1=sc["f1"], f1_lo=sc["f1_lo"], f1_hi=sc["f1_hi"], f1_float=sc["f1"], sel_acc90=sc["sel_acc90"], acc=sc["acc"], macs=macs,
                flash=(macs + 5) * 4 + 8 * 25, ram=64, cycles=macs * 5.0 + 25 * 12 + 5 * 140 + 200,
                cycles_simd=macs * 5.0 + 25 * 12 + 5 * 140 + 200)


def run(quick=False, log=print):
    X, y, Xt, yt = datasets(quick, log)
    rows = []

    def add(family, config, r):
        r.update(family=family, config=config)
        r["us"] = r["cycles"] / F_MHZ
        r["uJ"] = r["us"] * 1e-6 * spec.i_run_ma(F_MHZ) * 1e-3 * spec.V_RAIL * 1e6
        rows.append(r)
        log(f"    {family:14s} {config:12s} F1 {r['f1']:.3f} [{r['f1_lo']:.3f}-{r['f1_hi']:.3f}]  flash {r['flash'] / 1024:7.1f} KB  cycles {r['cycles']:>9,.0f}")

    add("Logistic", "linear", _logreg_row(X, y, Xt, yt))
    for h in [(8,), (16,), (16, 8), (32, 16), (64, 32), (128, 64)]:
        add("MLP int8", "-".join(map(str, h)), _mlp_row(X, y, Xt, yt, h))
    for d in (6, 10):
        add("Decision tree", f"depth {d}", _tree_row(X, y, Xt, yt, d))
    for n, d in ((10, 6), (10, 10), (30, 8), (30, 12), (100, 8), (100, 12)):
        add("Random forest", f"{n} trees d{d}", _rf_row(X, y, Xt, yt, n, d))
    df = pd.DataFrame(rows)

    # feature-fragility ablation: drop the noisiest features and see what it costs
    names = P.FEATURE_NAMES
    drops = {"all 25 features": [], "without pm_gas_corr": [21],
             "without pm_gas_corr + skew + cv": [21, 10, 9]}
    abl = {}
    for label, d in drops.items():
        keep = [i for i in range(25) if i not in d]
        abl[label] = dict(
            mlp=_mlp_row(X, y, Xt, yt, (32, 16), keep)["f1"],
            rf=_rf_row(X, y, Xt, yt, 30, 8, keep)["f1"])
        log(f"    ablation {label:34s} MLP {abl[label]['mlp']:.3f}  RF {abl[label]['rf']:.3f}")
    return df, abl, dict(n_train=len(X), n_test=len(Xt))
