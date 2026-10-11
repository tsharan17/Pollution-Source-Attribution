"""
Pollution Source Attribution — Physics-Grounded Simulation
=============================================================
A more accurate simulation than a pure toy model:

  PM channel   : each source is modeled as a lognormal aerosol size
                 distribution (scipy.stats.lognorm). Particle-bin counts
                 are obtained by integrating the PDF between the
                 PMS7003 bin edges, then corrupted with Poisson counting
                 noise (physically correct for particle counters).

  Gas channel  : each source drives the MOX/BME688 sensor resistance
                 through a first-order relaxation ODE
                     dR/dt = (R_ss - R) / tau
                 integrated with scipy.integrate.solve_ivp. Per-window
                 features (tau, R_ss, initial slope) are recovered from
                 noisy synthetic traces with scipy.optimize.curve_fit,
                 mimicking real on-device parameter estimation.

  Fusion + ML  : concatenated PM + gas features -> PCA -> RandomForest,
                 evaluated with scikit-learn; results are visualized in
                 a single matplotlib summary figure (aerosol physics,
                 sensor dynamics, PCA separation, confusion matrix,
                 feature importance, ROC curves).

Run:  python3 pollution_source_attribution_scipy_sim.py
Output: pollution_source_attribution_results.png
"""

import numpy as np
from scipy.stats import lognorm
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

rng = np.random.default_rng(7)

CLASSES = ["vehicle_exhaust", "biomass_burning", "dust", "industrial", "background"]
COLORS = {"vehicle_exhaust": "#d95f02", "biomass_burning": "#e6194B",
          "dust": "#b8860b", "industrial": "#377eb8", "background": "#4daf4a"}

# PMS7003-style bin edges (um); counts reported per bin between edges
BIN_EDGES = np.array([0.3, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0])
BIN_LABELS = ["0.3-0.5", "0.5-1.0", "1.0-2.5", "2.5-5.0", "5.0-10", "10-15"]

# Lognormal aerosol parameters per source: (geometric mean diameter um, geometric std dev, total number conc.)
AEROSOL_PARAMS = {
    "vehicle_exhaust": dict(gmd=0.40, gsd=1.8, n_total=5000),
    "biomass_burning": dict(gmd=0.60, gsd=2.0, n_total=8000),
    "dust":            dict(gmd=4.00, gsd=1.9, n_total=2000),
    "industrial":      dict(gmd=1.20, gsd=2.2, n_total=3500),
    "background":      dict(gmd=0.30, gsd=1.6, n_total=300),
}

# Gas-sensor relaxation params per source: (R0 kOhm, R_ss kOhm, tau seconds)
GAS_PARAMS = {
    "vehicle_exhaust": dict(R0=100.0, Rss=40.0, tau=8.0),
    "biomass_burning": dict(R0=100.0, Rss=25.0, tau=15.0),
    "dust":            dict(R0=100.0, Rss=90.0, tau=3.0),
    "industrial":      dict(R0=100.0, Rss=55.0, tau=25.0),
    "background":      dict(R0=100.0, Rss=95.0, tau=5.0),
}

# ---------------------------------------------------------------
# 1. PM: lognormal aerosol physics -> binned particle counts
# ---------------------------------------------------------------
def lognormal_bin_fractions(gmd, gsd):
    """Integrate the lognormal PDF between bin edges (scipy.stats.lognorm)."""
    shape = np.log(gsd)          # sigma of underlying normal
    scale = gmd                  # median diameter
    cdf = lognorm.cdf(BIN_EDGES, s=shape, scale=scale)
    fractions = np.diff(cdf)
    fractions = fractions / fractions.sum()   # renormalize within sensor's measurable range
    return fractions

def simulate_pm_window(cls, n_samples=30):
    p = AEROSOL_PARAMS[cls]
    fractions = lognormal_bin_fractions(p["gmd"], p["gsd"])
    expected = fractions * p["n_total"]
    # Poisson counting noise, one draw per 1s sample in the 30s window
    counts = rng.poisson(lam=np.clip(expected, 1e-3, None), size=(n_samples, len(expected)))
    return counts.astype(float), fractions

# ---------------------------------------------------------------
# 2. Gas: first-order ODE sensor relaxation (scipy.integrate.solve_ivp)
# ---------------------------------------------------------------
def gas_ode(t, R, Rss, tau):
    return [(Rss - R[0]) / tau]

def simulate_gas_window(cls, n_samples=30, dt=1.0):
    p = GAS_PARAMS[cls]
    t_eval = np.arange(n_samples) * dt
    sol = solve_ivp(gas_ode, [0, t_eval[-1]], y0=[p["R0"]],
                     t_eval=t_eval, args=(p["Rss"], p["tau"]))
    clean = sol.y[0]
    noisy = clean + rng.normal(0, 1.2, size=clean.shape)   # sensor read noise
    return noisy, t_eval

def exp_model(t, Rss, R0, tau):
    return Rss + (R0 - Rss) * np.exp(-t / tau)

def fit_gas_response(trace, t_eval, R0_hint):
    """Recover (tau, Rss) from a noisy trace via scipy.optimize.curve_fit,
    the way an edge device would estimate sensor dynamics on the fly."""
    try:
        popt, _ = curve_fit(
            lambda t, Rss, tau: exp_model(t, Rss, R0_hint, tau),
            t_eval, trace, p0=[trace[-1], 10.0],
            bounds=([0, 0.5], [200, 60]), maxfev=4000,
        )
        Rss_hat, tau_hat = popt
    except RuntimeError:
        Rss_hat, tau_hat = trace[-1], 10.0
    initial_slope = (trace[min(3, len(trace) - 1)] - trace[0]) / (t_eval[min(3, len(t_eval) - 1)] - t_eval[0] + 1e-6)
    return Rss_hat, tau_hat, initial_slope

# ---------------------------------------------------------------
# 3. Feature extraction (per window fingerprint)
# ---------------------------------------------------------------
def extract_features(pm_counts, gas_trace, t_eval, R0_hint=100.0):
    total = pm_counts.sum(axis=1, keepdims=True)
    bin_frac = (pm_counts / np.clip(total, 1, None)).mean(axis=0)   # normalized size fingerprint

    bin_mids = np.sqrt(BIN_EDGES[:-1] * BIN_EDGES[1:])               # geometric bin midpoints
    log_mids = np.log(bin_mids)
    mean_counts = pm_counts.mean(axis=0)
    w = mean_counts / (mean_counts.sum() + 1e-9)
    gmd_hat = np.exp((w * log_mids).sum())                           # weighted geometric mean diameter
    gsd_hat = np.exp(np.sqrt(np.clip((w * (log_mids - np.log(gmd_hat)) ** 2).sum(), 1e-6, None)))
    total_count = mean_counts.sum()

    Rss_hat, tau_hat, slope = fit_gas_response(gas_trace, t_eval, R0_hint)

    return np.concatenate([bin_frac, [gmd_hat, gsd_hat, total_count, Rss_hat, tau_hat, slope]])

FEATURE_NAMES = BIN_LABELS + ["GMD_hat", "GSD_hat", "total_count", "Rss_hat", "tau_hat", "gas_slope"]

def build_dataset(samples_per_class=140):
    X, y = [], []
    for cls in CLASSES:
        for _ in range(samples_per_class):
            pm_counts, _ = simulate_pm_window(cls)
            gas_trace, t_eval = simulate_gas_window(cls)
            X.append(extract_features(pm_counts, gas_trace, t_eval))
            y.append(cls)
    return np.array(X), np.array(y)

# ---------------------------------------------------------------
# 4. Train / evaluate fusion pipeline
# ---------------------------------------------------------------
def train_and_evaluate():
    X, y = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    pca = PCA(n_components=6, random_state=1).fit(Xtr_s)
    Xtr_p, Xte_p = pca.transform(Xtr_s), pca.transform(Xte_s)

    clf = RandomForestClassifier(n_estimators=150, max_depth=9, random_state=1)
    clf.fit(Xtr_p, ytr)
    preds = clf.predict(Xte_p)

    print("=== Classification report (held-out test set) ===")
    print(classification_report(yte, preds))

    return dict(X=X, y=y, scaler=scaler, pca=pca, clf=clf,
                Xte_p=Xte_p, yte=yte, preds=preds)

# ---------------------------------------------------------------
# 5. Matplotlib summary figure
# ---------------------------------------------------------------
def make_figure(res):
    X, y, scaler, pca, clf = res["X"], res["y"], res["scaler"], res["pca"], res["clf"]
    Xte_p, yte, preds = res["Xte_p"], res["yte"], res["preds"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Pollution Source Attribution — Sensor Fusion Simulation", fontsize=15, fontweight="bold")

    # (A) Lognormal aerosol size distributions
    ax = axes[0, 0]
    d = np.logspace(np.log10(0.1), np.log10(15), 400)
    for cls in CLASSES:
        p = AEROSOL_PARAMS[cls]
        pdf = lognorm.pdf(d, s=np.log(p["gsd"]), scale=p["gmd"])
        ax.plot(d, pdf, label=cls.replace("_", " "), color=COLORS[cls], lw=2)
    for e in BIN_EDGES:
        ax.axvline(e, color="grey", lw=0.5, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Particle diameter (µm)")
    ax.set_ylabel("PDF (lognormal)")
    ax.set_title("A. Aerosol size distributions by source")
    ax.legend(fontsize=7, loc="upper right")

    # (B) Gas sensor ODE relaxation curves + noisy trace + fit
    ax = axes[0, 1]
    for cls in CLASSES:
        trace, t_eval = simulate_gas_window(cls)
        Rss_hat, tau_hat, _ = fit_gas_response(trace, t_eval, GAS_PARAMS[cls]["R0"])
        ax.plot(t_eval, trace, "o", ms=2.5, color=COLORS[cls], alpha=0.5)
        ax.plot(t_eval, exp_model(t_eval, Rss_hat, GAS_PARAMS[cls]["R0"], tau_hat),
                 color=COLORS[cls], lw=2, label=f"{cls.replace('_',' ')} (τ̂={tau_hat:.1f}s)")
    ax.set_xlabel("Time in window (s)")
    ax.set_ylabel("Sensor resistance (kΩ)")
    ax.set_title("B. Gas-sensor relaxation: ODE + curve_fit recovery")
    ax.legend(fontsize=6.5, loc="upper right")

    # (C) PCA scatter of fused features
    ax = axes[0, 2]
    Xs = scaler.transform(X)
    X2 = PCA(n_components=2, random_state=1).fit_transform(Xs)
    for cls in CLASSES:
        m = y == cls
        ax.scatter(X2[m, 0], X2[m, 1], s=10, alpha=0.6, color=COLORS[cls], label=cls.replace("_", " "))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("C. Fused feature space (PCA, 2D view)")
    ax.legend(fontsize=6.5, loc="best")

    # (D) Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(yte, preds, labels=CLASSES)
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=7)
    ax.set_yticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=7)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("D. Confusion matrix (held-out test)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (E) Feature importance
    ax = axes[1, 1]
    pc_importance = clf.feature_importances_
    # project PCA-component importance back to original feature loadings for interpretability
    loadings = np.abs(pca.components_).T @ pc_importance
    order = np.argsort(loadings)[::-1]
    ax.barh([FEATURE_NAMES[i] for i in order][::-1], loadings[order][::-1], color="#377eb8")
    ax.set_title("E. Feature importance (approx., mapped to raw features)")
    ax.set_xlabel("Importance (a.u.)")

    # (F) ROC curves (one-vs-rest)
    ax = axes[1, 2]
    clf_classes = list(clf.classes_)          # RandomForest sorts classes alphabetically internally
    y_bin_test = label_binarize(yte, classes=clf_classes)
    probs = clf.predict_proba(Xte_p)
    for i, cls in enumerate(clf_classes):
        fpr, tpr, _ = roc_curve(y_bin_test[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=COLORS[cls], lw=1.8,
                 label=f"{cls.replace('_',' ')} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("F. ROC curves (one-vs-rest)")
    ax.legend(fontsize=6.5, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = "pollution_source_attribution_results.png"
    fig.savefig(out_path, dpi=170)
    print(f"Saved figure -> {out_path}")

if __name__ == "__main__":
    results = train_and_evaluate()
    make_figure(results)
