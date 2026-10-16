"""Figures for the verification rehearsal."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from polso_twin import params as P
from polso_twin.report import _png, INK, INK2, GRID, CLR
from . import metrics as MT

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def fig_pm_scatter(R):
    h = R["pm_hourly"]
    test = h
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1), sharex=True, sharey=True)
    lim = float(np.quantile(test["ref"], 0.99)) * 1.15
    for ax, tag, col, title in ((axs[0], "raw", "pm", "Device as it reads"),
                                (axs[1], "corrected", "pm_corr", "After humidity correction"),
                                (axs[2], "source-aware", "pm_src", "After source-aware correction")):
        m = R["pm"][tag]["pooled"]
        ax.scatter(test["ref"], test[col].clip(upper=lim * 3), s=9, alpha=0.55, color=BLUE, lw=0)
        xs = np.array([0, lim])
        ax.plot(xs, xs, color=INK, lw=1)
        ax.fill_between(xs, 0.65 * xs - 5, 1.35 * xs + 5, color=AQUA, alpha=0.12, lw=0)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{title}\nslope {m['slope']:.2f}  R² {m['r2']:.2f}  RMSE {m['rmse']:.0f} µg/m³", fontsize=9, loc="left")
        ax.set_xlabel("reference PM2.5, hourly (µg/m³)")
    axs[0].set_ylabel("device PM2.5, hourly (µg/m³)")
    fig.text(0.99, 0.005, "green band = EPA slope 1.0 ± 0.35 with ±5 µg/m³ intercept; every hour predicted by a correction fitted on other days", ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return _png(fig)


def fig_bias_by_source(df):
    d = df[df["pm25"].notna() & (df["total"] > 15) & df["truth"].isin(P.CLASSES)]
    ratio = (d["pm25"] / d["total"]).clip(upper=6)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    data = [ratio[d["truth"] == c].values for c in P.CLASSES]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for p, c in zip(bp["boxes"], P.CLASSES):
        p.set(facecolor=CLR[c], alpha=0.75, edgecolor=INK2)
    for m in bp["medians"]:
        m.set(color=INK, lw=1.5)
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(1, 6), [c.replace("_", "\n") for c in P.CLASSES], fontsize=8)
    ax.set_ylabel("device PM2.5 / true PM2.5")
    ax.set_title("The sensor reads differently for different sources", loc="left", fontsize=9)
    fig.tight_layout()
    return _png(fig)


def fig_confusion(R):
    cm = R["confusion_observer"]
    arr = cm.values.astype(float)
    arr = arr / np.maximum(arr.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.imshow(arr, cmap="Blues", vmin=0, vmax=1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if cm.values[i, j]:
                ax.text(j, i, f"{arr[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if arr[i, j] > 0.55 else INK)
    ax.set_xticks(range(arr.shape[1]), [c.replace("_", "\n") for c in cm.columns], fontsize=7.5)
    ax.set_yticks(range(arr.shape[0]), [c.replace("_", " ") for c in cm.index], fontsize=8)
    ax.set(xlabel="device output (10-min majority)", ylabel="observer log")
    ax.grid(False)
    fig.tight_layout()
    return _png(fig)


def fig_calibration(R):
    rel = R["reliability"]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    xs = [r[0] for r in rel]
    ys = [r[1] for r in rel]
    ax.plot(xs, ys, "-o", color=BLUE, ms=5)
    for x, y, n in rel:
        ax.text(x, y - 0.06, str(n), fontsize=7, ha="center", color=INK2)
    ax.set(xlim=(0.3, 1.02), ylim=(0.3, 1.02), xlabel="stated confidence", ylabel="fraction correct")
    ax.set_title(f"Calibration  (ECE {R['ece']:.3f})", loc="left", fontsize=9)
    fig.tight_layout()
    return _png(fig)


def fig_share(R):
    c = R["apportion"]["cells"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for cls in P.CLASSES:
        g = c[c["cls"] == cls]
        ax.scatter(g["reference"], g["device"], s=34, color=CLR[cls], label=cls.replace("_", " "), lw=0, alpha=0.9)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="tracer-chemistry share of PM2.5", ylabel="device share (PM-weighted)")
    ax.set_title(f"Daily source shares   MAE {R['apportion']['mae']:.2f}, rank corr {R['apportion']['rho']:.2f}", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_samples(R):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))
    k, w = zip(*R["ci_curve"])
    axs[0].plot(k, w, "-o", color=BLUE)
    axs[0].set(xlabel="days of data", ylabel="width of the 95 % interval on macro-F1", title="How much data is enough")
    axs[0].title.set_fontsize(9)
    n, lo, hi = zip(*R["wilson"])
    axs[1].fill_between(n, lo, hi, color=ORANGE, alpha=0.25)
    axs[1].plot(n, [0.9] * len(n), color=ORANGE)
    axs[1].set_xscale("log")
    axs[1].set(xlabel="independent episodes of one source", ylabel="95 % interval on a true recall of 0.90",
               title="Per-source recall needs episodes, not windows")
    axs[1].title.set_fontsize(9)
    fig.tight_layout()
    return _png(fig)
