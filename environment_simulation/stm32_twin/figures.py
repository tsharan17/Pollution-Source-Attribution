"""Figures for the STM32 twin report (matplotlib -> base64 PNG)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from polso_twin.report import _png, INK, INK2, GRID
from . import schedule as SC

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                         "#008300", "#4a3aa7", "#e34948")
STATE = {"Run": ORANGE, "Sleep": BLUE, "Stop 1": AQUA}
FAMILY = {"Logistic": GREEN, "MLP int8": BLUE, "Decision tree": YELLOW, "Random forest": VIOLET}

LANES = ["PMS7003 UART+DMA", "GPS UART+DMA", "BME688 I2C", "SD SPI+DMA",
         "PMS ISR + parse", "GPS ISR + parse", "I2C ISR", "1 Hz sample task", "ADC",
         "Window: features+MLP", "FatFs CPU", "SD busy-wait"]


def fig_timeline(rec0, rec4, t0=29.60, t1=30.80):
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, rec, title in ((axs[0], rec0, "A0  typical starter project"), (axs[1], rec4, "A4  optimised architecture")):
        lanes = [l for l in LANES if any(e[0] == l for e in rec["nom"]["events"])]
        for k, lane in enumerate(lanes):
            for name, st, en in rec["nom"]["events"]:
                if name != lane:
                    continue
                m = (en > t0) & (st < t1)
                if not m.any():
                    continue
                col = BLUE if lane in LANES[:4] else ORANGE
                for s, e in zip(st[m], en[m]):
                    ax.barh(k, max(e - s, 0.0025), left=s, height=0.62, color=col, alpha=0.9, lw=0)
        ax.set_yticks(range(len(lanes)), lanes, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlim(t0, t1)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axs[1].set_xlabel("time (s)   |   blue = peripheral moving data (needs clocks, i.e. Sleep), orange = CPU running; "
                      "bars thinner than 2.5 ms are drawn 2.5 ms wide")
    fig.tight_layout()
    return _png(fig)


def fig_zoom(rec4, t0=29.9490, t1=30.0000):
    fig, ax = plt.subplots(figsize=(11, 2.3))
    names = ["1 Hz sample task", "Window: features+MLP", "FatFs CPU", "I2C ISR", "ADC"]
    for k, lane in enumerate(names):
        for name, st, en in rec4["nom"]["events"]:
            if name != lane:
                continue
            m = (en > t0) & (st < t1)
            for s, e in zip(st[m], en[m]):
                ax.barh(k, (e - s) * 1e3, left=(s - t0) * 1e3, height=0.62, color=ORANGE, lw=0)
                ax.text((e - t0) * 1e3 + 0.4, k, f"{(e - s) * 1e6:,.0f} µs", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ms after 29.949 s   (one 30 s window boundary: features + MLP, then the log write)")
    ax.set_xlim(0, (t1 - t0) * 1e3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def fig_current(rec0, rec4, t0=29.60, t1=30.80):
    fig, axs = plt.subplots(3, 1, figsize=(11, 6.6))
    t, i = SC.current_trace(rec0["arch"], rec0["nom"]["cpu"], rec0["nom"]["per"], t0, t1)
    n0, n4 = rec0["nom"], rec4["nom"]
    ma = np.convolve(i, np.ones(400) / 400, mode="same")           # 20 ms moving average
    axs[0].fill_between(t[200:-200], 0, ma[200:-200], color=BLUE, alpha=0.35, lw=0)
    axs[0].plot(t[200:-200], ma[200:-200], color=BLUE, lw=1)
    axs[0].set_title(f"A0  typical starter project, 1.2 s, 20 ms moving average   -   average {n0['avg_ma']:.2f} mA  "
                     f"(Run {n0['run']:.2%}, Sleep {n0['sleep']:.1%}, Stop {n0['stop']:.1%})", loc="left", fontsize=9)
    z0 = 30.300
    m = (t >= z0) & (t < z0 + 0.012)
    axs[1].fill_between((t[m] - z0) * 1e3, 0, i[m], step="post", color=BLUE, alpha=0.35, lw=0)
    axs[1].plot((t[m] - z0) * 1e3, i[m], color=BLUE, lw=0.9, drawstyle="steps-post")
    axs[1].set_title("A0, zoom on 12 ms: the core wakes every millisecond for the SysTick, so it can never enter Stop 1",
                     loc="left", fontsize=9)
    axs[1].set_xlabel("ms")
    t4, i4 = SC.current_trace(rec4["arch"], rec4["nom"]["cpu"], rec4["nom"]["per"], t0, t1)
    axs[2].fill_between(t4, 0, i4, step="post", color=BLUE, alpha=0.35, lw=0)
    axs[2].plot(t4, i4, color=BLUE, lw=0.8, drawstyle="steps-post")
    axs[2].set_title(f"A4  optimised architecture, 1.2 s, raw   -   average {n4['avg_ma']:.2f} mA  "
                     f"(Run {n4['run']:.2%}, Sleep {n4['sleep']:.1%}, Stop {n4['stop']:.1%})", loc="left", fontsize=9)
    axs[2].set_xlabel("time (s): simulated current at the MCU supply pin, as a power profiler would record it")
    for ax in axs:
        ax.set_ylabel("MCU current (mA)")
        ax.set_ylim(0, 22)
    fig.tight_layout()
    return _png(fig)


def fig_arch(archs):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    names = [r["arch"].name.split("  ")[0] for r in archs]
    nom = np.array([r["nom"]["avg_ma"] for r in archs])
    lo = np.array([r["low"]["avg_ma"] for r in archs])
    hi = np.array([r["high"]["avg_ma"] for r in archs])
    axs[0].bar(names, nom, color=BLUE, width=0.6, yerr=[nom - lo, hi - nom], capsize=3, ecolor=INK2)
    for i, v in enumerate(nom):
        axs[0].text(i, hi[i] + 0.3, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    axs[0].set(title="Average MCU current (bars: nominal; whiskers: low-high assumptions)", ylabel="mA")
    axs[0].title.set_fontsize(9)
    run = np.array([r["nom"]["run"] for r in archs]) * 100
    sl = np.array([r["nom"]["sleep"] for r in archs]) * 100
    st = np.array([r["nom"]["stop"] for r in archs]) * 100
    axs[1].bar(names, st, color=AQUA, label="Stop 1", width=0.6)
    axs[1].bar(names, sl, bottom=st, color=BLUE, label="Sleep", width=0.6)
    axs[1].bar(names, run, bottom=st + sl, color=ORANGE, label="Run", width=0.6)
    for i in range(len(names)):
        axs[1].text(i, 50, f"Run {run[i]:.2f}%\nSleep {sl[i]:.0f}%\nStop {st[i]:.0f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if st[i] > 20 or sl[i] > 50 else INK)
    axs[1].set(title="Share of time in each power state", ylabel="%", ylim=(0, 100))
    axs[1].title.set_fontsize(9)
    axs[1].legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    return _png(fig)


def fig_compute(variants):
    keys = list(variants)
    labels = [f"{k}\n{variants[k]['label']}" for k in keys]
    st = {"1 Hz tasks (x30)": [], "Feature extraction": [], "MLP": [], "Window checks + gate": []}
    lo, hi = [], []
    for k in keys:
        c = variants[k]["cycles"]["nom"]
        st["1 Hz tasks (x30)"].append(30 * (c["sample"] + c["uart_frame"]))
        st["Feature extraction"].append(c["features"])
        st["MLP"].append(c["mlp"])
        st["Window checks + gate"].append(c["wcheck"] + c["post"])
        lo.append(variants[k]["hop_cycles"]["low"])
        hi.append(variants[k]["hop_cycles"]["high"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bottom = np.zeros(len(keys))
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    for (name, vals), c in zip(st.items(), cols):
        ax.bar(range(len(keys)), vals, bottom=bottom, color=c, label=name, width=0.62)
        bottom += np.array(vals)
    for i, k in enumerate(keys):
        us = bottom[i] / 170.0
        ax.errorbar(i, bottom[i], yerr=[[bottom[i] - lo[i]], [hi[i] - bottom[i]]], color=INK2, capsize=4, lw=1)
        ax.text(i, hi[i] + 2500, f"{bottom[i] / 1e3:.0f} k cycles\n{us:.0f} µs", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(keys)), labels, fontsize=8)
    ax.set_ylabel("Cortex-M4F cycles per 30 s hop")
    ax.set_ylim(0, max(hi) * 1.18)
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.set_title("Computation per 30 s hop at 170 MHz (whiskers: low-high cost assumptions)", loc="left", fontsize=9)
    fig.tight_layout()
    return _png(fig)


def fig_pareto(df, deployed="32-16"):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, xcol, xlabel in ((axs[0], "cycles", "cycles per inference (log)"), (axs[1], "flash", "flash for the model, bytes (log)")):
        for fam, g in df.groupby("family"):
            ax.errorbar(g[xcol], g["f1"], yerr=[g["f1"] - g["f1_lo"], g["f1_hi"] - g["f1"]], fmt="o", color=FAMILY[fam],
                        ms=6, capsize=2, lw=1, alpha=0.9, label=fam)
        d = df[(df["family"] == "MLP int8") & (df["config"] == deployed)]
        ax.scatter(d[xcol], d["f1"], s=170, facecolors="none", edgecolors=INK, lw=1.6, zorder=5)
        ax.annotate("deployed MLP 25-32-16-5", (d[xcol].iloc[0], d["f1"].iloc[0]), xytext=(8, -22), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
    axs[0].set_ylabel("macro-F1 on held-out field sessions\n(bars: 95 % interval over sessions)")
    axs[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_memory(mem):
    fig, axs = plt.subplots(2, 1, figsize=(11, 4.6))
    pal = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    for ax, key, cap, unit in ((axs[0], "flash", mem["flash_cap"], "KB"), (axs[1], "ram", mem["ram_cap"], "KB")):
        left = 0.0
        for (name, v), c in zip(mem[key].items(), pal):
            ax.barh(0, v[1] / 1024, left=left, color=c, height=0.5, label=name)
            left += v[1] / 1024
        lo, nom, hi = (mem[f"{key}_total"][i] / 1024 for i in range(3))
        ax.errorbar(nom, 0, xerr=[[nom - lo], [hi - nom]], color=INK, capsize=5, lw=1.4, zorder=5)
        ax.axvline(cap / 1024, color=INK2, lw=1.5, ls="--")
        ax.text(cap / 1024, 0.42, f"capacity {cap / 1024:.0f} KB  ", color=INK2, fontsize=8, va="bottom", ha="right")
        ax.set_xlim(0, cap / 1024 * 1.02)
        ax.set_yticks([])
        ax.set_title(f"{'Flash' if key == 'flash' else 'SRAM'}: {nom:.0f} KB nominal ({lo:.0f}-{hi:.0f} KB range), "
                     f"{nom / (cap / 1024):.0%} of capacity", loc="left", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def fig_levers(levers):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    comps = list(levers[0][1]["parts"].keys())
    comps = [c for c in comps if c.startswith(("PMS", "BME", "STM32", "SD", "Reg", "NUCLEO"))]
    names_all = []
    for _, b in levers:
        for k in b["parts"]:
            k2 = "GPS" if k.startswith("GPS") else k
            if k2 not in names_all:
                names_all.append(k2)
    pal = dict(zip(names_all, [ORANGE, BLUE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]))
    y = np.arange(len(levers))[::-1]
    for yi, (name, b) in zip(y, levers):
        left = 0.0
        for k, v in b["parts"].items():
            k2 = "GPS" if k.startswith("GPS") else k
            ax.barh(yi, v, left=left, color=pal[k2], height=0.6, lw=0)
            left += v
        ax.text(left + 4, yi, f"{b['total_ma']:.0f} mA  =  {b['hours']:.0f} h  ({b['hours'] / 24:.1f} days)", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y, [textwrap(n, 46) for n, _ in levers], fontsize=8)
    ax.set_xlabel("average current drawn from the 2000 mAh battery (mA)")
    ax.set_xlim(0, max(b["total_ma"] for _, b in levers) * 1.45)
    ax.legend(handles=[Patch(color=c, label=n) for n, c in pal.items()], fontsize=7.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _png(fig)


def textwrap(s, w):
    import textwrap as tw
    return "\n".join(tw.wrap(s, w))


def fig_duty(test):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, site in zip(axs, ("busy", "quiet")):
        pol = test[site][0]
        names = list(pol)
        x = np.arange(len(names))
        w = 0.36
        ax.bar(x - w / 2, [pol[n]["held_acc"] for n in names], w, color=BLUE, label="answer shown to the user is right (last answer held <= 10 min)")
        ax.bar(x + w / 2, [pol[n]["ep_recall"] for n in names], w, color=ORANGE, label="source episodes (>= 90 s) detected")
        for i, n in enumerate(names):
            ax.text(i, 1.03, f"fan on {pol[n]['duty']:.0%}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title("Busy site (sources about 70 % of the time)" if site == "busy" else "Quiet site (two episodes per day)",
                     loc="left", fontsize=9)
    axs[0].set_ylabel("fraction")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=2, loc="lower center", frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _png(fig)


def fig_freq(sweep):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    f = [s["f"] for s in sweep]
    ax.plot(f, [s["avg_ma"] for s in sweep], "-o", color=BLUE)
    for s in sweep:
        ax.text(s["f"], s["avg_ma"] + 0.012, f"{s['avg_ma']:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("CPU clock (MHz)")
    ax.set_ylabel("average MCU current (mA)")
    ax.set_xscale("log")
    ax.set_xticks(f, [str(v) for v in f])
    ax.set_title("Race to idle: lower clock saves little", loc="left", fontsize=9)
    fig.tight_layout()
    return _png(fig)
