"""Figures and the HTML report for the virtual test campaign."""
import base64
import html
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from . import params as P

# Reference categorical palette (validated, fixed order). Classes are mapped once and never re-used.
CLR = {"vehicle_exhaust": "#2a78d6", "biomass_burning": "#eb6834", "background": "#1baf7a",
       "dust": "#eda100", "industrial": "#4a3aa7", P.UNCERTAIN: "#b9b8b2", P.FAULT: "#52514e",
       P.WARMUP: "#e3e2dd", "mixed": "#d0cfca", "confounder": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
NICE = {c: c.replace("_", " ") for c in CLR}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "lines.linewidth": 2})


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_fingerprints(df):
    """Mean size-fraction and heater-profile fingerprints per class (chamber windows)."""
    d = df[df["truth"].isin(P.CLASSES) & (df["status"] == "ok")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    bins = ["0.3-0.5", "0.5-1", "1-2.5", "2.5-5", "5-10", ">10"]
    for c in P.CLASSES:
        F = np.stack(d[d["truth"] == c]["feat"])
        ax[0].plot(bins, F[:, 0:6].mean(0), "-o", ms=4, color=CLR[c], label=NICE[c])
        rg = F[:, 13:19] + F[:, 12:13]
        ax[1].plot(P.HEATER_GROUPS, rg.mean(0), "-o", ms=4, color=CLR[c], label=NICE[c])
    ax[0].set(title="PMS7003 size fingerprint", xlabel="size bin (um)", ylabel="log10 fraction of counts")
    ax[1].set(title="BME688 heater-profile fingerprint", xlabel="heater step temperature (C)",
              ylabel="ln(R / clean-air baseline)")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].legend(fontsize=8, loc="lower left")
    return _png(fig)


def fig_confusions(dB, dA):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    labs = P.CLASSES + [P.UNCERTAIN]
    for ax, d, t in ((axs[0], dA, "Chamber-only training"), (axs[1], dB, "Chamber + field training (deployed)")):
        d = d[d["truth"].isin(P.CLASSES) & (d["status"] == "ok")]
        cm = confusion_matrix(d["truth"], d["pred"].astype(str), labels=labs)[:5]
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(6):
                ax.text(j, i, f"{cmn[i, j]:.0%}" if cm[i, j] else "", ha="center", va="center", fontsize=7.5,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(6), [NICE.get(l, l).replace(" ", "\n") for l in labs], fontsize=7)
        ax.set_yticks(range(5), [NICE[l] for l in P.CLASSES], fontsize=7.5)
        ax.set(title=t, xlabel="device output", ylabel="true dominant source")
        ax.grid(False)
    return _png(fig)


def fig_robustness(R):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    t = R["T03"]["data"]["table"]
    axs[0].plot(list(t), [v["acc"] for v in t.values()], "-o", color=CLR["vehicle_exhaust"], ms=5)
    axs[0].set(title="Humidity", xlabel="RH (%)", ylabel="accuracy", ylim=(0, 1))
    t = R["T04"]["data"]["table"]
    axs[1].plot(list(t), [v["acc"] for v in t.values()], "-o", color=CLR["vehicle_exhaust"], ms=5)
    axs[1].set(title="Temperature", xlabel="ambient temperature (C)", ylim=(0, 1))
    dr = R["T08"]["data"]
    for k, c, ls in (("tracking", CLR["vehicle_exhaust"], "-"), ("frozen", CLR["biomass_burning"], "--")):
        axs[2].plot(list(dr[k]), list(dr[k].values()), ls, marker="o", ms=5, color=c,
                    label="baseline tracking" if k == "tracking" else "frozen factory baseline")
    axs[2].set(title="Sensor ageing", xlabel="device age (days)", ylabel="macro-F1", ylim=(0, 1))
    axs[2].legend(fontsize=7.5, loc="lower left")
    acc = R["T05"]["data"]["acc"]
    for enc, c in zip(acc, (CLR["biomass_burning"], CLR["vehicle_exhaust"], CLR["background"])):
        axs[3].plot(list(acc[enc]), list(acc[enc].values()), "-o", ms=5, color=c, label=enc.replace("_", "-"))
    axs[3].set(title="Wind x inlet design", xlabel="wind speed (m/s)", ylim=(0, 1))
    axs[3].legend(fontsize=7.5, loc="lower left")
    return _png(fig)


def fig_detection(R):
    tab = R["T02"]["data"]["table"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for c in tab.index:
        ax.plot(tab.columns, tab.loc[c].values, "-o", ms=5, color=CLR[c], label=NICE[c])
    ax.axhline(0.8, color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set(title="Detection vs source strength", xlabel="excess PM2.5 over 30 ug/m3 background (ug/m3)",
           ylabel="correctly attributed", ylim=(0, 1.02))
    ax.legend(fontsize=7.5, loc="lower right")
    return _png(fig)


def fig_day(df, name):
    d = df[df["scenario"] == name].sort_values("t_end")
    h = 7 + d["t_end"].to_numpy() / 3600
    fig, ax = plt.subplots(2, 1, figsize=(12, 3.8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    keys = ["background", "vehicle_exhaust", "biomass_burning", "dust", "industrial", "confounder"]
    stack = np.array([[r.get(k, 0) * t for k in keys] for r, t in zip(d["shares"], d["total"])]).T
    ax[0].stackplot(h, stack, colors=[CLR[k] for k in keys], labels=[NICE[k] for k in keys], lw=0)
    ax[0].set(title=name, ylabel="true PM2.5 by source (ug/m3)")
    ax[0].legend(fontsize=7, ncol=6, loc="upper left")
    for x, lab in zip(h, d["pred"]):
        ax[1].axvspan(x - 1 / 240, x + 1 / 240, color=CLR.get(lab, "#ffffff"), lw=0)
    ax[1].set_yticks([])
    ax[1].set(ylabel="device", xlabel="hour of day")
    ax[1].grid(False)
    return _png(fig)


def fig_bars(R):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.2))
    f = R["T15"]["data"]
    names = ["PM only", "gas only", "fused (MLP)", "fused (RF)"]
    vals = [f["B_pm_only"], f["B_gas_only"], f["B_fused"], f["B_rf"]]
    axs[0].bar(names, vals, color=[CLR["dust"], CLR["industrial"], CLR["vehicle_exhaust"], CLR["background"]], width=0.6)
    for i, v in enumerate(vals):
        axs[0].text(i, v + 0.02, f"{v:.2f}", ha="center", color=INK, fontsize=8)
    axs[0].set(title="Does fusion help? (held-out field data)", ylabel="macro-F1", ylim=(0, 1.08))
    pu = R["T09"]["data"]["per_unit"]
    axs[1].bar([str(k) for k in pu], list(pu.values()), color=CLR["vehicle_exhaust"], width=0.6)
    axs[1].set(title="10 pilot units, same scenarios", xlabel="unit id", ylabel="macro-F1", ylim=(0, 1.05))
    pm = R["T14"]["data"]["modes"]
    labels = [k.replace(", ", "\n") for k in pm]
    hrs = [v["hours"] for v in pm.values()]
    axs[2].barh(labels, hrs, color=CLR["background"], height=0.55)
    axs[2].axvline(24, color=INK2, lw=0.8, ls=":")
    for i, v in enumerate(hrs):
        axs[2].text(v + 0.5, i, f"{v:.1f} h", va="center", fontsize=8, color=INK)
    axs[2].set(title="Battery life, 2000 mAh", xlabel="hours")
    axs[2].tick_params(axis="y", labelsize=7.5)
    return _png(fig)


# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#f7f7f5;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e3e2dd;--pass:#0a7a3e;--fail:#b3261e;--info:#2a78d6;--warnbg:#fff6e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1f1f1d;--ink:#f4f3ee;--ink2:#c3c2b7;--line:#34332f;--pass:#4cc38a;--fail:#ff8a80;--info:#6aa8f0;--warnbg:#3a3220}}
:root[data-theme="dark"]{--bg:#141413;--card:#1f1f1d;--ink:#f4f3ee;--ink2:#c3c2b7;--line:#34332f;--pass:#4cc38a;--fail:#ff8a80;--info:#6aa8f0;--warnbg:#3a3220}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 80px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:40px 0 12px}
h3{font-size:16px;margin:0 0 6px}p{margin:0 0 10px}.sub{color:var(--ink2)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:12px 0}
.warn{background:var(--warnbg)}table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--ink2);font-weight:600}
.v{font-weight:700;font-size:12px;letter-spacing:.04em;padding:2px 8px;border-radius:99px;border:1.5px solid currentColor;white-space:nowrap}
.PASS{color:var(--pass)}.FAIL{color:var(--fail)}.INFO{color:var(--info)}
img{max-width:100%;height:auto;border-radius:6px;background:#fff;display:block;margin:8px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}.kpi b{display:block;font-size:26px}
.kpi span{color:var(--ink2);font-size:13px}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:13px}
.flow div{border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:var(--card)}
.tablewrap{overflow-x:auto}code{font-size:13px}li{margin-bottom:6px}
"""


def build_html(R, figs, notes, assumptions, meta):
    e = html.escape
    n_pass = sum(r["verdict"] == "PASS" for r in R.values())
    n_fail = sum(r["verdict"] == "FAIL" for r in R.values())
    rows = "".join(f"<tr><td>{e(r['id'])}</td><td>{e(r['name'])}</td><td>{e(r['criterion'])}</td>"
                   f"<td><span class='v {r['verdict']}'>{r['verdict']}</span></td></tr>" for r in R.values())
    details = ""
    for r in R.values():
        details += (f"<div class='card'><h3>{e(r['id'])} - {e(r['name'])} <span class='v {r['verdict']}'>{r['verdict']}</span></h3>"
                    f"<p class='sub'>Criterion: {e(r['criterion'])}</p><p>{e(r['summary'])}</p>"
                    + "".join(f"<img alt='{e(cap)}' src='data:image/png;base64,{figs[k]}'>"
                              for k, cap in notes.get("figs_for", {}).get(r["id"], []) if k in figs) + "</div>")
    arows = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td><td>{e(c)}</td></tr>" for a, b, c in assumptions)
    kp = "".join(f"<div class='card kpi'><b>{e(v)}</b><span>{e(k)}</span></div>" for k, v in notes["kpis"])
    rec = "".join(f"<li>{x}</li>" for x in notes["recommendations"])
    fnd = "".join(f"<li>{x}</li>" for x in notes["findings"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PolSo Virtual Test Report</title><style>{CSS}</style></head><body><main>
<h1>PolSo virtual test campaign</h1>
<p class="sub">Digital twin of the PM + gas source-attribution device (STM32 NUCLEO-G474RE, PMS7003, BME688,
hooded PolSo-19 enclosure, 2000 mAh Li-ion), run before buying hardware. Generated {e(meta['date'])};
{e(meta['sim_hours'])} simulated sensor-hours, {e(meta['runtime'])} on this PC.</p>
<div class="card warn"><b>What this is and is not.</b> Every number below comes from physics-based models whose parameters are
literature/datasheet estimates (listed under Assumptions), not from PolSo hardware. The simulation is good at finding
<i>design problems</i> (which conditions break the classifier, what the firmware must handle, whether the battery is big enough)
and at comparing options. It cannot prove real-world accuracy: training and test data come from the same simulator, so real
accuracy will be lower. Re-fit the assumptions with bench data from the first unit (see the last section).</div>
<div class="kpis">{kp}</div>
<h2>System modelled</h2>
<div class="flow"><div>Weather + background + time-varying sources</div>&rarr;<div>Inlet losses (PolSo-19)</div>&rarr;
<div>PMS7003 optics + 32-byte UART frames</div><div>BME688 MOX + 10-step heater scan</div>&rarr;
<div>Firmware: parser, baseline tracker, 25 features</div>&rarr;<div>int8 MLP 25-32-16-5 + confidence gate</div>&rarr;
<div>Label / uncertain / fault + SD log</div></div>
<h2>Results at a glance &mdash; {n_pass} pass, {n_fail} fail</h2>
<div class="card tablewrap"><table><tr><th>ID</th><th>Test</th><th>Acceptance criterion</th><th>Result</th></tr>{rows}</table></div>
<h2>Key findings</h2><div class="card"><ol>{fnd}</ol></div>
<h2>What to change before / while buying hardware</h2><div class="card"><ol>{rec}</ol></div>
<h2>How the sources look to the sensors</h2>
<div class="card"><img alt="fingerprints" src="data:image/png;base64,{figs['fingerprints']}">
<p class="sub">Mean features from chamber tests on the 10 unseen pilot units. Dust shows up in the coarse bins; combustion sources
separate mainly through the gas heater profile (NO2 from diesel raises resistance at 200 C, VOCs lower it at 250-300 C).</p></div>
<h2>Test details</h2>{details}
<h2>Example field days</h2>
<div class="card">{''.join(f"<img alt='field day' src='data:image/png;base64,{figs[k]}'>" for k in figs if k.startswith('day_'))}
<p class="sub">Top: true PM2.5 contribution of each source (window averages). Bottom strip: what the device reported every 30 s
(grey = uncertain, light grey = warming up).</p></div>
<h2>Assumptions (re-fit these with real data)</h2>
<div class="card tablewrap"><table><tr><th>Item</th><th>Value used</th><th>Basis / how to measure</th></tr>{arows}</table></div>
<h2>Reproduce</h2><div class="card"><p><code>cd simulation</code> &rarr; <code>python run_all.py</code> (full, ~{e(meta['runtime'])})
or <code>python run_all.py --quick</code>. Unit tests: <code>python -m unittest discover tests</code>. Firmware core:
<code>firmware_c/</code> (portable C99, drop into STM32CubeIDE; <code>model_params.h</code> is generated).</p></div>
</main></body></html>"""
