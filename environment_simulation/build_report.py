"""
Build the combined report:  python build_report.py
  needs results/mcu_results.pkl (run_mcu.py) and results/verification_results.pkl (run_verification.py)
  writes results/stm32_and_verification_report.html (standalone) and results/stm32_report_artifact.html (fragment)
"""
import datetime
import html
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from polso_twin import params as P                                     # noqa: E402
from polso_twin.report import CSS as BASE_CSS                          # noqa: E402
from stm32_twin import figures as F, spec, platform_map                 # noqa: E402
from verify import figures as VF, metrics as MT                         # noqa: E402

RES = os.path.join(ROOT, "results")
e = html.escape

EXTRA_CSS = """
nav.toc{display:flex;flex-wrap:wrap;gap:8px 16px;margin:14px 0 4px;font-size:14px}
nav.toc a{color:var(--info);text-decoration:none;border-bottom:1px solid transparent}
nav.toc a:hover{border-bottom-color:currentColor}
h2{scroll-margin-top:12px}h3{margin:22px 0 6px;font-size:16px}
.fig{margin:10px 0 4px}.cap{color:var(--ink2);font-size:13px;margin:0 0 14px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.two>div{min-width:0}
table.t{font-size:13px}table.t td,table.t th{padding:6px 8px}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;padding:1px 7px;border-radius:99px;border:1.5px solid currentColor;white-space:nowrap}
.sup{color:var(--pass)}.need{color:#b36b00}.no{color:var(--fail)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .need{color:#f0b35a}}
:root[data-theme="dark"] .need{color:#f0b35a}
.kpis b{font-size:24px}ul.tight li{margin-bottom:4px}
code,pre{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto}
.scroll{overflow-x:auto}
"""


class Raw(str):
    pass


def cell(x):
    return x if isinstance(x, Raw) else e(str(x))


def tbl(headers, rows, num_cols=(), cls="t"):
    th = "".join(f"<th class='{'n' if i in num_cols else ''}'>{e(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td class='{'n' if i in num_cols else ''}'>{cell(c)}</td>" for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<div class='scroll'><table class='{cls}'><tr>{th}</tr>{body}</table></div>"


def img(b64, alt, cap=None):
    s = f"<img class='fig' alt='{e(alt)}' src='data:image/png;base64,{b64}'>"
    return s + (f"<p class='cap'>{cap}</p>" if cap else "")


def card(inner, cls=""):
    return f"<div class='card {cls}'>{inner}</div>"


def status(kind):
    t = {"sup": ("SUPPORTED IN SIMULATION", "sup"), "need": ("NEEDS HARDWARE", "need"), "no": ("NOT SUPPORTED", "no")}[kind]
    return Raw(f"<span class='tag {t[1]}'>{t[0]}</span>")


def campaign_counts():
    """PASS counts of the 16-test campaign before and after the PMS7003 sample-heating fix."""
    import json
    out = {}
    for tag, path in (("before", os.path.join(RES, "previous_run", "summary_before_pms_heating_fix.json")),
                      ("after", os.path.join(RES, "summary.json"))):
        try:
            r = json.load(open(path))["results"]
            out[tag] = (sum(v["verdict"] == "PASS" for v in r.values()), len(r))
        except Exception:
            out[tag] = None
    return out


def campaign_t11():
    import json
    try:
        o = json.load(open(os.path.join(RES, "summary.json")))["results"]["T11"]["data"]["overall"]
        return dict(sel_acc=o["sel_acc"], coverage=o["coverage"])
    except Exception:
        return None


def num(x, d=0):
    return f"{x:,.{d}f}"


# ---------------------------------------------------------------------------------------------
def build(mcu, ver):
    V, A = mcu["variants"], mcu["archs"]
    v0, v3, v4 = V["v0"], V["v3"], V["v4"]
    a0, a1, a2, a3, a4 = (r["nom"] for r in A)
    hop0, hop3, hop4 = v0["hop_cycles"]["nom"], v3["hop_cycles"]["nom"], v4["hop_cycles"]["nom"]
    us0 = hop0 / spec.F_MAX_MHZ
    uj0 = us0 * 1e-6 * spec.i_run_ma(spec.F_MAX_MHZ) * 1e-3 * spec.V_RAIL * 1e6
    mem = mcu["memory"]
    levers = mcu["levers"]
    best = levers[-1][1]
    df = mcu["pareto"]
    duty = mcu["duty"]["test"]
    figs = {}

    # figures ------------------------------------------------------------------------------
    figs["timeline"] = F.fig_timeline(A[0], A[-1])
    figs["zoom"] = F.fig_zoom(A[-1])
    figs["current"] = F.fig_current(A[0], A[-1])
    figs["arch"] = F.fig_arch(A)
    figs["freq"] = F.fig_freq(mcu["freq_sweep"])
    figs["memory"] = F.fig_memory(mem)
    figs["compute"] = F.fig_compute(V)
    figs["pareto"] = F.fig_pareto(df)
    figs["levers"] = F.fig_levers(levers)
    figs["duty"] = F.fig_duty(duty)
    figs["pm"] = VF.fig_pm_scatter(ver)
    figs["bias"] = VF.fig_bias_by_source(ver["win"])
    figs["conf"] = VF.fig_confusion(ver)
    figs["calib"] = VF.fig_calibration(ver)
    figs["share"] = VF.fig_share(ver)
    figs["samples"] = VF.fig_samples(ver)

    # derived facts ----------------------------------------------------------------------------
    top = df.loc[df["f1"].idxmax()]
    biggest_cpu = df["cycles"].max() / (spec.F_MAX_MHZ * 1e6 * 30)
    deployed = df[(df["family"] == "MLP int8") & (df["config"] == "32-16")].iloc[0]
    lin = df[df["family"] == "Logistic"].iloc[0]
    abl = mcu["ablation"]
    mlp_share = best["parts"]["STM32 (MCU)"] / best["total_ma"]
    l0 = levers[0][1]
    pms_share0 = l0["parts"]["PMS7003 fan + laser (via 5 V boost)"] / l0["total_ma"]
    eq = {k: V[k]["equivalence"] for k in V}
    rings0 = A[0]["nom"]["rings"]
    pc = ver["pm"]
    raw, cor, src = pc["raw"]["pooled"], pc["corrected"]["pooled"], pc["source-aware"]["pooled"]
    n_ok = sum(c["passed"] for c in ver["criteria"])

    kpis = [
        (f"{us0:.0f} µs", f"CPU time per 30 s hop, reference firmware ({hop0 / 1e3:.0f} k cycles at 170 MHz)"),
        (f"{a4['run']:.3%}", "of the time the core is running (optimised architecture)"),
        (f"{a0['avg_ma']:.1f} → {a4['avg_ma']:.1f} mA", "average MCU current: typical starter firmware → optimised architecture"),
        (f"{mem['ram_total'][1] / 1024:.0f} KB / {mem['flash_total'][1] / 1024:.0f} KB", f"SRAM / flash used, of {spec.SRAM_KB} / {spec.FLASH_KB} KB"),
        (f"{best['hours'] / 24:.1f} days", f"battery life on 2000 mAh with every lever ({levers[0][1]['hours']:.0f} h as specified)"),
    ]

    findings = [
        f"<b>Compute is not the constraint.</b> One 30 s hop of the whole pipeline (parsing, compensation, 25 features, int8 MLP, "
        f"confidence gate) costs about {hop0 / 1e3:.0f} k cycles, {us0:.0f} µs and {uj0:.0f} µJ at 170 MHz. That is "
        f"{hop0 / (spec.F_MAX_MHZ * 1e6 * 30):.4%} of the CPU. Even with the pessimistic cost assumptions it stays below "
        f"{v0['hop_cycles']['high'] / (spec.F_MAX_MHZ * 1e6 * 30):.3%}.",
        f"<b>What decides MCU energy is the firmware architecture, not the algorithm.</b> The same pipeline draws {a0['avg_ma']:.1f} mA "
        f"with a typical HAL starter project (1 ms SysTick, blocking SD writes, 8-sentence GPS) and {a4['avg_ma']:.2f} mA with tickless "
        f"Stop 1 idle, DMA-driven SD, an RMC-only GPS and register-level drivers. Optimising the computation on top of that "
        f"({a3['avg_ma']:.2f} → {a4['avg_ma']:.2f} mA) changes almost nothing.",
        f"<b>The MCU is a small part of the battery budget.</b> In the as-specified design the PMS7003 fan and laser take "
        f"{pms_share0:.0%} of the battery current; the MCU in the best design is {mlp_share:.0%}. Duty-cycling the PMS7003, dropping the "
        f"Nucleo board and slowing the GPS take battery life from {levers[0][1]['hours']:.0f} h to {best['hours']:.0f} h "
        f"({best['hours'] / 24:.1f} days).",
        f"<b>Accuracy rises only slowly with model size, so choose the model by accuracy, not by cost.</b> On held-out simulated field sessions macro-F1 runs from "
        f"{lin['f1']:.2f} (logistic regression, {lin['cycles']:,.0f} cycles) to {top['f1']:.2f} ({top['family']} {top['config']}, {top['cycles']:,.0f} cycles, "
        f"{top['us']:.0f} µs); the deployed MLP 25-32-16-5 scores {deployed['f1']:.2f}. The intervals overlap, so small rankings are not supported. What is supported: every "
        f"candidate uses under {biggest_cpu:.3%} of the CPU, so the compute budget never forces a trade-off with accuracy.",
        f"<b>Duty-cycling the PMS7003 is cheap in accuracy; waking it from the gas signal does not work.</b> In a busy site a 1/3 duty "
        f"cycle keeps {duty['busy'][0]['periodic 1/3']['held_acc']:.0%} of the answers right against "
        f"{duty['busy'][0]['continuous']['held_acc']:.0%} with the fan always on; in a quiet site 1/6 duty keeps "
        f"{duty['quiet'][0]['periodic 1/6']['held_acc']:.0%} against {duty['quiet'][0]['continuous']['held_acc']:.0%}. "
        f"The gas-triggered policy reached only {duty['quiet'][0]['adaptive']['held_acc']:.0%} in the quiet site because dust has no gas signature.",
        f"<b>PM2.5 mass does not meet the EPA targets in the rehearsal, and 12 days is too few to fit more than a humidity correction.</b> The raw device reads {raw['slope']:.1f}x the reference "
        f"(R² {raw['r2']:.2f}), because low-cost counters read different sources differently. A humidity correction, scored by cross-validation over days, brings the slope to "
        f"{cor['slope']:.2f} (R² {cor['r2']:.2f}, RMSE {cor['rmse']:.0f} µg/m³ over hours up to {ver['pm_hourly']['ref'].max():.0f} µg/m³). "
        + (f"A correction in which the device's own source label chooses the gain did better (R² {src['r2']:.2f})."
           if src['r2'] > cor['r2'] else
           f"A correction in which the device's own source label chooses the gain did worse (R² {src['r2']:.2f}): six parameters fitted from a short, heavy-tailed record do not generalise.")
        + " Plan at least 30 days of co-location and always report the cross-validated result.",
        f"<b>The verification rehearsal met {n_ok} of {len(ver['criteria'])} proposed criteria.</b> That is a statement about the simulator "
        f"and the metrics, not about the hardware, but it shows the pipeline, file formats and sample sizes work before any data exists.",
    ]

    # ---------------------------------------------------------------------------------------------
    S = []
    S.append(f"""<h1>PolSo on the STM32G474</h1>
<p class="sub">Running parameters, computation cost and how to verify it. A virtual NUCLEO-G474RE (Cortex-M4F, 170 MHz, 128 KB SRAM, 512 KB flash)
running the real firmware core, plus a rehearsed verification plan. Generated {datetime.date.today().isoformat()}.</p>
<nav class="toc"><a href="#summary">Summary</a><a href="#running">Running parameters</a><a href="#compute">Computation</a><a href="#energy">Energy</a>
<a href="#reference">What to verify against</a><a href="#rehearsal">Rehearsal</a><a href="#changes">Changes</a><a href="#measure">On the board</a></nav>
<div class="card warn"><b>What this is and is not.</b> No ARM compiler, emulator or board was available. The operation counts are exact for your firmware
source (the unchanged C code was built with every floating-point operation counted). The cycle costs per operation, the currents and the SD-card
behaviour are <b>estimates with low / nominal / high ranges</b> (<code>stm32_twin/spec.py</code>); conclusions are stated only where they hold across
the range. Benchmark firmware and a comparison tool are included so each estimate can be replaced by a measurement in the first week.</div>""")

    S.append("<h2 id='summary'>Summary</h2><div class='kpis'>" + "".join(card(f"<b>{e(a)}</b><span>{e(b)}</span>", "kpi") for a, b in kpis) + "</div>")
    S.append("<div class='card'><ol>" + "".join(f"<li>{f}</li>" for f in findings) + "</ol></div>")

    # ---- claims table ---------------------------------------------------------------------------
    claims = [
        ("The complete attribution pipeline runs on a Cortex-M4F in a few hundred microseconds per 30 s window using about "
         f"{mem['ram_total'][1] / 1024:.0f} KB of SRAM and {mem['flash_total'][1] / 1024:.0f} KB of flash", status("sup"),
         "Operation counts are exact; cycle costs and memory are ranges. Confirm with <code>bench/dwt_bench.c</code> and the linker map."),
        (f"Architecture-level co-design cuts MCU energy about {a0['avg_ma'] / a4['avg_ma']:.0f}x without changing the algorithm", status("sup"),
         "Needs a current measurement across the Nucleo IDD jumper; the Stop 1 and Run currents are the sensitive inputs."),
        ("Caching the log, averaging before decimating the humidity term and using a polynomial log / exp cut compute by "
         f"{(1 - hop3 / hop0):.0%} with the same labels", status("sup"),
         f"Log caching is bit-identical; the others change {eq['v3']['gate_flips']} of {eq['v3']['windows']} windows (an answer becoming “uncertain”) and none between two different sources."),
        ("Compute budget does not limit accuracy: even the largest candidate model needs a tiny fraction of the CPU", status("sup"),
         "Holds on simulated data. Which model is most accurate on real sensors is open: repeat the comparison on real data."),
        ("A source-aware PM2.5 correction improves mass accuracy", status("no"),
         "Did not beat the humidity-only correction with 12 days in the rehearsal, and the composition bias it targets is an assumption of the simulator. Revisit with 30+ days of real data."),
        ("The approach is novel compared with existing work", status("no"),
         "Prior art was not surveyed here. Check before claiming novelty; the comparison slide already names IIT Kanpur's DHSA as closest."),
    ]
    S.append("<h3>What you can claim, and on what evidence</h3>" + tbl(["Claim", "Status", "Evidence and what is missing"], [(Raw(c), s, Raw(n)) for c, s, n in claims]))

    # ---- 1. running parameters ----------------------------------------------------------------------
    S.append("<h2 id='running'>1. Running parameters on the STM32</h2>")
    S.append("<h3>Clocks and low-power plan</h3>" + tbl(["Item", "Setting"], platform_map.CLOCK_PLAN))
    S.append("<h3>Peripheral allocation (proposal)</h3>" + tbl(
        ["Function", "Peripheral", "Configuration", "Data path", "NVIC priority"],
        [(a, b, c, d, "–" if p is None else p) for a, b, c, d, p in platform_map.PERIPHERALS]) +
        f"<p class='cap'>{e(platform_map.IRQ_NOTE)} Confirm pin availability against the NUCLEO-G474RE user manual and CubeMX; "
        f"keep DMA buffers in SRAM1 (check whether the CCM SRAM is reachable by DMA in RM0440).</p>")
    S.append("<h3>Memory</h3>" + img(figs["memory"], "Flash and SRAM budget",
             "Nominal use by component; whiskers show the low-high range. The firmware core size comes from a host build scaled for Thumb-2; the rest are estimates to replace from the linker map."))
    S.append("<div class='two'><div>" + tbl(["Flash component", "KB (nominal)"], [(k, f"{v[1] / 1024:.1f}") for k, v in mem["flash"].items()] +
             [(Raw("<b>Total</b>"), Raw(f"<b>{mem['flash_total'][1] / 1024:.0f}</b> ({mem['flash_total'][0] / 1024:.0f}-{mem['flash_total'][2] / 1024:.0f})"))], num_cols=(1,)) +
             "</div><div>" + tbl(["SRAM component", "KB (nominal)"], [(k, f"{v[1] / 1024:.2f}") for k, v in mem["ram"].items()] +
             [(Raw("<b>Total</b>"), Raw(f"<b>{mem['ram_total'][1] / 1024:.1f}</b> ({mem['ram_total'][0] / 1024:.1f}-{mem['ram_total'][2] / 1024:.1f})"))], num_cols=(1,)) + "</div></div>")
    S.append("<h3>What the CPU does, and how often</h3>" + tbl(
        ["Task", "Rate", "CPU time (µs), optimised architecture"], [(n, r, f"{t:,.1f}") for n, r, t in mcu["tasks"]], num_cols=(2,)) +
        f"<p class='cap'>The window task is {mcu['tasks'][5][2]:,.0f} µs every 30 s. The SD write is the only slow CPU step ({mcu['tasks'][6][2]:,.0f} µs of CPU) and its "
        f"card-side wait, which can reach a second on a bad card, is why it must not block the main loop.</p>")
    S.append("<h3>A simulated logic-analyser view</h3>" + img(figs["timeline"], "Task timeline", "1.2 s around a 30 s window boundary. Blue: peripheral busy moving "
             "bytes (UART frames, I²C, SPI). Orange: CPU running. In A0 the SD write busy-waits with the CPU running for as long as the card takes (a few milliseconds here, hundreds on a slow write); in A4 it is DMA and the CPU sleeps.") +
             img(figs["zoom"], "Zoom on the window task", "The 30 s window task itself: feature extraction and the int8 MLP take well under a millisecond."))
    S.append("<h3>Simulated current at the MCU supply pin</h3>" + img(figs["current"], "Current trace",
             "What a power profiler would show for the same 1.2 s. The starter firmware sits at the Sleep floor and wakes every millisecond (middle panel), so it never reaches Stop 1; the optimised one spends most of its time there."))
    S.append("<h3>Firmware architecture options</h3>" + img(figs["arch"], "Architecture comparison", None) + tbl(
        ["Architecture", "Avg mA", "Range", "Run", "Sleep", "Stop 1"],
        [(r["arch"].name, f"{r['nom']['avg_ma']:.2f}", f"{r['low']['avg_ma']:.2f}-{r['high']['avg_ma']:.2f}", f"{r['nom']['run']:.3%}",
          f"{r['nom']['sleep']:.1%}", f"{r['nom']['stop']:.1%}") for r in A], num_cols=(1, 2, 3, 4, 5)) +
        "<p class='cap'>Every step is a design choice, not an algorithm change: tickless idle (the 1 ms HAL tick forbids Stop), DMA-driven SD logging (no busy-wait), "
        "configuring the GPS to emit only the RMC sentence (the default 8 sentences keep the UART busy about half of every second), and register-level drivers.</p>")
    S.append("<div class='two'><div><h3>Race to idle</h3>" + img(figs["freq"], "Clock sweep", "Running slower saves little because the core is already idle more than 99.9 % of the time; "
             "the current floor is the peripherals moving data and the Stop 1 floor.") + "</div><div><h3>Interrupt and buffer risk</h3>" + tbl(
        ["Item", "Value (A0 starter firmware)"],
        [("SD write stall, 99th / 99.9th percentile / worst", f"{rings0['p99_stall_ms']:.0f} / {rings0['p999_stall_ms']:.0f} / {rings0['max_stall_ms']:.0f} ms"),
         ("GPS ring (256 B) overruns while the SD write blocks", f"{rings0['gps']['per_hour']:.3f} per hour"),
         ("PMS ring (128 B) overruns", f"{rings0['pms']['per_hour']:.3f} per hour"),
         ("Same, with DMA-driven SD", "0 (the main loop is never blocked)")]) +
        "<p class='cap'>The SD latency distribution is an estimate (garbage-collection tail, rare bad card); measure it on the card you will use.</p></div></div>")

    # ---- 2. computation -----------------------------------------------------------------------------
    S.append("<h2 id='compute'>2. Computation</h2>")
    S.append(f"<p>The unmodified firmware sources were built with a class that counts every floating-point add, multiply, divide, compare and conversion "
             f"and every libm call, on three hours of simulated sensor streams. The counting build produces output identical to the plain build "
             f"(maximum difference 0.0). Counts are then multiplied by Cortex-M4F cycle costs.</p>")
    pc0 = v0["counts_per_call"]
    S.append(tbl(["Stage", "Runs", "FP add / mul", "FP div", "log10f", "logf", "expf", "floorf", "Cycles (nominal)"],
                 [("1 Hz sample task", "every second", f"{pc0['sample']['add'] + pc0['sample']['mul']:.0f}", f"{pc0['sample']['div']:.0f}", "0", f"{pc0['sample']['log']:.0f}", f"{pc0['sample']['exp']:.0f}", "0", f"{v0['cycles']['nom']['sample']:,.0f}"),
                  ("Feature extraction", "every 30 s", f"{pc0['features']['add'] + pc0['features']['mul']:,.0f}", f"{pc0['features']['div']:.0f}", f"{pc0['features']['log10']:.0f}", "0", "0", "0", f"{v0['cycles']['nom']['features']:,.0f}"),
                  ("int8 MLP 25-32-16-5", "every 30 s", f"{pc0['mlp']['add'] + pc0['mlp']['mul']:.0f}", f"{pc0['mlp']['div']:.0f}", "0", "0", f"{pc0['mlp']['exp']:.0f}", f"{pc0['mlp']['floor']:.0f}", f"{v0['cycles']['nom']['mlp']:,.0f}")],
                 num_cols=(2, 3, 4, 5, 6, 7, 8)) +
             f"<p class='cap'>{pc0['features']['log10']:.0f} <code>log10f</code> calls per window dominate feature extraction: the same value is recomputed in several loops. "
             f"The 1,392 int8 multiply-accumulates of the MLP are added at 3.5 cycles each (scalar C).</p>")
    S.append("<h3>Optimisation ladder, each step checked against the reference firmware</h3>" + img(figs["compute"], "Cycles per hop by variant", None) + tbl(
        ["Variant", "Cycles / hop", "µs", "Labels identical", "Different source answered", "Answer ↔ uncertain flips", "Max prob. change"],
        [(f"{k} {V[k]['label']}", f"{V[k]['hop_cycles']['nom']:,.0f}", f"{V[k]['hop_cycles']['nom'] / 170:,.0f}",
          f"{eq[k]['agree']:.2%}", eq[k].get("hard_disagree", 0), eq[k].get("gate_flips", 0), f"{eq[k]['max_dprob']:.3f}") for k in V],
        num_cols=(1, 2, 3, 4, 5, 6)) +
        f"<p class='cap'>v1 is bit-identical. v2 and v3 change {eq['v3']['gate_flips']} of {eq['v3']['windows']} windows, an answer flipping to or from “uncertain”, and no window changes from one source "
        f"to another. The largest single-window probability change is {eq['v3']['max_dprob']:.2f}, caused by the numerically fragile feature discussed below. v4 uses SMLAD / CMSIS-NN style int8 dot products, which give identical integers; "
        "its cycle count is an estimate until measured. A first attempt at the humidity step (hold one noisy reading for 10 s) was rejected by exactly this "
        "check: it moved the gas features enough to flip windows, so the final version averages ten readings before evaluating the exponential once.</p>")
    S.append("<h3>Which model should run on the MCU?</h3>" + img(figs["pareto"], "Accuracy versus compute and flash", None) +
             tbl(["Model", "Macro-F1", "95 % interval", "Selective acc. @90 % cov.", "Cycles", "µs", "µJ", "Flash (KB)"],
                 [(f"{r.family} {r.config}", f"{r.f1:.3f}", f"{r.f1_lo:.3f}-{r.f1_hi:.3f}", f"{r.sel_acc90:.3f}", f"{r.cycles:,.0f}", f"{r.us:.1f}", f"{r.uJ:.2f}", f"{r.flash / 1024:.1f}")
                  for r in df.itertuples()], num_cols=(1, 2, 3, 4, 5, 6, 7)) +
             f"<p class='cap'>Trained on {mcu['pareto_meta']['n_train']:,} windows and scored on {mcu['pareto_meta']['n_test']:,} windows from unseen sessions and the 10 unseen pilot units. "
             "Intervals come from resampling whole sessions, so differences smaller than the interval are not real. All models fit easily: the largest forest "
             "uses under half the flash, and every candidate needs under 0.1 % of the CPU.</p>")
    d_mlp = abl["without pm_gas_corr"]["mlp"] - abl["all 25 features"]["mlp"]
    S.append("<h3>A numerically fragile feature</h3><p>Comparing windows after the humidity change showed that <code>pm_gas_corr</code> (the correlation between particle count and gas response, "
             "computed from about 12 points per window) swings by 0.3 to 0.4 for a 0.003 change in gas resistance. "
             f"Removing it changes the MLP from {abl['all 25 features']['mlp']:.3f} to {abl['without pm_gas_corr']['mlp']:.3f} and the forest from "
             f"{abl['all 25 features']['rf']:.3f} to {abl['without pm_gas_corr']['rf']:.3f}; also removing skew and variation gives "
             f"{abl['without pm_gas_corr + skew + cv']['mlp']:.3f} and {abl['without pm_gas_corr + skew + cv']['rf']:.3f}. "
             + ("It is not clearly harmful in the simulator, so no change is recommended, but features built from so few points deserve a second look on real, noisier data."
                if d_mlp < 0.01 else "Removing it helps, so consider dropping it or computing it over a longer span.") + "</p>")
    S.append(f"""<h3>Where the computation story is, and is not</h3>
<ul class="tight">
<li><b>Is:</b> a fixed-memory, no-malloc pipeline with bounded loops ({V['v3']['state_bytes'] / 1024:.1f} KB of state) whose worst-case time is known; it fits a small Cortex-M4F with
more than 99.9 % of the CPU idle; the architecture around it, not the arithmetic, sets the energy.</li>
<li><b>Is not:</b> an energy claim for the algorithm itself. The whole inference costs {uj0:.0f} µJ per 30 s, about {uj0 / 30:.1f} µW averaged, against hundreds of milliwatts for the sensors.</li>
<li><b>Headroom:</b> at 1 % CPU duty the budget is {0.01 * spec.F_MAX_MHZ * 1e6 * 30 / 1e6:.0f} M cycles per hop, about {0.01 * spec.F_MAX_MHZ * 1e6 * 30 / hop0:,.0f}x the current pipeline. That room could hold
an ensemble, on-device uncertainty estimates, a longer feature window, online baseline adaptation or self-calibration, which are the places to look for a computation contribution.</li>
</ul>""")

    # ---- 3. energy -------------------------------------------------------------------------------------
    S.append("<h2 id='energy'>3. Energy</h2>")
    S.append(img(figs["levers"], "Power budget levers", "Average current drawn from the battery. Each row adds one change to the row above.") + tbl(
        ["Design", "Total (mA)", "Battery life", "PMS7003", "MCU", "GPS", "Nucleo board"],
        [(n, f"{b['total_ma']:.0f}", f"{b['hours']:.0f} h ({b['hours'] / 24:.1f} d)",
          f"{b['parts']['PMS7003 fan + laser (via 5 V boost)']:.0f}", f"{b['parts']['STM32 (MCU)']:.2f}",
          f"{[v for k, v in b['parts'].items() if k.startswith('GPS')][0]:.0f}", f"{b['parts'].get('NUCLEO board overhead (ST-LINK etc.)', 0):.0f}") for n, b in levers],
        num_cols=(1, 2, 3, 4, 5, 6)))
    S.append("<h3>Duty-cycling the particle sensor</h3>" + img(figs["duty"], "Duty cycling policies", None) + tbl(
        ["Site", "Policy", "Fan on", "Answer shown is right", "Source episodes detected", "Median delay (s)"],
        [(("Busy" if s == "busy" else "Quiet"), p, f"{v['duty']:.0%}", f"{v['held_acc']:.0%}", f"{v['ep_recall']:.0%} ({v['episodes']} episodes)",
          f"{v['delay_med']:.0f}") for s in ("busy", "quiet") for p, v in duty[s][0].items()], num_cols=(2, 3, 4, 5)) +
        f"<p class='cap'>A burst is 90 s (30 s fan warm-up plus a full 60 s window). “Answer shown is right” counts the last answer held for up to 10 minutes, which is what a user "
        f"sees; counting only fresh answers would unfairly penalise duty cycling. The gas-triggered policy (wake the PMS7003 when the BME688 signal departs from its recent median; "
        f"threshold {mcu['duty']['thr']} chosen on separate development days) fails because only {duty['quiet'][1]['recall']:.0%} of source windows trip it. Dust and some "
        f"combustion sources barely move the gas sensor.</p>")

    # ---- 4. what to verify against -----------------------------------------------------------------------------------
    S.append("<h2 id='reference'>4. What the device is verified against</h2>")
    S.append("<p>Three different things are verified, and each needs a different reference: <b>how much</b> particulate there is (mass and size), "
             "<b>which gases</b> are present, and <b>which source</b> it came from. The third has no instrument that simply reads it out, so it is "
             "verified in layers, from strongest to weakest evidence.</p>")
    ref_rows = [
        ("PM2.5 mass", "FEM / regulatory monitor (beta-attenuation, TEOM, or an optical monitor with FEM status) at an official air-quality station; "
                       "filter-based gravimetric sampling is the gold standard", "Hourly or 24 h. Co-locate at least 3 units for at least 30 days. EPA PM2.5 sensor targets apply (below).",
         "Station co-location through the state pollution control board or a university partner"),
        ("Particle size bins", "Reference optical particle sizer or aerodynamic particle sizer (APS-class) in a chamber; a second reference OPC outdoors",
         "Verifies the six PMS7003 bins and the coarse-particle under-response the model assumes.", "University aerosol lab"),
        ("CO, NO2, SO2", "Reference gas analysers at the same station", "Hourly. The BME688 is a non-selective sensor, so compare as correlation / regression, not as an accurate reading.", "Same station as PM"),
        ("VOC", "Photo-ionisation detector for totals; GC-FID or PTR-MS for species", "Short campaigns only; essential for solvent and fuel signatures.", "University / research partner"),
        ("Temperature, RH, wind", "Calibrated meteorological station (cup or sonic anemometer)", "Verifies the BME688 T / RH (it reads high from self-heating) and wind direction for the source geometry.", "Station or a portable mast"),
        ("Source: controlled", "Known sources in a chamber or test field: burning wood and straw in a combustion chamber, a diesel generator, resuspended road dust, incense and cooking as confounders",
         "Label is true by construction. Strongest evidence for per-source accuracy and for abstention on confounders.", "You run these; needs a ventilated chamber"),
        ("Source: daily chemistry", "PM2.5 on filters analysed for ions, organic / elemental carbon, elements and levoglucosan, then source apportionment (EPA PMF 5.0 or CMB 8.2)",
         "Tracers: levoglucosan and potassium (biomass), hopanes and elemental carbon (vehicles), calcium, silicon, aluminium, iron (dust), SO2 / arsenic / selenium (coal). Daily resolution only, with 10-30 % uncertainty per factor.", "Analytical lab"),
        ("Source: real-time proxies", "Multi-wavelength aethalometer (wood vs fossil black carbon), aerosol chemical speciation monitor where available, NASA FIRMS satellite fire detections for crop burning, traffic counts, stack monitoring data for industry",
         "Minute-to-hour resolution but each covers only some classes.", "Partner institutions; FIRMS is free"),
        ("Source: observer log", "Trained observer plus a camera, 10-minute blocks", "Cheapest and weakest: the observer is wrong about 10 % of the time in the rehearsal (an assumption). Report accuracy against it as a lower bound.", "You"),
        ("Computation and power", "Cortex-M4 cycle counter (DWT), a current profiler across the Nucleo IDD jumper, the linker map", "Confirms or corrects every estimate in the STM32 twin.", "On the bench"),
    ]
    S.append(tbl(["Quantity", "Reference", "Resolution and how it is used", "Where to get it"], ref_rows))
    S.append("<h3>Strength of the source label</h3>" + tbl(
        ["Tier", "How the label is obtained", "Use it for", "Never use it for"],
        [("A", "Controlled source in a chamber or test field", "Per-source accuracy, confusion between sources, abstention on confounders, detection limits", "Real-world prevalence or mixing"),
         ("B", "Near-source field site with independent tracers (chemistry, aethalometer, FIRMS, stack data)", "Field accuracy, daily source shares", "Window-level accuracy (reference is daily)"),
         ("C", "Observer log and camera", "Event detection and delay, plausibility checks", "Headline accuracy (label noise caps it)"),
         ("D", "Labels derived from another sensor or from the device itself", "Nothing for acceptance", "Anything quoted as verification")]))
    S.append("<h3>Phases</h3>" + tbl(
        ["Phase", "What", "Duration", "Passes when"],
        [("0  Bench", "Rail currents, DWT cycles, stack, SD latency, 10 units side by side in clean (HEPA) air to fit unit spread and noise", "1-2 weeks", "Every measured value falls inside the twin's low-high range"),
         ("1  Chamber", "Tier A sources across humidity 30-90 % and temperature, the three inlet designs in a fan test, size bins against a reference sizer", "3-4 weeks", "Per-source recall and detection limits as planned; chamber-only model is not used for field claims"),
         ("2  Co-location", "At least 3 units next to FEM-class PM2.5 and gas analysers, at least 30 days; the correction is fitted and scored by cross-validation over whole days, never on the day it is applied to", "30+ days", "EPA targets after correction, cross-validated by day"),
         ("3  Near-source field", "3-4 sites (roadside, industrial fence-line, crop-burning region in season, construction): observer logs, cameras, traffic counts, FIRMS, daily filters", "2-4 weeks per site", "At least 30 independent episodes per class collected for the blind test"),
         ("4  Blind pilot", "The 10 prototypes, model and thresholds frozen (record the file hash), pre-registered metrics, new sites and new units", "4-8 weeks", "Criteria below met with the lower end of the 95 % interval")]))
    S.append("<h3>Metrics and proposed acceptance criteria</h3>" + tbl(
        ["Measurement", "Metric", "Proposed target", "Origin"],
        [("PM2.5 mass", "Slope, intercept, R², RMSE / NRMSE, unit-to-unit SD / CV against FEM, after correction, cross-validated by day", "Slope 1.0 ± 0.35, intercept ±5 µg/m³, R² ≥ 0.70, RMSE ≤ 7 µg/m³ or NRMSE ≤ 30 %, SD ≤ 5 µg/m³ or CV ≤ 30 %", "US EPA PM2.5 sensor targets (check the current document)"),
         ("Source attribution", "Macro-F1, per-class precision and recall, with day-block bootstrap intervals", "Macro-F1 ≥ 0.75 against an observer log; higher against Tier A", "Proposal"),
         ("Abstention", "Selective accuracy at coverage", "≥ 90 % correct at ≥ 70 % answered", "Proposal"),
         ("Calibration", "Expected calibration error", "≤ 0.10", "Proposal"),
         ("Confounders", "Share of cooking / incense blocks answered with a source", "≤ 30 %", "Proposal"),
         ("Reproducibility", "Agreement between co-located units", "≥ 85 % of windows", "Proposal"),
         ("Computation", "Measured vs predicted cycles, RAM, energy per hop", "Inside the twin's range", "This report")]) +
        "<p class='cap'>Only the PM2.5 targets come from an external protocol, and they apply to mass, not to source labels. The others are proposals to agree with your supervisor "
        "<b>before</b> collecting data.</p>")
    S.append("""<h3>Pitfalls that ruin a verification</h3><ul class="tight">
<li><b>Time alignment.</b> Log in UTC everywhere. A one-hour offset or averaging periods that start on different minutes costs more R² than any sensor error.</li>
<li><b>Reference conditions.</b> FEM monitors report dry mass after conditioning; the PMS7003 does not dry its sample. Record RH on both.</li>
<li><b>Split by day and by site, never by window.</b> Neighbouring windows are almost identical, so a random split inflates accuracy.</li>
<li><b>Freeze the model before the blind phase</b> and write down its file hash, thresholds and firmware version.</li>
<li><b>Label noise sets a ceiling.</b> An observer who is wrong 10 % of the time makes a perfect device look worse; report the gap (the rehearsal shows it below).</li>
<li><b>Intervals need independent samples.</b> Windows within one episode are not independent; plan in episodes and days.</li>
<li><b>Log raw data</b> (counts and per-step gas resistance at 1 Hz) so the labels can be recomputed with a better model.</li></ul>""")
    S.append("<h3>Closing the loop between simulation and reality</h3><p>After the first week of real data run <code>python -m verify.domain_gap sim.csv real.csv</code>. "
             "It trains a classifier to tell simulated windows from real ones: an area under the curve of 0.5 means indistinguishable and 1.0 means the simulator is far off. "
             "It also ranks the 25 features by how much they differ and names the group of parameters in <code>params.py</code> to fix first. "
             "Re-fit, re-run <code>run_all.py</code>, and track that one number down.</p>")

    # ---- 5. rehearsal ---------------------------------------------------------------------------------------------------
    S.append("<h2 id='rehearsal'>5. Rehearsal on simulated data</h2>")
    S.append(f"<p>Three prototypes next to a simulated FEM-class monitor, reference gas analysers, tracer chemistry and an observer log for {ver['n_days']} days of "
             f"{ver['hours']} h; corrections are scored by 4-fold cross-validation over whole days, so every hour is predicted by a correction that never saw its own day. Each reference carries its own error, so the analysis is "
             f"exercised against imperfect references. The file formats it reads and writes are in <code>results/verification/</code>.</p>")
    S.append(tbl(["Criterion", "Result", "Value"], [(c["name"], Raw(f"<span class='tag {'sup' if c['passed'] else 'no'}'>{'MET' if c['passed'] else 'NOT MET'}</span>"), c["value"]) for c in ver["criteria"]]))
    S.append("<h3>Particle mass</h3>" + img(figs["pm"], "PM2.5 device versus reference", "Every hour predicted by a correction fitted on other days. Left to right: raw device reading, humidity correction, "
             "and a correction in which the device's own source attribution selects the gain.") +
             "<div class='two'><div>" + img(figs["bias"], "Bias by source", "Why the raw reading is hard to correct in the simulator: the same sensor reads different sources differently "
             "(small combustion particles are under-counted, hygroscopic ones over-counted). The size of this effect is an assumption of the simulator, so its size on real hardware is the thing to measure.") +
             "</div><div>" + tbl(["Correction", "Slope", "R²", "RMSE (µg/m³)"], [(k, f"{v['pooled']['slope']:.2f}", f"{v['pooled']['r2']:.2f}", f"{v['pooled']['rmse']:.0f}") for k, v in pc.items()], num_cols=(1, 2, 3)) +
             f"<p class='cap'>Reference-vs-truth error of the simulated FEM monitor: {ver['ref_vs_truth_rmse']:.1f} µg/m³ RMSE. "
             f"Gas channel (information only; MOX sensors are not selective): R² for CO {ver['gas']['CO']['r2']:.2f}, NO2 {ver['gas']['NO2']['r2']:.2f}, SO2 {ver['gas']['SO2']['r2']:.2f}.</p></div></div>")
    t11 = campaign_t11()
    S.append("<h3>Source attribution</h3>" + (f"<p>For context: on the six 24-hour strong-source field days of the main campaign the same device answers correctly {t11['sel_acc']:.0%} of the time when it answers "
             f"(at {t11['coverage']:.0%} coverage). The rehearsal places sources at random distances and bearings, so many episodes are faint or short, and its observer labels carry about 10 % error; "
             f"both lower the score, which is what a real, messy campaign will also do.</p>" if t11 else "") + "<div class='two'><div>" + img(figs["conf"], "Confusion matrix versus observer log",
             f"Rows are the observer's label, columns the device's. Macro-F1 {ver['f1_observer'][0]:.2f} (95 % interval {ver['f1_observer'][1]:.2f}-{ver['f1_observer'][2]:.2f}) against the observer; "
             f"{ver['f1_truth'][0]:.2f} against the hidden truth. The gap is the cost of an imperfect reference.") + "</div><div>" + img(figs["calib"], "Calibration", None) + "</div></div>")
    S.append("<div class='two'><div>" + img(figs["share"], "Daily source shares versus tracer chemistry", "Source-share comparison across days and classes, PM-weighted.") +
             "</div><div>" + tbl(["Class", "Precision", "Recall", "Blocks", "Episodes"], [(c.replace("_", " "), f"{v['precision']:.2f}", f"{v['recall']:.2f}", v["support"], ver["episodes"][c]) for c, v in ver["per_class"].items()], num_cols=(1, 2, 3, 4)) +
             f"<p class='cap'>Blocks are counted over three units that see the same air, so independent episodes are far fewer (last column; many are short fragments where a faint plume crosses the 40 % dominance threshold). "
             f"Per-class values from a dozen days are therefore very noisy: biomass burning has few, short episodes in this site mix and none were recognised, which a longer campaign would resolve. "
             f"Co-located units agree on {ver['unit_agreement']:.0%} of windows.</p></div></div>")
    S.append("<h3>How much data is enough</h3>" + img(figs["samples"], "Sample size", "Left: the 95 % interval on macro-F1 narrows as days are added (this site mix, resampled by day). "
             "Right: a recall near 0.90 is only pinned down to ±0.05 with roughly 140 independent episodes; with 30 it is ±0.11. Plan the number of <i>episodes</i> per class, not hours.") +
             tbl(["Independent episodes", "95 % interval on a recall of 0.90"], [(n, f"{lo:.2f}-{hi:.2f}") for n, lo, hi in ver["wilson"]], num_cols=(0, 1)))

    # ---- changes made to the simulator ----------------------------------------------------------------------------------------
    cc = campaign_counts()
    ba = (f"The 16-test campaign was re-run: {cc['after'][0]} of {cc['after'][1]} tests meet their criteria now, {cc['before'][0]} of {cc['before'][1]} before. "
          if cc["before"] and cc["after"] else "")
    S.append("<h2 id='changes'>What this work changed in the simulator</h2><div class='card'><ul class='tight'>"
             "<li><b>PMS7003 sample heating added.</b> Rehearsing the PM2.5 comparison showed device readings of thousands of µg/m³ in near-fog air, because particles grew with the "
             "ambient humidity. A real PMS7003 warms the sample by a few degrees inside its housing (assumed +5 °C here) and field comparisons show 2-3x over-reading at high humidity, not 70x. "
             "The effective humidity is now lower and capped at 95 %. " + ba + "The earlier run is kept in <code>results/previous_run/</code>.</li>"
             "<li><b>Firmware variants</b> (<code>FW_OPT_*</code> flags in <code>polso_fw.c</code>) are off by default; the reference build is unchanged apart from profiling hooks.</li>"
             "<li><b>Humidity decimation</b> was first written as “hold one reading for 10 s”; the equivalence check rejected it and it now averages ten readings.</li>"
             "<li><b>Correction of an earlier note:</b> <code>params.py</code> described a “Stop 2” low-power mode. The G4 family has Stop 0 and Stop 1 only.</li></ul></div>")

    # ---- 6. on the board ----------------------------------------------------------------------------------------------------
    S.append("<h2 id='measure'>6. First week with the board</h2>")
    S.append(tbl(["What", "How", "Replaces"], [
        ("MCU current in Run, Sleep, Stop 1", "Remove jumper JP6 on the Nucleo and put a current meter or power profiler (PPK2, Otii) across it", "Run, Sleep and Stop currents"),
        ("Cycles per function and stage", Raw("Flash <code>firmware_c/bench/dwt_bench.c</code> with release flags; paste the BENCH lines into <code>python -m stm32_twin.compare_measured</code>"), "Operation costs and the stage cycles"),
        ("Flash and SRAM used, stack depth", "Linker map; paint the stack and read the high-water mark after an hour with GPS and SD running", "Memory budget"),
        ("SD write latency", "Time every write on the real card for 24 h and keep the histogram", "SD stall distribution and ring-overrun estimate"),
        ("UART overruns", "Count lost bytes on the GPS and PMS ports over 24 h", "Ring-buffer risk"),
        ("Stop 1 wake time, PMS frame period, BME688 read time", "Logic analyser on a GPIO / SCL", "Wake penalty and schedule assumptions")]) +
        "<p class='cap'>Still to confirm from the ST datasheet (DS12288, not retrievable when this was written): run-mode current at 170 MHz, Stop 1 current, "
        "flash wait states at 170 MHz in boost mode, and whether DMA can reach the CCM SRAM. The G4 family has Sleep, Stop 0, Stop 1, Standby and Shutdown; it has no “Stop 2”.</p>")
    S.append("""<h3>Reproduce</h3><pre>cd environment_simulation
python run_all.py          # train + the 16-test campaign          (~12 min)
python run_mcu.py          # STM32 twin                              (~15 min)
python run_verification.py # virtual co-location rehearsal            (~2 min)
python build_report.py     # this report
python -m unittest discover tests</pre>""")
    S.append(f"<p class='sub'>Total run time of the STM32 twin: {mcu['meta']['runtime_min']:.0f} min; verification rehearsal: {ver['runtime_min']:.1f} min.</p>")

    body = "\n".join(S)
    return body


def main():
    mcu = pickle.load(open(os.path.join(RES, "mcu_results.pkl"), "rb"))
    ver = pickle.load(open(os.path.join(RES, "verification_results.pkl"), "rb"))
    body = build(mcu, ver)
    css = BASE_CSS + EXTRA_CSS
    title = "PolSo STM32 Twin"
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{css}</style></head><body><main>{body}</main></body></html>")
    with open(os.path.join(RES, "stm32_and_verification_report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    frag = f"<meta charset='utf-8'><title>{title}</title><style>{css}</style><main>{body}</main>"
    with open(os.path.join(RES, "stm32_report_artifact.html"), "w", encoding="utf-8") as f:
        f.write(frag)
    print(f"wrote results/stm32_and_verification_report.html ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
