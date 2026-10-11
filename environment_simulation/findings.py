"""Findings / recommendations for the report, written from the actual numbers in the results."""


def _pct(x):
    return f"{x:.0%}"


def findings_and_recommendations(R, meta):
    g = R["T00"]["data"]
    f15 = R["T15"]["data"]
    pw = R["T14"]["data"]["modes"]
    dr = R["T08"]["data"]
    t13 = R["T13"]["data"]
    rh = R["T03"]["data"]
    t11 = R["T11"]["data"]["overall"]
    unseen = R["T10"]["data"]["table"]
    faults = R["T12"]["data"]["cases"]
    lim = R["T02"]["data"]["limits"]

    kpis = [
        ("held-out field macro-F1 (deployed int8 model)", f"{g['B']['macro_f1']:.2f}"),
        ("same, if trained on chamber data only", f"{g['A']['macro_f1']:.2f}"),
        ("fusion gain over best single sensor", f"+{f15['B_fused'] - max(f15['B_pm_only'], f15['B_gas_only']):.2f}"),
        ("battery life, NUCLEO + 2000 mAh", f"{pw['NUCLEO board, continuous']['hours']:.1f} h"),
        ("firmware RAM / model flash", f"{t13['state_bytes'] / 1024:.1f} / {t13['weight_bytes'] / 1024:.1f} KB"),
        ("tests passed", f"{sum(r['verdict'] == 'PASS' for r in R.values())} / {len(R)}"),
    ]

    findings = [
        f"<b>Fusion is justified.</b> On held-out field data the fused model reaches macro-F1 {f15['B_fused']:.2f} vs "
        f"{f15['B_pm_only']:.2f} (PM only) and {f15['B_gas_only']:.2f} (gas only). PM alone separates dust from combustion; "
        f"the BME688 heater profile is what separates vehicle, biomass and industrial combustion.",
        f"<b>Chamber data alone will not generalise.</b> A model trained only on chamber sessions scores macro-F1 "
        f"{g['A']['macro_f1']:.2f} outdoors vs {g['B']['macro_f1']:.2f} when diverse labelled field sessions (humidity, wind, "
        f"background, ageing, several units) are added. Budget time for labelled field collection, not just chamber runs.",
        f"<b>Humidity is the biggest environmental risk.</b> Accuracy by RH: "
        + ", ".join(f"{k}% {_pct(v['acc'])}" for k, v in rh["table"].items())
        + f". In fog, {_pct(rh['fog_false'])} of clean-air windows were blamed on a source (hygroscopic growth inflates PMS counts).",
        "<b>The twin caught a firmware bug.</b> The first gas-baseline tracker (rise 0.01 / fall 1e-4 per sample) slowly "
        "absorbed sustained pollution into the 'clean-air' baseline: on the 24 h test days accuracy fell from ~99 % in the first "
        "hours to 25-40 % by evening, and field-day selective accuracy was 73 %. Slower rates (0.0005 / 1e-6), chosen on separate "
        "development days (different seeds and unit), fixed it. Results below use the fixed firmware.",
        f"<b>Sensor drift needs on-device baseline tracking.</b> macro-F1 at day 180: {dr['tracking'][180]:.2f} with tracking vs "
        f"{dr['frozen'][180]:.2f} with the frozen factory baseline (day 365: {dr['tracking'][365]:.2f} vs {dr['frozen'][365]:.2f}).",
        f"<b>Detection limit.</b> Smallest excess PM2.5 attributed correctly 80 % of the time over a 30 ug/m3 background: "
        + ", ".join(f"{k.replace('_', ' ')} {v if v else '>200'} ug/m3" for k, v in lim.items())
        + ". Vehicle soot is mostly below the PMS7003 0.3 um cut, so traffic is recognised largely through gases.",
        f"<b>Unseen sources:</b> " + "; ".join(
            f"{k.replace('_', ' ')} -> mostly '{v['top'].replace('_', ' ')}' (wrong {_pct(v['wrong'])})" for k, v in unseen.items())
        + ". Cooking and incense have no class of their own, so expect them to be mislabelled unless a class is added.",
        f"<b>24 h field days:</b> selective accuracy {_pct(t11['sel_acc'])} at {_pct(t11['coverage'])} coverage "
        f"(the device abstains on the rest).",
        f"<b>Firmware is small and fast.</b> C core matches the Python twin on {t13['agree']:.2%} of windows; "
        f"{t13['state_bytes'] / 1024:.1f} KB RAM, {t13['weight_bytes'] / 1024:.1f} KB model, ~{t13['est_ms']:.2f} ms per inference. "
        f"The STM32G474 (128 KB SRAM, 512 KB flash) is far more than enough.",
        f"<b>Fan stall is invisible to the firmware.</b> A slowed PMS fan still produces valid frames; the device kept reporting "
        f"confident labels on {_pct(faults['pms_fan_stall_after_1h']['confident_after'])} of windows afterwards. Dead sensors and "
        f"UART corruption are handled (flagged / resynchronised).",
        f"<b>Power is the hard constraint.</b> " + "; ".join(f"{k}: {v['hours']:.1f} h" for k, v in pw.items())
        + ". The PMS7003 fan, the GPS and the NUCLEO's ST-LINK dominate.",
    ]

    # confidence-gate trade-off, recomputed from stored window probabilities (no re-simulation)
    import numpy as np
    from polso_twin import campaign as C
    sweep = []
    for thr in (0.5, 0.7, 0.9, meta.get("conf_thr", 0.95)):
        ms = {}
        for key in ("T00", "T11"):
            d = R[key]["data"]["df"]
            d = d.assign(pred=C.relabel(d, np.stack(d["probs"]), thr))
            ms[key] = C.metrics(d)
        sweep.append(f"gate {thr:.2f}: field sessions {ms['T00']['sel_acc']:.0%} correct @ {ms['T00']['coverage']:.0%} answered, "
                     f"24 h days {ms['T11']['sel_acc']:.0%} @ {ms['T11']['coverage']:.0%}")
    findings.append(f"<b>The confidence gate is the main accuracy/coverage dial.</b> The deployed gate "
                    f"({meta.get('conf_thr', 0):.2f}) was set on validation data to answer 90 % of clear windows. "
                    "Several FAILs (T01, T02, T07) come mostly from abstentions being scored as errors. "
                    "Trade-off on held-out data: " + "; ".join(sweep) + ".")

    if f15["B_rf"] > f15["B_fused"] + 0.03:
        findings.insert(1, f"<b>A decision forest generalised better than the MLP</b> on held-out field windows "
                           f"(macro-F1 {f15['B_rf']:.2f} vs {f15['B_fused']:.2f}). The deployed int8 MLP is tiny "
                           f"({R['T13']['data']['weight_bytes'] / 1024:.1f} KB); a pruned forest (e.g. 30 trees, depth 8, "
                           f"tens of KB) also fits the G474 and is worth comparing on real data.")

    recs = [
        "<b>Battery:</b> 2000 mAh is not enough for all-day continuous use. Buy 2x 18650 cells in parallel "
        "(~6000-7000 mAh) or plan duty cycling; prefer a GPS with a power-save mode (u-blox M8/M10) over a NEO-6M.",
        "<b>Prototype on the NUCLEO, deploy on a bare STM32G474 board</b> (or cut the ST-LINK power): the debugger alone costs "
        "roughly 45 mA in this model, so measure it on day one.",
        "<b>Configure the BME688 heater profile</b> (10 steps, 150-400 C, parallel mode) - the static IAQ number is not enough; "
        "the per-step resistances are the fingerprint.",
        "<b>Add a PMS fan-health check</b> in firmware: flag when all count channels collapse together while gas and RH are steady, "
        "and periodically compare PM against the gas response; consider the PMS SET pin to power-cycle the sensor.",
        f"<b>Revisit the 80 % data-validity rule:</b> with 30 % of PMS frames lost (loose wire, noisy UART) the device "
        f"went silent on {_pct(faults['pms_frames_dropped_30%']['fault_flag_after'])} of windows even though the remaining "
        "frames were good. Use short, shielded UART leads, and consider accepting windows with >= 50 % valid frames.",
        "<b>Handle humidity explicitly:</b> keep the separate BME688 vent, log RH with every window, and treat RH > 90 % as "
        "low-confidence (or add a small inlet heater / RH correction once real PM-vs-RH data is collected).",
        "<b>Keep the hooded inlet</b> from PolSo-19 and print it first; verify inlet loss with a fan test (the model assumes "
        "PolSo-19 coefficients).",
        "<b>Data plan:</b> chamber runs for each source + at least several days of labelled field data per class across "
        "humidity/seasons, with 3+ different units, logged at 1 Hz raw (counts + per-step resistance) - not just PM/IAQ outputs.",
        "<b>First week with hardware:</b> (1) measure each rail's current; (2) 48 h co-location of all units in clean air to fit "
        "unit spread and noise; (3) RH step test; (4) a few chamber exposures (incense/candle, road dust, vehicle exhaust); then "
        "re-fit params.py and re-run this campaign before building all 10 units.",
    ]

    figs_for = {"T00": [("confusions", "confusion matrices")], "T02": [("detection", "detection curves")],
                "T08": [("robustness", "robustness curves")], "T15": [("bars", "fusion, units and battery")]}
    return dict(kpis=kpis, findings=findings, recommendations=recs, figs_for=figs_for)
