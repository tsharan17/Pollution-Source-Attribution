"""
Build sheet for the real PolSo device, generated from the simulation's own source files.

    python build_spec.py            -> results/build_params.json   (shown in the viewer's Build tab)
                                     -> ../BUILD_GUIDE.md            (same content, for the repo)

Every value that the firmware or the twin uses is read from the code (params.py, stm32_twin/spec.py,
firmware_c/*.h, results/pm_cal.json), so this sheet cannot drift from what was simulated.
Hardware choices that the simulation does not decide (pins, clock tree, wiring) are marked PROP:
a proposal that must be checked in STM32CubeMX and against UM2505 / RM0440 before soldering.

Tags:  DOC  datasheet / reference-manual value      SIM   value the twin and firmware actually use
       EST  engineering estimate, measure it         PROP  proposed hardware choice, verify
"""
import json
import os
import re

from polso_twin import params as P
from stm32_twin import spec

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, "firmware_c")


def _defines(path):
    out = {}
    for m in re.finditer(r"^#define\s+(\w+)\s+([^\s/]+)", open(path, encoding="utf-8").read(), re.M):
        out[m.group(1)] = m.group(2)
    return out


def _f(x, n=4):
    return f"{x:.{n}g}"


def build():
    fwh = _defines(os.path.join(FW, "polso_fw.h"))
    mp = _defines(os.path.join(FW, "model_params.h"))
    cal = json.load(open(os.path.join(HERE, "results", "pm_cal.json")))
    n_in, h1, h2, n_out = (int(mp[k]) for k in ("MP_N_IN", "MP_H1", "MP_H2", "MP_N_OUT"))
    macs = n_in * h1 + h1 * h2 + h2 * n_out
    w_bytes = macs + 4 * (h1 + h2 + n_out) + 4 * 2 * n_in
    S = []

    def sec(sid, title, intro, rows=None, cols=("Item", "Value", "Notes", "Tag"), steps=None, code=None):
        S.append(dict(id=sid, title=title, intro=intro, cols=list(cols), rows=rows or [], steps=steps or [], code=code))

    sec("bom", "1 · Parts list",
        "Everything the simulated device is made of. Part numbers are the ones the twin models; any equivalent works if it matches the electrical numbers.",
        [["Microcontroller board", "STM32 NUCLEO-G474RE", "Cortex-M4F + FPU, 170 MHz, 512 KB flash, 128 KB SRAM, on-board ST-LINK-V3E", "DOC"],
         ["Particle sensor", "Plantower PMS7003", "Laser OPC, 0.3–10 µm, 5 V fan, 3.3 V UART, 1.25 mm 10-pin connector (get the adapter board)", "DOC"],
         ["Gas / T / RH / p sensor", "Bosch BME688 breakout", "MOX gas sensor with programmable heater, I2C, 1.71–3.6 V", "DOC"],
         ["GPS", "u-blox NEO-6M class module", "UART 9600 baud NMEA, 3.3 V; power-save mode used for battery life", "DOC"],
         ["Storage", "microSD SPI breakout + card ≤ 32 GB", "FAT32, 3.3 V logic (no 5 V level shifter needed on Nucleo)", "DOC"],
         ["Battery", f"Li-ion 1S {P.POWER['battery_mah']} mAh, {P.POWER['battery_v']} V", f"Usable fraction assumed {P.POWER['usable_frac']}; add a protection board", "SIM"],
         ["Charger", "TP4056 (with protection) or equivalent", "USB-C 5 V in, 1S Li-ion charge", "PROP"],
         ["5 V boost", "MT3608 / TPS61023 class, ≥ 300 mA", f"Feeds PMS7003 fan and Nucleo E5V. Efficiency modelled {P.POWER['boost_eff']}", "PROP"],
         ["Enclosure", "PolSo-19 hooded enclosure", f"Inlet losses from PolSo-19 Run A: fine {P.ENCLOSURES['hooded']['loss_fine']}, coarse {P.ENCLOSURES['hooded']['loss_coarse']}, directional {P.ENCLOSURES['hooded']['dir_amp']}", "SIM"],
         ["Misc.", "Dupont wires, 2 × 10 kΩ I2C pull-ups (if breakout lacks them), N-MOSFET for GPS power gate", "", "PROP"]])

    sec("pins", "2 · Wiring and pin map",
        "Proposed pins on the Nucleo Arduino / morpho headers. They avoid the ST-LINK virtual COM port (LPUART1 on PA2/PA3), the user LED (PA5) and the user button (PC13). Open STM32CubeMX with the NUCLEO-G474RE board selected and confirm each pin and alternate function before wiring.",
        [["PMS7003 TX → MCU RX", "PA10 (D2) · USART1_RX", "9600 baud 8N1, RX by DMA circular buffer", "PROP"],
         ["PMS7003 RX ← MCU TX", "PA9 (D8) · USART1_TX", "Only needed to send passive / sleep commands", "PROP"],
         ["PMS7003 SET", "PB5 (D4) · GPIO out", "High = run, low = sleep (fan off)", "PROP"],
         ["PMS7003 RESET", "PB4 (D5) · GPIO out", "Active low; keep high", "PROP"],
         ["PMS7003 VCC / GND", "5 V boost / GND", "Fan draws ~100 mA peak; logic is 3.3 V", "DOC"],
         ["BME688 SCL", "PB8 (D15) · I2C1_SCL", "400 kHz fast mode, 4.7–10 kΩ pull-up to 3.3 V", "PROP"],
         ["BME688 SDA", "PB9 (D14) · I2C1_SDA", "", "PROP"],
         ["BME688 SDO", "GND → address 0x76 (or 3.3 V → 0x77)", "Chip ID register 0xD0 must read 0x61", "DOC"],
         ["BME688 VCC", "3.3 V", "", "DOC"],
         ["GPS TX → MCU RX", "PC11 (CN7) · USART3_RX", "9600 baud NMEA, DMA circular", "PROP"],
         ["GPS RX ← MCU TX", "PC10 (CN7) · USART3_TX", "UBX commands for power-save mode", "PROP"],
         ["GPS power gate", "PB0 (A3) · GPIO → N-MOSFET", "Optional: switch the module off between fixes", "PROP"],
         ["microSD SCK / MISO / MOSI", "PB13 / PB14 / PB15 · SPI2", "10 MHz after init (400 kHz during card init)", "PROP"],
         ["microSD CS", "PB12 · GPIO out", "", "PROP"],
         ["Battery sense", "PA0 (A0) · ADC1_IN1 via 100k/100k divider", "Optional battery % on the log", "PROP"],
         ["Debug / log", "LPUART1 PA2 / PA3 → ST-LINK VCP", "115200 baud to PC over the USB cable", "DOC"],
         ["Status LED", "PA5 (LD2)", "Blink pattern = state (warm-up / ok / fault)", "DOC"]])

    sec("mcu", "3 · Microcontroller configuration",
        "Settings to enter in STM32CubeMX (or set in code). The cycle and power numbers in the STM32 tab assume exactly this configuration.",
        [["System clock", f"{spec.F_MAX_MHZ:.0f} MHz", "HSI16 → PLL: M = /4, N = ×85, R = /2 → 170 MHz SYSCLK; AHB/APB1/APB2 = /1", "PROP"],
         ["Voltage scaling", "Range 1 boost mode", "Required above 150 MHz", "DOC"],
         ["Flash latency", "4 wait states (check RM0440 table)", f"ART accelerator + prefetch ON. The twin's cost model allows {spec.ART_INFLATION[0]}–{spec.ART_INFLATION[2]}× cycle inflation from flash; the twin keeps a conservative {spec.FLASH_WAIT_STATES} WS figure", "PROP"],
         ["Low-speed clock", "LSE 32.768 kHz (X2 on Nucleo)", "Drives RTC and LPTIM1 so timing survives Stop mode", "PROP"],
         ["Wake-up timer", "LPTIM1, 1 Hz", "One wake per second: read the PMS frame, run a BME688 heater step, update the window", "SIM"],
         ["USART1 (PMS7003)", "9600 8N1, RX DMA circular 64 B", "Parse 32-byte frames in the half/full-transfer callbacks", "PROP"],
         ["USART3 (GPS)", "9600 8N1, RX DMA circular 256 B", "", "PROP"],
         ["I2C1 (BME688)", "400 kHz, 7-bit address 0x76", "Polling is fine (≈ 30 bytes per second)", "PROP"],
         ["SPI2 (microSD)", "Mode 0, 8-bit, TX DMA", "Write one CSV line per 30 s window; batch to 512-byte sectors", "PROP"],
         ["NVIC priorities", "DMA/USART 5, LPTIM1 6, SysTick off in Stop", "Keep the wake-up interrupt below the UART DMA", "PROP"],
         ["Low-power mode", "Stop 1 between wakes (optimised firmware A4)", f"Run {spec.I_RUN_170_MA[1]} mA / Sleep {spec.I_SLEEP_170_MA[1]} mA / Stop {spec.I_STOP_MA[1]} mA nominal at 3.3 V", "EST"],
         ["FPU", "Enabled, hard-float ABI (-mfloat-abi=hard -mfpu=fpv4-sp-d16)", "", "DOC"],
         ["Compiler flags", "-O2 -mcpu=cortex-m4 -mthumb -ffunction-sections -fdata-sections", "Plus -DFW_OPT_LOGCACHE (proved identical answers in the SIL test)", "PROP"]])

    sec("sensors", "4 · Sensor settings",
        "How each sensor is driven. These are the settings the sensor models assume.",
        [["PMS7003 mode", "Active (continuous) mode", "Firmware keeps the newest valid frame each second; one frame = 32 bytes, header 0x42 0x4D, checksum = sum of bytes 0–29", "SIM"],
         ["PMS7003 start-up", f"discard first {P.PMS_WAKE_STABLE_S} s after fan start", "Datasheet: data stable 30 s after wake", "DOC"],
         ["PMS7003 channels used", "6 counts (>0.3 … >10 µm per 0.1 L) + PM1/PM2.5/PM10 CF=1", "uint16 fields, big-endian", "DOC"],
         ["PMS7003 command: passive", "42 4D E1 00 00 01 70", "", "DOC"],
         ["PMS7003 command: active", "42 4D E1 00 01 01 71", "", "DOC"],
         ["PMS7003 command: sleep / wake", "42 4D E4 00 00 01 73 / 42 4D E4 00 01 01 74", "Or use the SET pin", "DOC"],
         ["BME688 mode", "Forced mode, one heater step per second", "Read T, RH, p and gas resistance after each step; check gas_valid and heat_stab bits", "SIM"],
         ["BME688 heater profile", " → ".join(f"{t}" for t in P.HEATER_PROFILE) + " °C", f"{len(P.HEATER_PROFILE)} steps, repeats every {len(P.HEATER_PROFILE)} s; set target and ~100 ms heat time per step", "SIM"],
         ["BME688 oversampling", "T ×2, p ×1 (unused), RH ×1, IIR filter off", "Use the Bosch BME68x driver for heater-resistance conversion", "PROP"],
         ["BME688 self-heating", f"+{P.BME_SELF_HEAT} °C (unit spread {P.BME_SELF_HEAT_UNIT_SD} °C)", "Modelled warming of T/RH reading inside the enclosure — measure against a reference thermometer", "EST"],
         ["GPS", "NMEA GGA/RMC only, power-save mode", "Position and UTC time for the log; not used by the classifier", "PROP"],
         ["microSD log line", "one per 30 s window", "time, lat, lon, label, label10, confidence, pm_raw, pm_corr, pm5, pm5_lo, pm5_hi, T, RH, fault flags, battery", "PROP"]])

    sec("fw", "5 · Firmware constants",
        "Values in the C firmware (firmware_c/polso_fw.h, polso_fw.c) that produced every result in this simulation. Change them only together with a re-run of the twin.",
        [["Window length", f"{P.FW_WINDOW_S} s", "Features are computed over the last 60 one-second samples", "SIM"],
         ["Hop", f"{P.FW_HOP_S} s", "One answer every 30 s", "SIM"],
         ["Warm-up", f"{P.FW_WARMUP_S} s", "Device reports 'warming up' until the MOX baseline settles", "SIM"],
         ["PM data needed", f"{P.FW_PM_VALID_FRAC:.0%} of window", "Else the window is a sensor fault", "SIM"],
         ["Gas data needed", f"{P.FW_GAS_VALID_FRAC:.0%} of window", "", "SIM"],
         ["Baseline tracker up", f"{P.FW_BASE_UP} per sample", "Rise toward cleaner air (time constant ≈ 5.6 h)", "SIM"],
         ["Baseline tracker down", f"{P.FW_BASE_DOWN} ln units per sample", "Slow fall (≈ 0.009 per day)", "SIM"],
         ["Probability smoothing", f"EMA {P.FW_PROB_EMA}", "Between consecutive windows", "SIM"],
         ["Confidence gate", f"{float(mp['MP_CONF_THR'].rstrip('f')):.2f}", "Below this the device says 'uncertain' instead of guessing", "SIM"],
         ["5-min PM2.5 average", f"last {fwh['REP_PM_N']} windows", "Mean of corrected PM2.5 with a 16–84 % band", "SIM"],
         ["10-min source", f"last {fwh['REP_LAB_N']} windows", f"Majority label; needs ≥ {fwh['REP_MIN_ANSWERS']} answers and ≥ 50 %, else 'mixed'", "SIM"],
         ["Sample rate", "1 Hz", "PMS frame + one BME688 heater step per second", "SIM"]])

    feat_desc = (["Share of counts in each PMS size bin, log10 (0.3–0.5 … >10 µm)"] * 6 +
                 ["log10(1 + count >0.3 µm)", "PM2.5 / PM10", "PM1 / PM2.5", "Count variability (sd / mean)",
                  "Count skewness (spikes)", "Fastest 5-s rise in log count",
                  "Mean gas response (baseline-corrected ln R)"] +
                 [f"Gas response at {t} °C minus mean" for t in P.HEATER_GROUPS] +
                 ["Gas trend at 300 °C over the window", "Gas trend at 200 °C over the window",
                  "Correlation of particles and gas at 300 °C", "Gas response per unit PM2.5",
                  "Relative humidity / 100", "Temperature / 50 °C"])
    sec("model", "6 · Classifier",
        f"Int8 neural network {n_in}-{h1}-{h2}-{n_out}, trained in polso_twin/model.py and exported to firmware_c/model_params.h. "
        f"{macs} multiply-accumulates per answer, ≈ {w_bytes / 1024:.1f} KB of weights and scales. Outputs: " + ", ".join(P.CLASSES) + ".",
        [[f"{i}", n, d, "SIM"] for i, (n, d) in enumerate(zip(P.FEATURE_NAMES, feat_desc))],
        cols=("#", "Feature", "How it is computed", "Tag"))

    rows = []
    for k, v in cal.items():
        rows.append([k.replace("_", " "), f"α {v['alpha']:.3f} · β {v['beta']:.3f} · κ {v['kappa']:.3f}",
                     f"[{v['q16']:+.2f}, {v['q84']:+.2f}] ln units (×{v.get('band_scale', 1):.2f} widened), fit on {v['n']} windows", "SIM"])
    sec("pmcal", "7 · PM2.5 correction",
        "PM_corrected = exp(β) · PM_raw^α / (1 + κ · h), with h = a / (1 − a), a = min(RH, 95) / 100, chosen per detected source. "
        "Values are in firmware_c/pm_cal_params.h. They come from simulated co-location; replace them with a real co-location against a reference monitor (step 10).",
        rows, cols=("Source", "Coefficients", "Uncertainty band", "Tag"))

    pw = P.POWER
    sec("power", "8 · Power budget inputs",
        "Datasheet typicals the battery-life numbers use. The memory map and the full current budget per design lever are shown live below from the STM32 run.",
        [["PMS7003 active / sleep", f"{pw['pms_active_ma']} / {pw['pms_sleep_ma']} mA at 5 V", "", "DOC"],
         ["BME688 (heater duty)", f"{pw['bme_ma']} mA average", "≈ 75 % heater duty at 12 mA", "EST"],
         ["MCU run / sleep / stop", f"{spec.I_RUN_170_MA} / {spec.I_SLEEP_170_MA} / {spec.I_STOP_MA} mA", "low / nominal / high at 3.3 V — measure on IDD jumper JP6", "EST"],
         ["Nucleo overhead", f"{pw['nucleo_overhead_ma']} mA", "ST-LINK + LEDs; cut it off (or use a bare G474 board) for long runs", "EST"],
         ["GPS track / power-save", f"{pw['gps_ma']} / {pw['gps_psm_ma']} mA", "", "DOC"],
         ["microSD write", f"{pw['sd_write_ma']} mA for {pw['sd_write_ms']} ms, {pw['sd_writes_per_min']}× per min", "", "EST"],
         ["Battery", f"{pw['battery_mah']} mAh × {pw['usable_frac']} usable", f"boost {pw['boost_eff']}, 3.3 V LDO {pw['ldo_eff']:.2f}", "SIM"]])

    est = [["PMS sample heating", f"+{P.PMS_SAMPLE_HEAT_C} °C", "Laser and fan warm the sampled air, lowering its RH. Measure: PMS vs reference PM2.5 at RH > 80 %", "EST"],
           ["PMS RH cap", f"{P.PMS_RH_CAP} %", "Fog/droplet regime not modelled — expect larger errors in fog", "EST"],
           ["PMS unit gain spread", f"σ {P.PMS_UNIT_GAIN_SIGMA}", "Co-locate every unit for one day", "EST"],
           ["PMS fouling", f"{P.PMS_GAIN_LOSS_PER_DAY} per day", "Re-check monthly against the reference", "EST"],
           ["MOX baseline drift", f"{P.MOX_DRIFT_PER_DAY} ln/day + random walk {P.MOX_RW_PER_SQRT_DAY}/√day", "Log clean-air resistance for 2 weeks", "EST"],
           ["MOX unit spread", f"R_ref σ {P.UNIT_MOX_RREF_SIGMA}, sensitivity σ {P.UNIT_MOX_SENS_SIGMA}", "Compare units side by side in clean air", "EST"],
           ["MOX noise", f"{P.MOX_NOISE} ln R per reading", "Standard deviation of ln R in steady clean air", "EST"],
           ["Enclosure losses", "PolSo-19 Run A numbers (hooded)", "Repeat with the printed enclosure next to an open sensor", "EST"]]
    sec("assume", "9 · Simulation assumptions to check first",
        "The twin depends on these modelled numbers. Each has a simple bench test. If a measurement falls outside the stated value, re-run the twin with the measured value before trusting the accuracy figures.",
        est, cols=("Assumption", "Value used", "How to measure", "Tag"))

    sec("steps", "10 · Build, flash and verify — step by step",
        "From parts on the desk to a device whose answers can be compared with the simulation.",
        steps=[
            "Install STM32CubeIDE (includes CubeMX and the GCC toolchain).",
            "New project → Board Selector → NUCLEO-G474RE. Apply the clock and peripheral settings from section 3 and the pins from section 2.",
            "Copy firmware_c/polso_fw.c, polso_fw.h, pms7003.c, pms7003.h, model_params.h, pm_cal_params.h (and fastmath.h) into Core/Src and Core/Inc.",
            "In main.c: start USART1/USART3 DMA reception, init the BME688 (Bosch BME68x driver) with the heater profile, call fw_init(&state, b0) where b0[10] is the clean-air ln(gas resistance) per heater step (measure it once outdoors in clean air and store it in flash), start LPTIM1 at 1 Hz.",
            "On each LPTIM1 wake: pass the UART DMA bytes to fw_uart_rx(), read the BME688 step, call fw_tick(); every 30 s it fills fw_result_t — write it to the SD card and LPUART1.",
            "Bench test 1 — parser and maths: run firmware_c/sil.exe on a recorded stream on the PC and on the board (feed the same bytes over the VCP); outputs must match line for line.",
            "Bench test 2 — timing: build firmware_c/bench/dwt_bench.c, read the DWT cycle counter, and enter the numbers in stm32_twin/compare_measured.py (see stm32_twin/MEASURE_ON_BOARD.md).",
            "Bench test 3 — current: remove JP6, put a multimeter / power profiler in series, compare with the STM32 tab's budget.",
            "Co-location: place the device next to a reference PM2.5 monitor (BAM / TEOM or a calibrated optical monitor) for ≥ 7 days. Log both.",
            "Fit the PM2.5 correction with polso_twin/pmcal.py on the co-location data, export pm_cal_params.h, re-flash.",
            "Source labels: keep a diary of known events (traffic peak, a nearby fire, dust) and compare with label10; verify/ scripts compute the agreement.",
            "Compare everything with the simulation: python run_verification.py — any number outside the twin's band is a model error to fix before trusting the map."])

    sec("files", "11 · Where everything lives in the repo",
        "Run commands from environment_simulation/.",
        [["Sensor and air models", "polso_twin/params.py, physics.py, sensors.py, environment.py", "All modelled numbers in one place (params.py)", "SIM"],
         ["Firmware twin (Python)", "polso_twin/firmware.py", "Mirrors the C code; used for training and tests", "SIM"],
         ["Firmware (C, runs on the board)", "firmware_c/polso_fw.c/.h, pms7003.c/.h", "Portable C99, no HAL calls", "SIM"],
         ["Model weights", "firmware_c/model_params.h", "Auto-generated: python run_all.py", "SIM"],
         ["PM2.5 correction", "firmware_c/pm_cal_params.h, results/pm_cal.json", "polso_twin/pmcal.py", "SIM"],
         ["STM32 timing / power model", "stm32_twin/", "python run_mcu.py", "EST"],
         ["Area map simulation", "polso_twin/area.py", "python run_area.py", "SIM"],
         ["Viewer", "viewer/area_viewer_template.html → results/area_viewer.html", "python viewer/build_viewer.py", "SIM"],
         ["This build sheet", "build_spec.py → results/build_params.json, BUILD_GUIDE.md", "python build_spec.py", "SIM"],
         ["Tests", "tests/", "python -m unittest discover tests", "SIM"]])
    return dict(sections=S, tags=dict(DOC="datasheet / reference manual", SIM="value the simulation and firmware use",
                                       EST="estimate — measure it", PROP="proposed — verify in CubeMX before wiring"))


def to_markdown(b):
    out = ["# PolSo build guide", "",
           "Generated by `environment_simulation/build_spec.py` from the simulation's source files — do not edit by hand.",
           "The same content is in the **Build** tab of the simulation viewer.", "",
           "Tags: " + " · ".join(f"**{k}** {v}" for k, v in b["tags"].items()), ""]
    for s in b["sections"]:
        out += [f"## {s['title']}", "", s["intro"], ""]
        if s["rows"]:
            out.append("| " + " | ".join(s["cols"]) + " |")
            out.append("|" + "---|" * len(s["cols"]))
            for r in s["rows"]:
                out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
            out.append("")
        for i, st in enumerate(s["steps"], 1):
            out.append(f"{i}. {st}")
        if s["steps"]:
            out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    b = build()
    with open(os.path.join(HERE, "results", "build_params.json"), "w", encoding="utf-8") as f:
        json.dump(b, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "..", "BUILD_GUIDE.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(b))
    print("wrote results/build_params.json and ../BUILD_GUIDE.md;", sum(len(s["rows"]) + len(s["steps"]) for s in b["sections"]), "entries")
