# PolSo digital twin – virtual test campaign

A full software model of the pollution-source-attribution device, used to test the design
**before buying hardware**: STM32 NUCLEO-G474RE + PMS7003 + BME688 + GPS + microSD +
2000 mAh Li-ion in the hooded PolSo-19 enclosure.

```
environment.py   weather, regional background, time-varying sources (traffic pulses, burning
                 episodes, wind-driven dust, meandering industrial plumes), 14 source subtypes
sensors.py       PMS7003 optics (hygroscopic growth, counting efficiency, sizing blur, Poisson
                 noise, smoothing, uint16 saturation, real 32-byte frames) and BME688 MOX
                 (10-step heater profile, per-gas temperature selectivity, oxidising NO2,
                 humidity, adsorption/desorption dynamics, drift, warm-up, unit spread)
physics.py       kappa-Koehler growth, humidity, lognormal modes, PolSo-19 inlet losses
simulate.py      scenario -> sensors -> transport (fault injection) -> device streams + truth
firmware.py      Python twin of the firmware: frame parser, baseline tracker, 25 features,
                 int8 MLP, probability smoothing, confidence gate, SD logging
model.py         dataset building, training, int8 quantisation, C header export
scenarios.py     chamber protocol, randomised field sessions, six 24 h field days
campaign.py      tests T00-T15 with fixed acceptance criteria
params.py        EVERY assumption in one place - re-fit these with real data
firmware_c/      the same firmware core in portable C99 (no malloc); model_params.h is generated
```

## Run

```bash
python run_all.py            # full campaign (~15 min on 8 cores) -> results/report.html
python run_all.py --quick    # ~4 min, smaller scenario counts
python run_all.py --report-only
python -m unittest discover tests
```

Needs Python 3.10+ with numpy, scipy, scikit-learn, pandas, matplotlib, and gcc on PATH for
the C software-in-the-loop test (T13).

## Using the C firmware on the board

`firmware_c/pms7003.c` and `polso_fw.c` (+ generated `model_params.h`) go into an STM32CubeIDE
project unchanged. Call `fw_uart_rx()` from the PMS7003 UART DMA/RX callback, and `fw_tick()`
once per second after reading the BME688 heater step (parallel mode, profile in `params.py`).
Initialise with `fw_init()` using the zero-air baseline measured for that unit.

## Honest limits

Parameters are literature/datasheet estimates, not PolSo measurements, and training and test
data come from the same simulator. Use the results to find design problems and compare options,
not as a real-world accuracy claim. After the first unit arrives, measure the items in the
report's assumptions table, update `params.py`, and re-run.

## STM32 twin and verification (added)

```
stm32_twin/     the device as it runs on the NUCLEO-G474RE: operation-count cycle model of the real firmware,
                schedule + power-state simulation (UART/DMA, I2C, SD, GPS, timers), model-family study,
                PMS7003 duty-cycling policies, memory/peripheral plan, MEASURE_ON_BOARD.md
verify/         verification toolkit: metrics (EPA PM2.5 targets, F1, calibration, bootstrap), simulated
                reference instruments, virtual co-location rehearsal, simulator-vs-reality gap check, file formats
firmware_c/prof/prof_build.cpp   counts every floating-point operation of the unchanged firmware core
firmware_c/bench/dwt_bench.c     on-board benchmark (DWT cycle counter) that replaces the twin's estimates
firmware_c/fastmath.h            polynomial log/exp used by the optimised variants (-DFW_OPT_FASTMATH)
```

```bash
python run_mcu.py            # STM32 twin (needs run_all.py first)       ~15 min
python run_verification.py   # virtual co-location rehearsal             ~2 min
python build_report.py       # results/stm32_and_verification_report.html
python -m stm32_twin.compare_measured lines.txt   # compare board measurements with the twin
python -m verify.domain_gap sim.csv real.csv       # how far is the simulator from real data?
```

## On-device reporting layer and the STM32 tab (added)

The firmware now also reports what a user should see: a **10-minute dominant source** (majority of the last 20 windows,
"mixed" when no source holds half) and a **corrected 5-minute PM2.5** with a band. The correction uses only the
device's own BME688 humidity and its detected source (`polso_twin/pmcal.py`, table in `firmware_c/pm_cal_params.h`);
it was fitted on training sessions, re-centred and banded on held-out sessions, chosen against new field days,
and the 3D map is the untouched final test (see "About & accuracy" in the viewer). C and Python agree on every window.

The 3D map viewer has three tabs: **3D map**, **Devices**, and **STM32**, the board in 3D with bus activity, power state and live
parameters. It also has a **Guided tour** and an **About & accuracy** panel. Start it with `../START_SIMULATION.bat`; how to present it is in
`../HOW_TO_PRESENT.md`.

## Build tab: constructing the real device (added)

The viewer's fourth tab, **Build**, is the construction sheet: parts list, wiring and pin map, clock / peripheral / DMA /
low-power settings, sensor commands and heater profile, every firmware constant, the 25 classifier features, the PM2.5
correction table, power inputs, memory map and battery life per design lever, the modelled assumptions to measure first,
and step-by-step build, flash and verification. It is generated by `build_spec.py` from `params.py`, `stm32_twin/spec.py`,
the firmware headers and `results/pm_cal.json` (rebuilt on every `python viewer/build_viewer.py`), and is also written to
`../BUILD_GUIDE.md`. `tests/test_build_spec.py` checks that it matches the C headers. Pins and the clock tree are tagged
PROP: proposals to confirm in STM32CubeMX before wiring.

Viewer layout: all panels sit in one CSS grid (header, case bar, main area, tour row, timeline) and each tab owns the main
area, so panels cannot overlap; on phones the side panels become bottom sheets.
