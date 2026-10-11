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
