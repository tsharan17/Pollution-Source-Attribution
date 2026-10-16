# Replacing the STM32 twin's estimates with measurements

Everything marked EST in `spec.py` should be measured on the NUCLEO-G474RE in the first week. Order of
value: current draw, cycle counts, stack, SD latency, then everything else.

| What | How | Replaces |
|---|---|---|
| MCU current in Run / Sleep / Stop 1 | Remove jumper JP6 (IDD) on the Nucleo and put a current meter or a power profiler (Nordic PPK2, Otii) across it. Run the benchmark loop at 170 MHz, then `__WFI()` with the SysTick off, then enter Stop 1. | `I_RUN_170_MA`, `I_SLEEP_170_MA`, `I_STOP_MA` |
| Cycles per function and per stage | Flash `firmware_c/bench/dwt_bench.c` (release flags: `-O2`, hard-float, ART on), copy the `BENCH ...` lines from the ST-LINK virtual COM port, run `python -m stm32_twin.compare_measured lines.txt`. | `COST`, `INT_WORK`, `MLP_CYCLES_PER_MAC` |
| Flash and RAM used | From the linker `.map` / `arm-none-eabi-size`. | `platform_map.memory_budget` |
| Stack depth | Paint the stack with a known pattern at start-up and read the high-water mark after an hour of running with GPS and SD logging on. | the 2-4 KB stack allowance |
| SD write latency tail | Time every `f_write` + `f_sync` with the DWT counter for 24 h on the actual card; keep the histogram. | `SD_*` distribution |
| UART ring overruns | Count HAL `ErrorCode` overrun flags and lost bytes on the GPS and PMS ports over 24 h. | ring-overflow estimate |
| Wake-up time from Stop 1 | Toggle a GPIO on wake and on first instruction after the PLL is locked; measure with a scope or logic analyser. | `STOP_WAKE_S` |
| PMS7003 frame period | Log frame timestamps for an hour in clean air and in smoke. | `pms_period_s` |
| BME688 read time and heater step | Scope SCL during a parallel-mode read; log the step sequence. | `BME_READ_BYTES`, step timing |

After measuring, edit the `(low, nominal, high)` tuples in `spec.py` (put the measurement in the nominal slot and
keep a spread for unit-to-unit and temperature variation) and re-run `python run_mcu.py`.

A model is considered **validated** when every measured value falls inside the predicted low-high range. If a
measurement falls outside, the report's conclusions that depended on it are the first to re-check.

Datasheet checks still open (not confirmed from the ST datasheet when the twin was written): run-mode current at
170 MHz (ST quotes roughly 84 uA/MHz with peripherals off), Stop 1 current, flash wait states at 170 MHz in boost
mode, DMA access to the CCM SRAM. The G4 family has Sleep, Stop 0, Stop 1, Standby and Shutdown; there is no
"Stop 2" as on the L4 and U5 parts.
