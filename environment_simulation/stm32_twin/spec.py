"""
STM32 NUCLEO-G474RE platform model: clocks, memory, power states and Cortex-M4F instruction costs.

Every number carries a status:
    DOC   documented by ARM / ST (instruction timing, memory sizes, peripheral list)
    EST   engineering estimate with a low / nominal / high range, NOT yet measured on the board
Each EST value is the first thing to replace with a measurement (stm32_twin/MEASURE_ON_BOARD.md).
Ranges are propagated through every result, so conclusions that hold across the whole range are
safe and conclusions that flip inside it are flagged.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Silicon (DOC)  - STM32G474RE, Arm Cortex-M4F with FPv4-SP, DSP instructions, ART accelerator
# ---------------------------------------------------------------------------
F_MAX_MHZ = 170.0
FLASH_KB = 512
SRAM_KB = 128            # 96 KB (SRAM1 80 + SRAM2 16) + 32 KB CCM SRAM
CCM_KB = 32
# Flash wait states at 170 MHz (range 1 boost) - EST, check RM0440 flash-latency table.
FLASH_WAIT_STATES = 8
# ART accelerator (instruction + data cache and prefetch) hides most of the flash latency.
# Cycle inflation of code running from flash relative to zero-wait-state: low / nominal / high.
ART_INFLATION = (1.00, 1.05, 1.25)

# ---------------------------------------------------------------------------
# Cortex-M4F cycle costs per operation: (low, nominal, high)        status: DOC/EST mix
# FPv4-SP: VADD/VSUB/VMUL 1 cycle, VDIV/VSQRT 14 cycles (ARM Cortex-M4 TRM). The compare path
# (VCMP + VMRS + branch) and int<->float moves (VCVT, VMOV) are small but not free.
# libm costs are for newlib-nano on a hard-float M4: EST, measure with the DWT cycle counter.
# ---------------------------------------------------------------------------
COST = {
    "add": (1.0, 1.0, 1.5), "mul": (1.0, 1.0, 1.5), "div": (14.0, 14.0, 16.0),
    "cmp": (2.0, 3.0, 4.0), "cvt": (1.0, 2.0, 3.0), "sqrt": (14.0, 14.0, 18.0),
    "log": (100.0, 150.0, 250.0), "log10": (110.0, 170.0, 280.0), "exp": (90.0, 130.0, 220.0),
    "floor": (12.0, 25.0, 45.0), "fmcall": (6.0, 8.0, 12.0),   # fmcall: bit tricks inside fast log/exp
}
FP_ARITH = ("add", "mul", "div", "cmp", "cvt")
# Extra cycles per FP operation for operand loads/stores, loop control and integer work.
FP_OVERHEAD = (0.7, 1.2, 2.0)

# Integer / control work that the operation counter does not see (cycles).   EST
INT_WORK = {
    "uart_frame_parse": (400, 600, 1000),      # 32-byte state machine + checksum + 12 word copies
    "sample_bookkeeping": (80, 120, 200),      # ring-buffer writes, range checks, step bookkeeping per second
    "window_integer": (2500, 3500, 5000),      # slot indexing, loops over the 60-sample ring (7 passes)
    "post": (60, 100, 200),
}
# int8 MLP 25-32-16-5: multiply-accumulate cost per MAC.
MLP_CYCLES_PER_MAC = {"scalar C": (2.5, 3.5, 5.0), "SMLAD / CMSIS-NN": (0.9, 1.3, 2.0)}
MLP_REQUANT_PER_NEURON = (8.0, 12.0, 18.0)

# ---------------------------------------------------------------------------
# Power at 3.3 V rail, by state - EST (ST quotes ~84 uA/MHz run mode, peripherals off; the
# nominal here adds clocked peripherals). Replace with the Nucleo IDD-jumper (JP6) measurement.
# ---------------------------------------------------------------------------
V_RAIL = 3.3
I_RUN_170_MA = (14.0, 18.0, 30.0)
I_SLEEP_170_MA = (5.0, 8.0, 14.0)         # core clock gated, DMA / UART / I2C clocks on
I_STOP_MA = (0.02, 0.06, 0.15)            # Stop 1 with RTC / LPTIM wake-up, SRAM retained
I_STATIC_MA = (1.0, 1.5, 3.0)             # frequency-independent part of run / sleep current
STOP_WAKE_S = 70e-6                       # Stop->Run incl. PLL relock, spent at run current   EST
NUCLEO_OVERHEAD_MA = (25.0, 45.0, 60.0)   # ST-LINK/V3E, LEDs, regulators on the Nucleo board   EST
# NOTE: the G4 family has Sleep, Stop 0, Stop 1, Standby and Shutdown (no "Stop 2" as on L4/U5).

# ---------------------------------------------------------------------------
# Peripheral timing (DOC from datasheets / protocol maths)
# ---------------------------------------------------------------------------
UART_BAUD_PMS = 9600
UART_BITS_PER_BYTE = 10
PMS_FRAME_BYTES = 32
I2C_HZ = 400_000
BME_READ_BYTES = 17 + 3                   # one parallel-mode field read + address/register bytes
SPI_SD_HZ = 10_000_000
SD_SECTOR_BYTES = 512
# SD write latency model (EST): body lognormal, plus a heavy tail from card garbage collection.
SD_BODY_MEDIAN_S, SD_BODY_SIGMA = 0.004, 0.8
SD_TAIL_P, SD_TAIL_RANGE_S = 0.029, (0.020, 0.250)
SD_BADCARD_P, SD_BADCARD_RANGE_S = 0.001, (0.250, 1.000)


def pick(rng3, which):
    """Select low / nominal / high from a 3-tuple. which in {'low','nom','high'}."""
    return rng3[{"low": 0, "nom": 1, "high": 2}[which]]


def i_run_ma(f_mhz, which="nom"):
    i170, i0 = pick(I_RUN_170_MA, which), pick(I_STATIC_MA, which)
    return i0 + (i170 - i0) * f_mhz / F_MAX_MHZ


def i_sleep_ma(f_mhz, which="nom"):
    i170, i0 = pick(I_SLEEP_170_MA, which), pick(I_STATIC_MA, which) * 0.6
    return i0 + max(i170 - i0, 0) * f_mhz / F_MAX_MHZ


@dataclass(frozen=True)
class Arch:
    """Firmware architecture choices that decide how the MCU spends its time."""
    name: str
    tick: str = "hal1k"            # 'hal1k': HAL SysTick every 1 ms (no Stop)   | 'tickless': LPTIM/RTC wake only
    isr_overhead_cycles: int = 1800  # HAL callback chain ~1.8k; LL/register-level ~0.2k
    sd: str = "blocking"           # 'blocking': SPI polling in the main loop     | 'async': DMA + state machine
    gps: str = "default"           # 'default': ~8 NMEA sentences, ~500 B/s       | 'rmc': one sentence, 70 B/s
    gps_parse_cycles: int = 12000  # per sentence; ~12k with sscanf, ~1.5k with a hand parser
    consumer: str = "main"         # 'main': main loop drains the DMA ring        | 'isr': drained in the IRQ
    pms_ring: int = 128
    gps_ring: int = 256
    f_mhz: float = 170.0
    variant: str = "v0"            # firmware computation variant (profile.VARIANTS)
    mlp_impl: str = "scalar C"
    pms_period_s: float = 1.0      # PMS7003 active-mode frame period (0.2 - 1 s depending on dust change)
