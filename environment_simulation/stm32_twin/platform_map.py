"""Memory budget, peripheral allocation and clock plan for the NUCLEO-G474RE build (proposal + estimates)."""
import os
import subprocess

from . import spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW = os.path.join(ROOT, "firmware_c")


def host_object_bytes():
    """Code size of the firmware core compiled for the PC (-Os). Thumb-2 is typically 0.55-0.9x of x86-64."""
    try:
        total = 0
        for f in ("polso_fw.c", "pms7003.c"):
            obj = os.path.join(FW, "_size.o")
            subprocess.run(["gcc", "-Os", "-c", "-std=c99", "-fno-asynchronous-unwind-tables", f, "-o", obj],
                           cwd=FW, check=True, capture_output=True)
            out = subprocess.run(["size", obj], capture_output=True, text=True, check=True).stdout.split("\n")[1].split()
            total += int(out[0])
            os.remove(obj)
        return total
    except Exception:
        return 7000


def memory_budget(state_bytes, weight_bytes):
    core = host_object_bytes()
    kb = 1024.0
    flash = {                                   # (low, nominal, high) bytes
        "Firmware core (features, MLP, parser), Thumb-2 -Os": (core * 0.55, core * 0.7, core * 0.9),
        "Model weights + scalers (const, int8)": (weight_bytes,) * 3,
        "libm (logf, expf, log10f, sqrtf, floorf), newlib-nano": (3 * kb, 4.5 * kb, 6 * kb),
        "Peripheral drivers (LL ~7 KB ... HAL ~30 KB)": (7 * kb, 15 * kb, 30 * kb),
        "BME688 driver (Bosch BME68x API, no BSEC)": (6 * kb, 8 * kb, 10 * kb),
        "FatFs + SD-over-SPI glue": (9 * kb, 12 * kb, 16 * kb),
        "printf / sscanf (newlib-nano; float support if used)": (2 * kb, 12 * kb, 20 * kb),
        "Startup, vector table, system init": (1 * kb, 1.5 * kb, 2.5 * kb),
    }
    ram = {
        "Firmware state (ring buffers, baseline, EMA)": (state_bytes,) * 3,
        "UART DMA rings (PMS 128 B, GPS 256 B)": (384, 384, 768),
        "BME688 + I2C buffers": (64, 128, 256),
        "FatFs objects + sector buffer": (1.5 * kb, 2.2 * kb, 3.2 * kb),
        "Log line + GPS parse buffers": (256, 384, 640),
        "Driver handles / .bss": (0.5 * kb, 1.2 * kb, 2.5 * kb),
        "Main stack (deepest path ~1 KB) + ISR nesting": (2 * kb, 3 * kb, 4 * kb),
    }
    sums = lambda d: tuple(sum(v[i] for v in d.values()) for i in range(3))
    return dict(flash=flash, ram=ram, flash_total=sums(flash), ram_total=sums(ram),
                flash_cap=spec.FLASH_KB * kb, ram_cap=spec.SRAM_KB * kb, core_host_bytes=core)


CLOCK_PLAN = [
    ("System clock", "HSI16 -> PLL (M=4, N=85, R=2) = 170 MHz, voltage range 1 boost"),
    ("Flash", "ART accelerator on (instruction cache + prefetch), 8 wait states at 170 MHz (check RM0440)"),
    ("Low-power clock", "LSE 32.768 kHz (or LSI) for the 1 Hz LPTIM tick / RTC wake-up and the IWDG"),
    ("Idle strategy", "Race to idle: do the work at 170 MHz in short bursts, then Stop 1 until the next event"),
    ("Not used", "CORDIC and FMAC accelerators (the workload needs only 4 square roots per window)"),
]

# proposal; confirm pin availability against UM2505 (NUCLEO-G474RE) and CubeMX before wiring
PERIPHERALS = [
    ("PMS7003 particle sensor", "USART1", "9600 8N1, RX + TX (commands)", "DMA circular 128 B + IDLE-line IRQ", 1),
    ("GPS receiver", "USART3", "9600 (raise to 38400 after configuring RMC-only)", "DMA circular 256-512 B + IDLE IRQ", 2),
    ("BME688 gas / T / RH", "I2C1", "400 kHz, 3.3 V", "DMA read, transfer-complete IRQ", 2),
    ("microSD card", "SPI2", "10 MHz, mode 0", "DMA TX/RX, non-blocking FatFs state machine", 3),
    ("Battery monitor", "ADC1 + divider", "oversampling x16, triggered once per second", "polled", 5),
    ("1 Hz sample tick", "LPTIM1 (LSE)", "wake-up from Stop 1", "-", 4),
    ("Watchdog", "IWDG", "4 s timeout, kicked from the 1 Hz task", "-", None),
    ("Debug console", "USART2 (ST-LINK VCP)", "115200", "TX only, polled", 6),
]
IRQ_NOTE = "NVIC priority: smaller number = more urgent. Keep SysTick at the lowest priority (or run tickless)."


def task_catalogue(sim, cyc, arch):
    """Tasks the CPU runs, rate and cost, from the schedule simulation of one architecture."""
    f = arch.f_mhz
    rows = [
        ("PMS7003 frame ISR + parse", f"{1 / arch.pms_period_s:.1f} /s", sim["tasks_us"]["pms_frame"]),
        ("GPS sentence ISR + parse", "8 /s" if arch.gps == "default" else "1 /s", sim["tasks_us"]["gps_sentence"]),
        ("BME688 I2C read", "1 /s", (2 * arch.isr_overhead_cycles + 1500) / f),
        ("1 Hz sample task (compensation, baseline)", "1 /s", sim["tasks_us"]["sample"]),
        ("Battery ADC", "1 /s", (arch.isr_overhead_cycles + 1500) / f),
        ("Window task (features + MLP + gate)", "1 / 30 s", sim["tasks_us"]["window"]),
        ("SD log write (FatFs CPU part)", "1 / 30 s", sim["tasks_us"]["sd_cpu"]),
    ]
    return rows
