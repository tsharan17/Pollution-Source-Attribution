"""
Real-time schedule and power model of the STM32G474 running the whole device.

Activities generated for every second of operation:
  PMS7003 UART frames (DMA), GPS NMEA sentences (DMA), BME688 I2C reads, the 1 Hz sample task,
  battery ADC, the 30 s window task (features + MLP), SD-card logging, and the HAL 1 ms SysTick.
Each activity has CPU time (cycles / clock) and/or peripheral-busy time (DMA moving bytes, which
needs the bus clocks running, i.e. Sleep not Stop). From the merged timeline the model derives the
time in Run / Sleep / Stop, the current drawn, ISR and ring-buffer behaviour, and the CPU load.
"""
import numpy as np

from . import spec
from .spec import Arch

SENTENCES_DEFAULT = [72, 50, 60, 66, 66, 66, 70, 40]   # GGA GLL GSA GSV x3 RMC VTG (bytes)
SENTENCES_RMC = [70]
BYTE_S = spec.UART_BITS_PER_BYTE / spec.UART_BAUD_PMS    # 1.04 ms per byte at 9600 baud
SYSTICK_CYCLES = 400
MIN_STOP_GAP_S = 300e-6
ADC_CYCLES = 1500
BME_CYCLES = 1500
SD_CPU_CYCLES = 60_000


def merge(iv):
    iv = np.asarray(iv, float).reshape(-1, 2)
    if len(iv) == 0:
        return iv
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    ce = np.maximum.accumulate(iv[:, 1])
    new = np.r_[True, iv[1:, 0] > ce[:-1]]
    idx = np.nonzero(new)[0]
    ends = np.r_[ce[idx[1:] - 1], ce[-1]]
    return np.column_stack([iv[idx, 0], ends])


def sd_stalls(rng, n):
    """Latency of one SD-card write (s): lognormal body, garbage-collection tail, rare bad card."""
    body = spec.SD_BODY_MEDIAN_S * np.exp(spec.SD_BODY_SIGMA * rng.standard_normal(n))
    u = rng.random(n)
    tail = rng.uniform(*spec.SD_TAIL_RANGE_S, n)
    bad = rng.uniform(*spec.SD_BADCARD_RANGE_S, n)
    return np.where(u < spec.SD_BADCARD_P, bad, np.where(u < spec.SD_BADCARD_P + spec.SD_TAIL_P, tail, body))


def build_timeline(arch: Arch, cyc, T, rng, which="nom"):
    """CPU and peripheral intervals over T seconds. cyc: stage cycles per call (profile.stage_cycles)."""
    f = arch.f_mhz * 1e6
    isr = arch.isr_overhead_cycles
    cpu, per, ev = [], [], []          # ev: (lane, start, end) for the timeline plot

    def add_cpu(starts, cycles, lane):
        starts = np.atleast_1d(starts)
        d = np.full(len(starts), cycles / f) if np.isscalar(cycles) else cycles / f
        cpu.append(np.column_stack([starts, starts + d]))
        ev.append((lane, starts, starts + d))

    def add_per(starts, dur, lane):
        starts = np.atleast_1d(starts)
        d = np.full(len(starts), dur) if np.isscalar(dur) else dur
        per.append(np.column_stack([starts, starts + d]))
        ev.append((lane, starts, starts + d))

    # --- PMS7003 frames ---
    n_pms = int(T / (arch.pms_period_s * 0.9)) + 2
    gaps = arch.pms_period_s * (1 + 0.1 * rng.uniform(-1, 1, n_pms))
    t_pms = np.cumsum(gaps) + rng.uniform(0, 1)
    t_pms = t_pms[t_pms < T - 0.05]
    d_pms = spec.PMS_FRAME_BYTES * BYTE_S
    add_per(t_pms, d_pms, "PMS7003 UART+DMA")
    add_cpu(t_pms + d_pms, isr + cyc["uart_frame"], "PMS ISR + parse")

    # --- GPS NMEA sentences ---
    sent = SENTENCES_DEFAULT if arch.gps == "default" else SENTENCES_RMC
    sec = np.arange(int(T))
    phase = rng.uniform(0, 0.4, len(sec))
    t0 = sec + phase
    lens = np.array(sent)
    offs = np.concatenate([[0], np.cumsum(lens * BYTE_S)[:-1]])
    st = (t0[:, None] + offs[None, :]).ravel()
    du = np.tile(lens * BYTE_S, len(sec))
    add_per(st, du, "GPS UART+DMA")
    add_cpu(st + du, isr + arch.gps_parse_cycles, "GPS ISR + parse")

    # --- BME688 I2C, 1 Hz ---
    t_bme = sec + 0.20
    add_per(t_bme, (spec.BME_READ_BYTES * 9) / spec.I2C_HZ, "BME688 I2C")
    add_cpu(t_bme, 2 * isr + BME_CYCLES, "I2C ISR")

    # --- 1 Hz sample task, ADC ---
    add_cpu(sec + 0.90, isr + cyc["sample"], "1 Hz sample task")
    add_cpu(sec + 0.95, isr + ADC_CYCLES, "ADC")

    # --- 30 s window task ---
    t_win = np.arange(30, int(T), 30) - 0.05
    add_cpu(t_win, cyc["wcheck"] + cyc["features"] + cyc["mlp"] + cyc["post"], "Window: features+MLP")

    # --- SD logging ---
    st_sd = t_win + 0.02
    stall = sd_stalls(rng, len(st_sd))
    add_cpu(st_sd, SD_CPU_CYCLES, "FatFs CPU")
    if arch.sd == "blocking":
        add_cpu(st_sd + SD_CPU_CYCLES / f, np.maximum(stall - SD_CPU_CYCLES / f, 0) * f, "SD busy-wait")
    else:
        add_per(st_sd + SD_CPU_CYCLES / f, stall, "SD SPI+DMA")
    return merge(np.vstack(cpu)), merge(np.vstack(per)), ev, stall


def power_stats(arch: Arch, cpu, per, T, which="nom"):
    f = arch.f_mhz
    run_cpu = float((cpu[:, 1] - cpu[:, 0]).sum())
    allu = merge(np.vstack([cpu, per]))
    busy = float((allu[:, 1] - allu[:, 0]).sum())
    edges = np.r_[0.0, allu[:, 1]]
    nxt = np.r_[allu[:, 0], T]
    gaps = np.maximum(nxt - edges, 0.0)
    i_run, i_sl, i_stop = spec.i_run_ma(f, which), spec.i_sleep_ma(f, which), spec.pick(spec.I_STOP_MA, which)
    if arch.tick == "tickless":
        big = gaps >= MIN_STOP_GAP_S
        n_exit = int(big.sum())
        wake = n_exit * spec.STOP_WAKE_S
        stop_t = float(gaps[big].sum()) - wake
        run_t = run_cpu + wake
        sleep_t = busy - run_cpu + float(gaps[~big].sum())
        systick_t = 0.0
    else:
        n_exit = 0
        systick_t = T * 1000 * SYSTICK_CYCLES / (f * 1e6)
        run_t = run_cpu + systick_t
        stop_t = 0.0
        sleep_t = T - run_t
    sleep_t = max(sleep_t, 0.0)
    charge = (run_t * i_run + sleep_t * i_sl + stop_t * i_stop) / T      # mA average at 3.3 V
    return dict(run=run_t / T, sleep=sleep_t / T, stop=stop_t / T, avg_ma=charge, run_ms_per_s=run_t / T * 1e3,
                cpu_app=run_cpu / T, systick=systick_t / T, stop_exits_per_s=n_exit / T,
                i_run=i_run, i_sleep=i_sl, i_stop=i_stop)


def ring_overflow(arch: Arch, n=400_000, seed=11):
    """Probability that one blocking SD write overruns a UART DMA ring that the main loop drains."""
    rng = np.random.default_rng(seed)
    stall = sd_stalls(rng, n)
    s0 = rng.uniform(0, 1, n)                      # stall start, phase within the second
    out = {}

    def overlap_bursts(period, burst_len, offsets):
        # bytes that arrive during [s0, s0 + stall] from bursts of length burst_len starting at offsets + k*period
        tot = np.zeros(n)
        for k in range(-1, int(np.ceil(1.0 / period)) + 3):
            b0 = offsets + k * period
            tot += np.clip(np.minimum(s0 + stall, b0 + burst_len) - np.maximum(s0, b0), 0, None)
        return tot / BYTE_S

    sent = SENTENCES_DEFAULT if arch.gps == "default" else SENTENCES_RMC
    gps_len = sum(sent) * BYTE_S
    gps_bytes = overlap_bursts(1.0, gps_len, rng.uniform(0, 0.4, n))
    pms_bytes = overlap_bursts(arch.pms_period_s, spec.PMS_FRAME_BYTES * BYTE_S, rng.uniform(0, arch.pms_period_s, n))
    blocking_main = arch.sd == "blocking" and arch.consumer == "main"
    for name, b, ring in (("gps", gps_bytes, arch.gps_ring), ("pms", pms_bytes, arch.pms_ring)):
        p = float((b > ring).mean()) if blocking_main else 0.0
        out[name] = dict(p_per_write=p, per_hour=p * 120)
    out["max_stall_ms"] = float(stall.max() * 1e3)
    out["p99_stall_ms"] = float(np.quantile(stall, 0.99) * 1e3)
    out["p999_stall_ms"] = float(np.quantile(stall, 0.999) * 1e3)
    return out


def simulate(arch: Arch, cyc, T=3600, seed=3, which="nom", with_events=False):
    rng = np.random.default_rng(seed)
    cpu, per, ev, stall = build_timeline(arch, cyc, T, rng, which)
    st = power_stats(arch, cpu, per, T, which)
    st["rings"] = ring_overflow(arch)
    st["arch"] = arch
    st["tasks_us"] = dict(
        sample=(cyc["sample"] + arch.isr_overhead_cycles) / (arch.f_mhz),
        window=(cyc["wcheck"] + cyc["features"] + cyc["mlp"] + cyc["post"]) / arch.f_mhz,
        pms_frame=(arch.isr_overhead_cycles + cyc["uart_frame"]) / arch.f_mhz,
        gps_sentence=(arch.isr_overhead_cycles + arch.gps_parse_cycles) / arch.f_mhz,
        sd_cpu=SD_CPU_CYCLES / arch.f_mhz)
    if with_events:
        st["events"] = ev
        st["cpu"], st["per"] = cpu, per
    return st


def state_segments(arch: Arch, cpu, per, t0, t1):
    """(start, end, state) segments clipped to [t0, t1]; state 0 Run, 1 Sleep, 2 Stop."""
    def clip(iv):
        iv = iv[(iv[:, 1] > t0) & (iv[:, 0] < t1)]
        return np.column_stack([np.maximum(iv[:, 0], t0), np.minimum(iv[:, 1], t1)])
    c, p = clip(cpu), clip(per)
    allu = merge(np.vstack([c, p])) if len(c) + len(p) else np.zeros((0, 2))
    segs = []
    cur = t0
    for s, e in allu:
        if s > cur:
            segs.append((cur, s, 2 if (arch.tick == "tickless" and s - cur >= MIN_STOP_GAP_S) else 1))
        inner = c[(c[:, 0] >= s - 1e-12) & (c[:, 1] <= e + 1e-12)]
        x = s
        for cs, ce in inner:
            if cs > x:
                segs.append((x, cs, 1))
            segs.append((cs, ce, 0))
            x = ce
        if e > x:
            segs.append((x, e, 1))
        cur = e
    if t1 > cur:
        segs.append((cur, t1, 2 if (arch.tick == "tickless" and t1 - cur >= MIN_STOP_GAP_S) else 1))
    return segs


def current_trace(arch: Arch, cpu, per, t0, t1, res=50e-6, which="nom"):
    t = np.arange(t0, t1, res)
    i = np.empty(len(t))
    segs = state_segments(arch, cpu, per, t0, t1)
    cur = {0: spec.i_run_ma(arch.f_mhz, which), 1: spec.i_sleep_ma(arch.f_mhz, which),
           2: spec.pick(spec.I_STOP_MA, which)}
    for s, e, k in segs:
        i[(t >= s) & (t < e)] = cur[k]
    if arch.tick == "hal1k":                         # SysTick wakes the core every millisecond
        tick_w = SYSTICK_CYCLES / (arch.f_mhz * 1e6)
        ph = np.mod(t, 1e-3)
        i[(ph < max(tick_w, res)) & (i < cur[0])] = cur[0]
    return t, i
