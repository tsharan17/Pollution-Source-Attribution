"""
Python twin of the STM32 firmware (firmware_c/). Same algorithm, same constants:

  UART bytes -> PMS7003 frame parser -> 1 Hz sample store
  BME688 R   -> humidity/temperature compensation -> clean-air baseline tracker
  every 30 s -> 60 s window features -> int8 MLP -> probability EMA -> confidence gate

The C version is checked against this one in the software-in-the-loop test (T13).
"""
import math
import numpy as np

from . import params as P

GROUP_OF_STEP = np.array([int(np.where(P.HEATER_GROUPS == t)[0][0]) for t in P.HEATER_PROFILE])
GAMMA_STEP = P.mox_humidity_gamma(P.HEATER_PROFILE)


# ---------------------------------------------------------------------------
# PMS7003 frame parser (byte-wise state machine, identical to pms7003.c)
# ---------------------------------------------------------------------------
class PMSParser:
    def __init__(self):
        self.buf = bytearray()
        self.frames_ok = 0
        self.checksum_errors = 0

    def feed(self, data: bytes):
        """Feed raw UART bytes; returns list of decoded frames (cf1, atm, cum)."""
        out = []
        for byte in data:
            n = len(self.buf)
            if n == 0:
                if byte == 0x42:
                    self.buf.append(byte)
                continue
            if n == 1:
                if byte == 0x4D:
                    self.buf.append(byte)
                elif byte != 0x42:
                    self.buf.clear()
                continue
            self.buf.append(byte)
            if n + 1 == 4 and (self.buf[2] << 8 | self.buf[3]) != 28:
                self.checksum_errors += 1           # bad length: resync
                self.buf.clear()
                continue
            if n + 1 == 32:
                cs = sum(self.buf[:30]) & 0xFFFF
                if cs == (self.buf[30] << 8 | self.buf[31]):
                    w = [self.buf[k] << 8 | self.buf[k + 1] for k in range(4, 28, 2)]
                    out.append((w[0:3], w[3:6], w[6:12]))
                    self.frames_ok += 1
                else:
                    self.checksum_errors += 1
                self.buf.clear()
        return out


def decode_frames_fast(matrix, present):
    """Vectorised decode of well-aligned frames (one 32-byte frame per second)."""
    m = matrix.astype(np.uint32)
    words = (m[:, 2:30:2] << 8) | m[:, 3:30:2]
    cs = m[:, :30].sum(1) & 0xFFFF
    ok = present & (m[:, 0] == 0x42) & (m[:, 1] == 0x4D) & (words[:, 0] == 28) & \
         (cs == ((m[:, 30] << 8) | m[:, 31]))
    return words[:, 1:4].astype(float), words[:, 7:13].astype(float), ok


def abs_humidity_fw(t_c, rh):
    return 216.7 * (rh / 100.0 * 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))) / (273.15 + t_c)


# ---------------------------------------------------------------------------
# Feature extraction for one window (indices a..b-1)
# ---------------------------------------------------------------------------
def window_features(cum, cf1, pm_ok, r, step, gas_ok, t_meas, rh_meas, a, b):
    f = np.zeros(P.N_FEATURES)
    pv = pm_ok[a:b]
    c = cum[a:b][pv]
    pm = cf1[a:b][pv]
    # ---- PM ----
    d = np.empty_like(c)
    d[:, :5] = np.clip(c[:, :5] - c[:, 1:], 0, None)
    d[:, 5] = c[:, 5]
    dm = d.mean(0)
    s = dm.sum()
    f[0:6] = np.log10(dm / s + 1e-4) if s > 0 else -4.0
    c0 = c[:, 0]
    m0 = c0.mean()
    f[6] = math.log10(1.0 + m0)
    pm1, pm25, pm10 = pm.mean(0)
    f[7] = (pm25 + 1.0) / (pm10 + 1.0)
    f[8] = (pm1 + 1.0) / (pm25 + 1.0)
    var = np.mean((c0 - m0) ** 2)
    sd = math.sqrt(var)
    f[9] = sd / (m0 + 1.0)
    f[10] = (np.clip(np.mean((c0 - m0) ** 3) / var ** 1.5, -10, 10)
             if var > 1e-6 * (m0 + 1.0) ** 2 else 0.0)
    lc = np.log10(1.0 + cum[a:b, 0])
    okp = pm_ok[a:b]
    rise = [lc[j] - lc[j - 5] for j in range(5, b - a) if okp[j] and okp[j - 5]]
    f[11] = max(rise) if rise else 0.0
    # ---- gas ----
    gv = gas_ok[a:b]
    rr = r[a:b][gv]
    grp = GROUP_OF_STEP[step[a:b][gv]]
    tt = (np.arange(a, b)[gv] - a) / 60.0
    rg = np.array([rr[grp == k].mean() for k in range(len(P.HEATER_GROUPS))])
    M = rg.mean()
    f[12] = M
    f[13:19] = rg - M
    for col, k in ((19, 3), (20, 1)):               # 300 C and 200 C groups
        x, y = tt[grp == k], rr[grp == k]
        if len(x) >= 2:
            xm = x.mean()
            den = np.sum((x - xm) ** 2)
            f[col] = np.sum((x - xm) * (y - y.mean())) / den if den > 0 else 0.0
    sec = np.arange(a, b)[gv][grp == 3]
    keep = pm_ok[sec]
    x = np.log10(1.0 + cum[sec[keep], 0])
    y = -rr[grp == 3][keep]
    if len(x) >= 3:
        sx, sy = x.std(), y.std()
        if sx > 1e-6 and sy > 1e-6:
            f[21] = np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)
    f[22] = -rg[3] / (0.5 + math.log10(1.0 + pm25))
    # ---- context ----
    f[23] = rh_meas[a:b][gv].mean() / 100.0
    f[24] = t_meas[a:b][gv].mean() / 50.0
    return f


# ---------------------------------------------------------------------------
# int8 quantised MLP (integer arithmetic identical to model.c)
# ---------------------------------------------------------------------------
def round_half_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def qmodel_forward(qm, X):
    """X (N,25) float -> (probabilities (N,5), int8 input codes). Integer path bit-exact with C."""
    X = np.atleast_2d(X).astype(np.float32)
    xs = (X - qm["mu"]) * qm["inv_sd"]
    q = np.clip(round_half_away(xs * qm["inv_s0"]), -127, 127).astype(np.int64)
    q0 = q.copy()
    for L in qm["layers"][:-1]:
        acc = q @ L["W"].T.astype(np.int64) + L["b"].astype(np.int64)
        acc = np.clip(acc, 0, None)
        q = np.clip((acc * L["m"] + (1 << (L["sh"] - 1))) >> L["sh"], 0, 127)
    L = qm["layers"][-1]
    acc = q @ L["W"].T.astype(np.int64) + L["b"].astype(np.int64)
    logits = acc.astype(np.float64) * L["scale"]
    logits -= logits.max(1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True), q0


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def run_device(streams, qm, conf_thr=None, use_ema=True, track_baseline=True,
               window=P.FW_WINDOW_S, hop=P.FW_HOP_S):
    """Run the firmware over a whole stream. Returns a dict of per-window arrays."""
    if conf_thr is None:
        conf_thr = qm["conf_thr"] if qm is not None else 1.0
    n = len(streams["gas_r"])

    # ---- PMS ----
    if "uart_bytes" in streams:
        parser = PMSParser()
        cf1 = np.zeros((n, 3))
        cum = np.zeros((n, 6))
        pm_ok = np.zeros(n, bool)
        for i, chunk in enumerate(streams["uart_bytes"]):
            fr = parser.feed(chunk)
            if fr:
                cf1[i], _, cum[i] = fr[-1][0], fr[-1][1], fr[-1][2]
                pm_ok[i] = True
        pms_errors = parser.checksum_errors
    else:
        cf1, cum, pm_ok = decode_frames_fast(streams["uart"], streams["uart_present"])
        pms_errors = 0

    # ---- gas compensation + baseline tracker ----
    R = streams["gas_r"].astype(float)
    step = streams["gas_step"].astype(int)
    t_meas = streams["t_meas"].astype(float)
    rh_meas = streams["rh_meas"].astype(float)
    gas_ok = streams["gas_ok"] & (R >= P.MOX_R_MIN) & (R <= P.MOX_R_MAX)
    ah = abs_humidity_fw(t_meas, rh_meas)
    lcomp = (np.log(np.clip(R, 1e-9, None)) + GAMMA_STEP[step] * (ah - P.AH_REF)
             + P.MOX_AMB_TEMP_COEF * ((t_meas - P.BME_SELF_HEAT) - 25.0))
    init = streams.get("b_tracked", streams["b0"]) if track_baseline else streams["b0"]
    base = np.array(init, float).copy()
    r = np.zeros(n)
    up, down, warm = P.FW_BASE_UP, P.FW_BASE_DOWN, P.FW_WARMUP_S
    lc_list, st_list, ok_list = lcomp.tolist(), step.tolist(), gas_ok.tolist()
    for i in range(n):
        if not ok_list[i]:
            continue
        s, l = st_list[i], lc_list[i]
        bs = base[s]
        r[i] = l - bs
        if track_baseline and i >= warm:
            base[s] = bs + up * (l - bs) if l > bs else bs - down

    # ---- windows ----
    ends = np.arange(window - 1, n, hop)
    feats = np.zeros((len(ends), P.N_FEATURES))
    status = np.empty(len(ends), object)
    for w, e in enumerate(ends):
        a, b = e - window + 1, e + 1
        if e < warm:
            status[w] = P.WARMUP
            continue
        gv = gas_ok[a:b]
        groups_seen = np.unique(GROUP_OF_STEP[step[a:b][gv]]).size
        if pm_ok[a:b].mean() < P.FW_PM_VALID_FRAC or gv.mean() < P.FW_GAS_VALID_FRAC \
                or groups_seen < len(P.HEATER_GROUPS):
            status[w] = P.FAULT
            continue
        status[w] = "ok"
        feats[w] = window_features(cum, cf1, pm_ok, r, step, gas_ok, t_meas, rh_meas, a, b)

    valid = status == "ok"
    probs_raw = np.zeros((len(ends), len(P.CLASSES)))
    if valid.any() and qm is not None:
        probs_raw[valid] = qmodel_forward(qm, feats[valid])[0]
    probs = probs_raw.copy()
    labels = np.empty(len(ends), object)
    prev_ok = False
    for w in range(len(ends)):
        if not valid[w]:
            labels[w] = status[w]
            prev_ok = False
            continue
        if use_ema and prev_ok:
            probs[w] = P.FW_PROB_EMA * probs[w - 1] + (1 - P.FW_PROB_EMA) * probs_raw[w]
        prev_ok = True
        k = int(np.argmax(probs[w]))
        labels[w] = P.CLASSES[k] if probs[w, k] >= conf_thr else P.UNCERTAIN

    # ---- SD logging with write failures and a 16-record retry buffer ----
    rng = np.random.default_rng(streams.get("seed", 0) + 31)
    pending, lost, written = 0, 0, 0
    for w in range(len(ends)):
        pending += 1
        if rng.random() >= streams.get("sd_fail", 0.0):
            written += pending
            pending = 0
        elif pending > 16:
            lost += pending - 16
            pending = 16
    return dict(t_end=ends, features=feats, probs=probs, probs_raw=probs_raw, label=labels,
                status=status, pms_errors=pms_errors, sd_written=written, sd_lost=lost,
                sd_pending=pending, pm_ok=pm_ok, gas_ok=gas_ok, baseline_final=base,
                cum=cum, r=r)
