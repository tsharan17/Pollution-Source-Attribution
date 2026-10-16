"""
Sensor models: PMS7003 optical particle counter and BME688 MOX gas / T / RH sensor.

Both models produce exactly what the real parts hand to the MCU: PMS7003 32-byte UART
frames (counts per 0.1 L, uint16 fields, checksum) and BME688 gas resistance in ohms
for the current heater step, plus temperature / humidity / pressure.
"""
import numpy as np
from scipy.special import ndtr

from . import params as P
from .physics import (kappa_growth, enclosure_loss, first_order, rel_humidity,
                      angle_diff, mode_volume_um3)


# ---------------------------------------------------------------------------
# Per-prototype variation
# ---------------------------------------------------------------------------
class Unit:
    """One physical prototype: fixed manufacturing spread of both sensors."""

    def __init__(self, unit_id: int):
        rng = np.random.default_rng(10_000 + unit_id)
        self.unit_id = unit_id
        self.pms_gain = rng.lognormal(0, P.PMS_UNIT_GAIN_SIGMA)
        self.pms_size = rng.lognormal(0, P.PMS_UNIT_SIZE_SIGMA)
        self.mox_rref = P.MOX_R_REF * rng.lognormal(0, P.UNIT_MOX_RREF_SIGMA)
        self.mox_sens = {g: rng.lognormal(0, P.UNIT_MOX_SENS_SIGMA) for g in P.GASES}
        self.mox_tpk = {g: rng.normal(0, P.UNIT_MOX_TPK_SD) for g in P.GASES}
        self.mox_gamma = rng.lognormal(0, P.UNIT_MOX_GAMMA_SIGMA)
        self.self_heat = P.BME_SELF_HEAT + rng.normal(0, P.BME_SELF_HEAT_UNIT_SD)
        self.rh_bias = rng.normal(0, P.BME_RH_UNIT_SD)
        self._rw = np.concatenate([[0.0], np.cumsum(rng.normal(0, 1, 400))]) * P.MOX_RW_PER_SQRT_DAY
        self.heater_phase = int(rng.integers(0, len(P.HEATER_PROFILE)))

    def mox_drift(self, age_days):
        return P.MOX_DRIFT_PER_DAY * age_days + np.interp(age_days, np.arange(len(self._rw)), self._rw)

    def factory_baseline(self):
        """Clean-air ln R per heater step measured at day 0 at reference conditions.

        This is what the firmware stores in flash after a factory/zero-air calibration.
        """
        return np.log(self.mox_rref) + P.MOX_R_TEMP_COEF * (320.0 - P.HEATER_PROFILE)


# ---------------------------------------------------------------------------
# PMS7003
# ---------------------------------------------------------------------------
_D = np.logspace(np.log10(0.02), np.log10(40.0), 160)
_LND = np.log(_D)
_DLN = _LND[1] - _LND[0]


def pms_counting_efficiency(d_opt):
    low = 1.0 / (1.0 + np.exp(-(d_opt - P.PMS_CE_LOW_D50) / P.PMS_CE_LOW_W))
    high = 1.0 / (1.0 + (d_opt / P.PMS_CE_HIGH_D50) ** P.PMS_CE_HIGH_P)
    return low * high


def _channel_response(mode, rh, u, theta, enclosure, size_factor):
    """Expected detected particles above each PMS threshold, per particle of this mode.

    Returns (T, 6). RH, wind and direction are quantised and cached so a day of data
    costs only a few thousand distinct evaluations.
    """
    gmd, gsd, _, _, kappa, ri = mode
    s = np.log(gsd)
    w = np.exp(-0.5 * ((_LND - np.log(gmd)) / s) ** 2) / (s * np.sqrt(2 * np.pi)) * _DLN
    rq = np.round(np.clip(rh, 0, 99.5) * 2) / 2
    uq = np.round(np.clip(u, 0, 12) * 4) / 4
    tq = np.where(uq == 0, 0.0, (np.round(theta / 15.0) * 15.0) % 360)
    keys = np.stack([rq, uq, tq], 1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.empty((len(uniq), 6))
    lthr = np.log(P.PMS_THRESHOLDS)
    for a in range(0, len(uniq), 1500):
        k = uniq[a:a + 1500]
        gf = kappa_growth(kappa, k[:, 0])
        dwet = _D[None, :] * gf[:, None]
        dopt = dwet * ri * size_factor
        eta = 1.0 - enclosure_loss(dwet, k[:, 1:2], k[:, 2:3], enclosure)
        det = w[None, :] * pms_counting_efficiency(dopt) * eta
        pgt = 1.0 - ndtr((lthr[None, None, :] - np.log(dopt)[:, :, None]) / P.PMS_SIZING_SIGMA)
        out[a:a + 1500] = np.einsum("kg,kgj->kj", det, pgt)
    return out[inv.ravel()]


def _mass_from_diff(diff_per01l):
    """PMS-firmware-style mass (ug/m3) from differential counts per 0.1 L."""
    v = np.pi / 6.0 * P.PMS_BIN_MID ** 3 * P.PMS_RHO_ASSUMED
    m = diff_per01l / 100.0 * v                       # (T,6) ug/m3 per channel
    return np.stack([m[:, :2].sum(1), m[:, :3].sum(1), m[:, :5].sum(1)], 1)


_KCAL = None


def pms_calibration_factor():
    """Factory scaling so background aerosol at 40 % RH reads its true PM2.5."""
    global _KCAL
    if _KCAL is None:
        from .physics import subtype_number_per_pm25
        npp = subtype_number_per_pm25(P.BACKGROUND_MODES)
        from .physics import abs_humidity
        rh_cal = np.array([float(rel_humidity(25.0 + P.PMS_SAMPLE_HEAT_C, abs_humidity(25.0, 40.0)))])
        cum = sum(k * _channel_response(m, rh_cal, np.array([0.0]), np.array([0.0]),
                                        "hooded", 1.0)
                  for m, k in zip(P.BACKGROUND_MODES, npp))
        diff = cum - np.concatenate([cum[:, 1:], np.zeros((1, 1))], 1)
        _KCAL = 1.0 / _mass_from_diff(diff * 100.0)[0, 1]
    return _KCAL


def simulate_pms(env, unit: Unit, age_days, enclosure, rng, inlet_facing=0.0):
    """Returns cumulative counts per 0.1 L (T,6) and PM1/2.5/10 CF=1 and ATM (T,3), all uint16."""
    theta = angle_diff(env["wdir"], inlet_facing) % 360
    cum = np.zeros((len(env["t"]), 6))
    # the sample is warmed inside the sensor, so particles see a lower RH than the ambient air
    rh_s = np.clip(rel_humidity(env["temp"] + P.PMS_SAMPLE_HEAT_C, env["ah"]), 0, P.PMS_RH_CAP)
    for mode, n_t in zip(env["modes"], env["mode_n"]):
        if np.max(n_t) <= 0:
            continue
        cum += n_t[:, None] * _channel_response(mode, rh_s, env["u"], theta, enclosure, unit.pms_size)
    cum *= np.exp(-cum[:, :1] * P.PMS_COINCIDENCE)               # coincidence loss
    diff = np.clip(cum - np.concatenate([cum[:, 1:], np.zeros((len(cum), 1))], 1), 0, None)
    counted = rng.poisson(diff * P.PMS_SAMPLE_CM3) / P.PMS_SAMPLE_CM3
    smooth = first_order(counted, P.PMS_SMOOTH_TAU)
    gain = unit.pms_gain * (1.0 - P.PMS_GAIN_LOSS_PER_DAY * age_days)
    diff01 = smooth * 100.0 * gain
    cum01 = np.cumsum(diff01[:, ::-1], axis=1)[:, ::-1]
    cum_u16 = np.clip(np.round(cum01), 0, 65535).astype(np.uint16)
    cf1 = _mass_from_diff(diff01) * pms_calibration_factor()
    cf1_u16 = np.clip(np.round(cf1), 0, 65535).astype(np.uint16)
    atm = np.where(cf1 < 30, cf1, 30 + (cf1 - 30) * 0.66)
    atm_u16 = np.clip(np.round(atm), 0, 65535).astype(np.uint16)
    return cum_u16, cf1_u16, atm_u16


def pms_frame(cf1, atm, cum, reserved=0x9700):
    """Encode one PMS7003 active-mode frame (32 bytes, big-endian, checksum)."""
    words = [28, *map(int, cf1), *map(int, atm), *map(int, cum), reserved]
    b = bytearray([0x42, 0x4D])
    for w in words:
        b += bytes([(w >> 8) & 0xFF, w & 0xFF])
    cs = sum(b) & 0xFFFF
    b += bytes([cs >> 8, cs & 0xFF])
    return bytes(b)


def pms_frames_matrix(cf1, atm, cum):
    """Vectorised pms_frame for a whole run: (T,32) uint8."""
    n = len(cum)
    words = np.concatenate([np.full((n, 1), 28), cf1, atm, cum, np.full((n, 1), 0x9700)], 1).astype(np.uint32)
    m = np.zeros((n, 32), np.uint8)
    m[:, 0], m[:, 1] = 0x42, 0x4D
    m[:, 2:30:2] = (words >> 8) & 0xFF
    m[:, 3:30:2] = words & 0xFF
    cs = m[:, :30].astype(np.uint32).sum(1) & 0xFFFF
    m[:, 30], m[:, 31] = cs >> 8, cs & 0xFF
    return m


# ---------------------------------------------------------------------------
# BME688
# ---------------------------------------------------------------------------
def simulate_bme(env, unit: Unit, age_days, enclosure, rng):
    """Gas resistance (ohm) per 1 s heater step, plus T/RH/P as the BME688 reports them."""
    n = len(env["t"])
    gases = env["gases"]
    enc = P.ENCLOSURES[enclosure]
    tau_v = enc["gas_tau"] / (1.0 + enc["gas_tau_wind"] * env["u"])
    tau_up = np.array([P.MOX_GAS[g]["tau_up"] for g in P.GASES])
    tau_dn = np.array([P.MOX_GAS[g]["tau_dn"] for g in P.GASES])

    # enclosure vent exchange, then asymmetric adsorption/desorption at the MOX surface
    surf = np.empty_like(gases)
    cv = gases[0].copy()
    cs = gases[0].copy()
    kv = np.minimum(1.0, 1.0 / tau_v)
    for i in range(n):
        cv += (gases[i] - cv) * kv[i]
        cs += (cv - cs) / np.where(cv > cs, tau_up, tau_dn)
        surf[i] = cs

    step = (np.arange(n) + unit.heater_phase) % len(P.HEATER_PROFILE)
    th = P.HEATER_PROFILE[step].astype(float)
    decay = max(0.0, 1.0 - P.MOX_SENS_DECAY_PER_DAY * age_days)
    red = np.zeros(n)
    ox = np.zeros(n)
    for j, g in enumerate(P.GASES):
        m = P.MOX_GAS[g]
        s_eff = m["S"] * unit.mox_sens[g] * decay * np.exp(
            -0.5 * ((th - (m["Tpk"] + unit.mox_tpk[g])) / m["w"]) ** 2)
        term = s_eff * (np.clip(surf[:, j], 0, None) / m["cref"]) ** m["beta"]
        if m["ox"]:
            ox += term
        else:
            red += term
    ln_r0 = (np.log(unit.mox_rref) + P.MOX_R_TEMP_COEF * (320.0 - th)
             - P.mox_humidity_gamma(th) * unit.mox_gamma * (env["ah"] - P.AH_REF)
             - P.MOX_AMB_TEMP_COEF * (env["temp"] - 25.0)
             + unit.mox_drift(age_days)
             + P.MOX_WARMUP_AMP * np.exp(-env["t"] / P.MOX_WARMUP_TAU))
    ln_r = ln_r0 + np.log1p(ox) - np.log1p(red) + rng.normal(0, P.MOX_NOISE, n)
    gas_r = np.clip(np.exp(ln_r), P.MOX_R_MIN, P.MOX_R_MAX).astype(np.float32)

    t_meas = env["temp"] + unit.self_heat + rng.normal(0, P.BME_T_NOISE, n)
    rh_meas = np.clip(rel_humidity(t_meas, env["ah"]) + unit.rh_bias + rng.normal(0, P.BME_RH_NOISE, n), 0, 100)
    p_meas = env["pressure"] + rng.normal(0, 0.12, n)
    return dict(gas_r=gas_r, gas_step=step.astype(np.uint8), t_meas=t_meas.astype(np.float32),
                rh_meas=rh_meas.astype(np.float32), p_meas=p_meas.astype(np.float32),
                surface_ppb=surf)
