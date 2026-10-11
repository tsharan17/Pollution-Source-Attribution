"""Small physics helpers shared by the environment and sensor models."""
import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtr

from . import params as P


def sat_vapour_hpa(t_c):
    """Magnus formula, saturation vapour pressure over water (hPa)."""
    t_c = np.asarray(t_c, float)
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def abs_humidity(t_c, rh):
    """Absolute humidity (g/m3) from temperature (C) and RH (%)."""
    return 216.7 * (np.asarray(rh, float) / 100.0 * sat_vapour_hpa(t_c)) / (273.15 + np.asarray(t_c, float))


def rel_humidity(t_c, ah):
    """RH (%) from temperature (C) and absolute humidity (g/m3)."""
    e = np.asarray(ah, float) * (273.15 + np.asarray(t_c, float)) / 216.7
    return 100.0 * e / sat_vapour_hpa(t_c)


def kappa_growth(kappa, rh):
    """Hygroscopic diameter growth factor, kappa-Koehler without the Kelvin term."""
    a_w = np.clip(np.asarray(rh, float), 0, 99.5) / 100.0
    return (1.0 + kappa * a_w / (1.0 - a_w)) ** (1.0 / 3.0)


def ou_process(n, tau, sigma, rng, x0=None):
    """Stationary Ornstein-Uhlenbeck (AR(1)) series with 1 s steps."""
    a = np.exp(-1.0 / max(tau, 1e-6))
    e = rng.normal(0.0, sigma * np.sqrt(1 - a * a), n)
    e[0] = rng.normal(0.0, sigma) if x0 is None else x0
    return lfilter([1.0], [1.0, -a], e)


def first_order(x, tau):
    """First-order low-pass with 1 s steps (unit DC gain), initialised at x[0]."""
    a = np.exp(-1.0 / max(tau, 1e-6))
    x = np.asarray(x, float)
    y, _ = lfilter([1 - a], [1.0, -a], x, axis=0, zi=a * x[:1])
    return y


def mode_volume_um3(gmd, gsd):
    """Mean particle volume (um3) of a lognormal number distribution."""
    s = np.log(gsd)
    return np.pi / 6.0 * np.exp(3 * np.log(gmd) + 4.5 * s * s)


def mode_mass_frac_below(gmd, gsd, d_cut):
    """Fraction of a mode's mass below d_cut (mass median = gmd*exp(3 s^2))."""
    s = np.log(gsd)
    mmd = gmd * np.exp(3 * s * s)
    return ndtr((np.log(d_cut) - np.log(mmd)) / s)


def subtype_number_per_pm25(modes):
    """For a list of modes, number conc (#/cm3) of each mode per 1 ug/m3 of PM2.5."""
    w = np.array([m[2] for m in modes])
    f25 = np.array([mode_mass_frac_below(m[0], m[1], 2.5) for m in modes])
    m_tot = 1.0 / np.sum(w * f25)                  # total mass per ug/m3 PM2.5
    return np.array([m_tot * m[2] / (mode_volume_um3(m[0], m[1]) * m[3]) for m in modes])


def enclosure_loss(d_um, u, theta_deg, enclosure):
    """Fractional particle loss at the inlet (PolSo-19 Run A derived)."""
    e = P.ENCLOSURES[enclosure]
    ld = np.log(np.clip(d_um, 1e-3, None))
    frac = np.clip((ld - np.log(1.0)) / (np.log(5.0) - np.log(1.0)), 0, 1)
    ref = e["loss_fine"] + frac * (e["loss_coarse"] - e["loss_fine"])
    ref = np.where(d_um < 1.0, e["loss_fine"] * np.clip(d_um, 0, 1) ** 0.3, ref)
    ref = np.where(d_um > 5.0, e["loss_coarse"] * (d_um / 5.0) ** 0.5, ref)
    speed = (np.clip(u, 0, None) / 4.0) ** 0.8
    direc = 1.0 - e["dir_amp"] * np.cos(np.deg2rad(theta_deg))
    return np.clip(ref * speed * direc, 0.0, 0.95)


def angle_diff(a, b):
    """Smallest signed difference a-b in degrees, in [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
