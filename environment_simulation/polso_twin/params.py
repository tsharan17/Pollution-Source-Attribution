"""
Every physical, hardware and firmware assumption of the PolSo digital twin lives here.

Nothing in this file is measured on PolSo hardware. Values are literature/datasheet
ballparks chosen to be *plausible*, and each block says where its numbers come from.
When real PMS7003 / BME688 data exists, re-fit these numbers first; the rest of the
code does not need to change.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
CLASSES = ["vehicle_exhaust", "biomass_burning", "dust", "industrial", "background"]
UNCERTAIN = "uncertain"
FAULT = "sensor_fault"
WARMUP = "warming_up"
MIXED = "mixed"          # ground-truth only: no single source dominates

GASES = ["CO", "NO2", "oVOC", "aVOC", "sVOC", "SO2"]   # all in ppb
# oVOC = oxygenated VOCs (methanol, aldehydes, furans; smoke & cooking)
# aVOC = aromatics (benzene/toluene/xylene; fuel, solvents, plastic burning)
# sVOC = solvent-type VOCs (ethanol, acetone, esters; painting/printing)

# ---------------------------------------------------------------------------
# Aerosol sources
# ---------------------------------------------------------------------------
# Mode = (count-median dry diameter um, geometric std dev, mass weight,
#         particle density g/cm3, hygroscopicity kappa, optical size factor)
# optical size factor < 1: absorbing particles (soot) scatter like smaller ones.
# kappa: soot ~0-0.1, smoke ~0.1-0.3, sulfate-rich ~0.4-0.6, mineral dust ~0.03
# (Petters & Kreidenweis 2007 kappa-Koehler; typical urban size modes from
# Seinfeld & Pandis). Gas emission ratios are ppb of gas per ug/m3 of PM2.5 excess,
# order-of-magnitude from emission-factor tables (Akagi 2011 for biomass,
# roadside CO/PM and NO2/PM ratios for traffic).
SUBTYPES = {
    # ---- vehicle exhaust ----
    "diesel_traffic": dict(
        cls="vehicle_exhaust", train=True,
        modes=[(0.08, 1.8, 0.75, 1.2, 0.05, 0.75),
               (0.25, 1.6, 0.15, 1.5, 0.15, 0.90),
               (1.5, 2.0, 0.10, 2.3, 0.03, 1.00)],      # brake/tyre/road coarse
        gas=dict(CO=25, NO2=2.5, oVOC=0.3, aVOC=0.3, sVOC=0.0, SO2=0.05),
        pattern="traffic"),
    "petrol_traffic": dict(
        cls="vehicle_exhaust", train=True,
        modes=[(0.06, 1.7, 0.60, 1.2, 0.10, 0.80),
               (1.5, 2.0, 0.40, 2.3, 0.03, 1.00)],
        gas=dict(CO=150, NO2=1.0, oVOC=0.5, aVOC=1.2, sVOC=0.1, SO2=0.02),
        pattern="traffic"),
    "two_stroke_traffic": dict(                  # unseen in training (autos/2-wheelers)
        cls="vehicle_exhaust", train=False,
        modes=[(0.10, 1.7, 0.80, 1.0, 0.10, 0.85),
               (1.5, 2.0, 0.20, 2.3, 0.03, 1.00)],
        gas=dict(CO=90, NO2=0.5, oVOC=0.8, aVOC=1.5, sVOC=0.1, SO2=0.02),
        pattern="traffic"),
    # ---- biomass burning ----
    "biomass_flaming": dict(
        cls="biomass_burning", train=True,
        modes=[(0.13, 1.6, 0.90, 1.3, 0.20, 0.85),
               (1.5, 2.0, 0.10, 1.5, 0.10, 0.90)],
        gas=dict(CO=6, NO2=0.25, oVOC=0.8, aVOC=0.15, sVOC=0.05, SO2=0.02),
        pattern="burning"),
    "biomass_smoldering": dict(
        cls="biomass_burning", train=True,
        modes=[(0.18, 1.55, 0.95, 1.3, 0.15, 0.95),
               (2.0, 2.0, 0.05, 1.5, 0.10, 0.95)],
        gas=dict(CO=10, NO2=0.05, oVOC=1.5, aVOC=0.2, sVOC=0.05, SO2=0.02),
        pattern="burning"),
    "trash_burning": dict(                       # unseen: plastic + mixed waste
        cls="biomass_burning", train=False,
        modes=[(0.12, 1.7, 0.85, 1.3, 0.15, 0.80),
               (1.5, 2.0, 0.15, 1.6, 0.10, 0.90)],
        gas=dict(CO=8, NO2=0.2, oVOC=0.6, aVOC=1.0, sVOC=0.1, SO2=0.1),
        pattern="burning"),
    # ---- dust ----
    "road_dust": dict(
        cls="dust", train=True,
        modes=[(0.4, 1.8, 0.10, 2.6, 0.03, 1.0),
               (1.5, 2.0, 0.90, 2.6, 0.03, 1.0)],
        gas=dict(CO=0.0, NO2=0.0, oVOC=0.0, aVOC=0.0, sVOC=0.0, SO2=0.0),
        pattern="resuspension"),
    "construction_dust": dict(
        cls="dust", train=True,
        modes=[(0.5, 1.8, 0.15, 2.6, 0.03, 1.0),
               (1.8, 2.1, 0.85, 2.6, 0.03, 1.0)],
        gas=dict(CO=1.0, NO2=0.15, oVOC=0.0, aVOC=0.02, sVOC=0.0, SO2=0.0),  # machinery
        pattern="steady"),
    "dust_storm": dict(                          # unseen: regional event
        cls="dust", train=False,
        modes=[(1.2, 2.0, 1.0, 2.6, 0.03, 1.0)],
        gas=dict(CO=0.0, NO2=0.0, oVOC=0.0, aVOC=0.0, sVOC=0.0, SO2=0.0),
        pattern="storm"),
    # ---- industrial ----
    "coal_combustion": dict(                     # boilers, brick kilns
        cls="industrial", train=True,
        modes=[(0.20, 1.8, 0.60, 1.8, 0.45, 1.0),
               (1.2, 2.2, 0.40, 2.2, 0.10, 1.0)],
        gas=dict(CO=8, NO2=0.6, oVOC=0.1, aVOC=0.1, sVOC=0.0, SO2=3.0),
        pattern="plume"),
    "solvent_industry": dict(                    # painting / printing
        cls="industrial", train=True,
        modes=[(0.25, 1.7, 1.0, 1.2, 0.05, 1.0)],
        gas=dict(CO=0.5, NO2=0.05, oVOC=0.5, aVOC=2.5, sVOC=8.0, SO2=0.0),
        pattern="plume"),
    "metal_processing": dict(
        cls="industrial", train=True,
        modes=[(0.30, 1.9, 0.5, 3.5, 0.10, 1.1),
               (2.0, 2.0, 0.5, 3.0, 0.05, 1.0)],
        gas=dict(CO=4, NO2=0.5, oVOC=0.05, aVOC=0.1, sVOC=0.0, SO2=2.0),
        pattern="plume"),
    # ---- confounders (no correct class; device should say "uncertain") ----
    "cooking": dict(
        cls=None, train=False,
        modes=[(0.12, 1.7, 1.0, 1.0, 0.10, 1.0)],
        gas=dict(CO=2, NO2=0.05, oVOC=2.5, aVOC=0.1, sVOC=0.3, SO2=0.0),
        pattern="burning"),
    "incense": dict(
        cls=None, train=False,
        modes=[(0.20, 1.6, 1.0, 1.1, 0.10, 1.0)],
        gas=dict(CO=5, NO2=0.05, oVOC=1.2, aVOC=0.8, sVOC=0.2, SO2=0.0),
        pattern="steady"),
}

# Regional (aged) background aerosol and its co-varying gases.
BACKGROUND_MODES = [(0.15, 1.65, 0.85, 1.6, 0.35, 1.0),
                    (1.5, 2.0, 0.15, 2.4, 0.05, 1.0)]
# background gas (ppb) = offset + slope * background PM2.5 (ug/m3)
BACKGROUND_GAS = dict(CO=(200, 6.0), NO2=(5, 0.2), oVOC=(2, 0.03),
                      aVOC=(0.5, 0.01), sVOC=(1.0, 0.0), SO2=(2, 0.03))

# ---------------------------------------------------------------------------
# BME688 metal-oxide gas sensor
# ---------------------------------------------------------------------------
# Conductance model (Clifford-Tuma power law). Reducing gas: G/G0 = 1 + S*(C/cref)^beta.
# Oxidising gas (NO2): R/R0 = 1 + S*(C/cref)^beta. S peaks at a gas-specific
# heater temperature (Gaussian in T_heater) - this is why a heater *profile*
# yields a fingerprint. tau_up / tau_dn: adsorption / desorption time constants (s).
MOX_GAS = {
    "CO":   dict(S=1.0, Tpk=340, w=90, beta=0.60, cref=1000, ox=False, tau_up=25, tau_dn=50),
    "NO2":  dict(S=2.0, Tpk=190, w=70, beta=0.80, cref=100,  ox=True,  tau_up=30, tau_dn=70),
    "oVOC": dict(S=3.0, Tpk=270, w=70, beta=0.60, cref=100,  ox=False, tau_up=20, tau_dn=60),
    "aVOC": dict(S=2.5, Tpk=360, w=60, beta=0.65, cref=100,  ox=False, tau_up=35, tau_dn=110),
    "sVOC": dict(S=3.5, Tpk=300, w=80, beta=0.60, cref=100,  ox=False, tau_up=15, tau_dn=40),
    "SO2":  dict(S=0.6, Tpk=390, w=60, beta=0.50, cref=100,  ox=False, tau_up=60, tau_dn=150),
}
# Heater profile for BME688 parallel mode: one step per second, 10 s per scan.
HEATER_PROFILE = np.array([150, 200, 250, 300, 350, 400, 350, 300, 250, 200])
HEATER_GROUPS = np.array([150, 200, 250, 300, 350, 400])

MOX_R_REF = 120e3          # clean-air resistance at 320 C, 25 C, AH=10 g/m3 (ohm)
MOX_R_TEMP_COEF = 0.010    # ln R per degC of heater below 320 C
MOX_AMB_TEMP_COEF = 0.005  # ln R drop per degC of ambient temperature
MOX_NOISE = 0.010          # ln R white noise per reading
MOX_R_MIN, MOX_R_MAX = 100.0, 5e7


def mox_humidity_gamma(t_heater):
    """d(ln R)/d(AH) magnitude, per g/m3 absolute humidity. MOX resistance falls with water."""
    return 0.035 - 0.000033 * (np.asarray(t_heater, float) - 150.0)


AH_REF = 10.0              # g/m3 reference absolute humidity
MOX_DRIFT_PER_DAY = 0.002  # deterministic ln R baseline drift per day
MOX_RW_PER_SQRT_DAY = 0.010  # random-walk ln R per sqrt(day)
MOX_SENS_DECAY_PER_DAY = 0.0005  # fractional sensitivity loss per day
MOX_WARMUP_AMP, MOX_WARMUP_TAU = 0.5, 600.0   # after power-on (ln R excess, s)

# Unit-to-unit spread (lognormal sigma unless stated)
UNIT_MOX_RREF_SIGMA = 0.35
UNIT_MOX_SENS_SIGMA = 0.15
UNIT_MOX_TPK_SD = 10.0     # degC
UNIT_MOX_GAMMA_SIGMA = 0.20
BME_SELF_HEAT = 1.5        # degC the enclosure/heater warms the T/RH sensor
BME_SELF_HEAT_UNIT_SD = 0.3
BME_T_NOISE, BME_RH_NOISE, BME_RH_UNIT_SD = 0.1, 1.0, 2.0

# ---------------------------------------------------------------------------
# PMS7003 optical particle counter
# ---------------------------------------------------------------------------
PMS_THRESHOLDS = np.array([0.3, 0.5, 1.0, 2.5, 5.0, 10.0])   # cumulative channels (um)
PMS_BIN_MID = np.array([0.4, 0.7, 1.6, 3.5, 7.0, 12.0])      # mid of differential bins
PMS_CE_LOW_D50, PMS_CE_LOW_W = 0.45, 0.06     # Plantower 0.3 um channel has low efficiency
PMS_CE_HIGH_D50, PMS_CE_HIGH_P = 3.5, 2.0     # coarse particles poorly sampled (inlet/flow)
PMS_SIZING_SIGMA = 0.30     # ln-space sizing blur
PMS_SAMPLE_CM3 = 2.0        # effective sampled volume per 1 s frame (counting noise)
PMS_SMOOTH_TAU = 4.0        # s, internal moving average
PMS_COINCIDENCE = 5e-5      # per (#/cm3) detected: exp(-N*k) coincidence loss
PMS_RHO_ASSUMED = 1.65      # density the PMS firmware assumes for mass
PMS_UNIT_GAIN_SIGMA = 0.10
PMS_UNIT_SIZE_SIGMA = 0.05
PMS_GAIN_LOSS_PER_DAY = 0.0003   # fan / optics fouling
PMS_WAKE_STABLE_S = 30      # datasheet: wait 30 s after fan start
PMS_SAMPLE_HEAT_C = 5.0      # laser + fan warm the sample air inside the housing, lowering its effective RH   EST
PMS_RH_CAP = 95.0           # effective RH seen by the particles is capped (droplet / fog regime not modelled)
# (PMS7003 datasheet: 0.3-10 um, counts per 0.1 L, uint16 fields, 32-byte frame 0x42 0x4D)
# Coarse under-response follows published low-cost OPC evaluations (e.g. Kuula 2020).

# ---------------------------------------------------------------------------
# Enclosure / inlet (taken from PolSo-19 Run A, 4 m/s numbers)
# ---------------------------------------------------------------------------
# loss(d,u,theta) = loss_ref(d) * (u/4)^0.8 * (1 - dir_amp*cos(theta))
# loss_ref interpolated in log d between "fine"(1 um) and "coarse"(5 um).
ENCLOSURES = {
    "baseline":   dict(loss_fine=0.073, loss_coarse=0.132, dir_amp=0.71, gas_tau=4.0, gas_tau_wind=0.15),
    "hooded":     dict(loss_fine=0.035, loss_coarse=0.044, dir_amp=0.39, gas_tau=6.0, gas_tau_wind=0.02),
    "multi_hole": dict(loss_fine=0.081, loss_coarse=0.092, dir_amp=0.20, gas_tau=5.0, gas_tau_wind=0.02),
    "ideal":      dict(loss_fine=0.0, loss_coarse=0.0, dir_amp=0.0, gas_tau=5.0, gas_tau_wind=0.0),  # reference only
}
DEFAULT_ENCLOSURE = "hooded"

# ---------------------------------------------------------------------------
# Firmware configuration (STM32G474RE)
# ---------------------------------------------------------------------------
FW_WINDOW_S = 60
FW_HOP_S = 30
FW_WARMUP_S = 300
FW_PM_VALID_FRAC = 0.80
FW_GAS_VALID_FRAC = 0.70
FW_BASE_UP = 0.0005        # baseline tracker: rise toward cleaner air (per sample, tau ~5.6 h)
FW_BASE_DOWN = 1e-6        # baseline tracker: slow fall (ln units per sample, ~0.009/day)
# (v1 used 0.01 / 1e-4: the twin showed it erased the gas signal within hours of pollution)
FW_PROB_EMA = 0.5          # on-device probability smoothing between windows
N_FEATURES = 25
FEATURE_NAMES = (
    ["lfrac_0.3-0.5", "lfrac_0.5-1", "lfrac_1-2.5", "lfrac_2.5-5", "lfrac_5-10", "lfrac_>10",
     "log_count", "pm25/pm10", "pm1/pm25", "cv_count", "skew_count", "max_rise"]
    + ["gas_resp"] + [f"gas_shape_{t}C" for t in HEATER_GROUPS]
    + ["gas_slope_300C", "gas_slope_200C", "pm_gas_corr", "gas_per_pm"]
    + ["rh", "temp"])
PM_FEATURES = list(range(0, 12)) + [23, 24]
GAS_FEATURES = list(range(12, 23)) + [23, 24]

# ---------------------------------------------------------------------------
# Power (datasheet typicals; *measure these on the bench*)
# ---------------------------------------------------------------------------
POWER = dict(
    battery_mah=2000, battery_v=3.7, usable_frac=0.85,
    boost_eff=0.88,                 # 3.7 V -> 5 V for PMS7003
    ldo_eff=3.3 / 3.7,              # 3.3 V rail from battery
    pms_active_ma=80, pms_sleep_ma=0.2,                 # at 5 V
    bme_ma=9.0,                                         # heater ~75 % duty at 12 mA
    mcu_run_ma=30.0, mcu_sleep_ma=6.0, mcu_stop_ma=0.1,  # G474 @170 MHz / Sleep / Stop2
    nucleo_overhead_ma=45.0,        # ST-LINK-V3E + LEDs on the NUCLEO board (assumed)
    gps_ma=45.0, gps_psm_ma=12.0,   # NEO-6M class tracking / power-save
    sd_idle_ma=1.0, sd_write_ma=60.0, sd_write_ms=20, sd_writes_per_min=2,
)
