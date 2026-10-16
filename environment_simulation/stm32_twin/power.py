"""System power budget at the battery: where every milliamp goes and what each design lever saves."""
from . import spec

BATT_MAH, BATT_V, USABLE = 2000.0, 3.7, 0.85
PMS_ACTIVE_MA_5V, PMS_SLEEP_MA_5V, BOOST_EFF = 80.0, 0.2, 0.88      # datasheet typical; EST for the converter
BME_MA = 9.0                      # heater duty included, EST
SD_MA = 1.04                      # idle 1.0 + write bursts
REG_MA = 0.5                      # LDO / boost quiescent, EST
GPS_MODES = {                      # average mA from the 3.3 V rail, EST (NEO-6M class receiver)
    "continuous NMEA": 45.0,
    "power-save mode": 12.0,
    "one fix every 10 min": 3.0,
}


def pms_batt_ma(duty):
    i5 = duty * PMS_ACTIVE_MA_5V + (1 - duty) * PMS_SLEEP_MA_5V
    return i5 * 5.0 / (BOOST_EFF * BATT_V)


def budget(pms_duty, mcu_ma, gps="continuous NMEA", nucleo=False, which="nom"):
    parts = {
        "PMS7003 fan + laser (via 5 V boost)": pms_batt_ma(pms_duty),
        "BME688 gas sensor": BME_MA,
        "STM32 (MCU)": mcu_ma,
        f"GPS ({gps})": GPS_MODES[gps],
        "SD card": SD_MA,
        "Regulators": REG_MA,
    }
    if nucleo:
        parts["NUCLEO board overhead (ST-LINK etc.)"] = spec.pick(spec.NUCLEO_OVERHEAD_MA, which)
    total = sum(parts.values())
    return dict(parts=parts, total_ma=total, hours=BATT_MAH * USABLE / total)
