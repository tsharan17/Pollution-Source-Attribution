"""Unit tests for the STM32 twin and the verification toolkit:  python -m unittest discover tests"""
import os
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stm32_twin import schedule as SC, spec, power                      # noqa: E402
from stm32_twin.spec import Arch                                        # noqa: E402
from verify import metrics as MT, domain_gap                            # noqa: E402

CYC = dict(sample=500.0, wcheck=50.0, features=30000.0, mlp=8000.0, post=150.0, uart_frame=600.0)


class Intervals(unittest.TestCase):
    def test_merge(self):
        m = SC.merge(np.array([[0, 1], [0.5, 2], [3, 4], [3.5, 3.6]]))
        np.testing.assert_allclose(m, [[0, 2], [3, 4]])

    def test_merge_empty(self):
        self.assertEqual(len(SC.merge(np.zeros((0, 2)))), 0)


class PowerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Arch("t", tick="hal1k")
        cls.tickless = Arch("t", tick="tickless", sd="async", gps="rmc", gps_parse_cycles=1500, isr_overhead_cycles=250)

    def test_state_fractions_sum_to_one(self):
        for a in (self.base, self.tickless):
            r = SC.simulate(a, CYC, T=300)
            self.assertAlmostEqual(r["run"] + r["sleep"] + r["stop"], 1.0, places=3)

    def test_tickless_uses_stop_and_saves_current(self):
        a = SC.simulate(self.base, CYC, T=300)
        b = SC.simulate(self.tickless, CYC, T=300)
        self.assertEqual(a["stop"], 0.0)
        self.assertGreater(b["stop"], 0.5)
        self.assertLess(b["avg_ma"], 0.5 * a["avg_ma"])

    def test_average_current_between_extremes(self):
        r = SC.simulate(self.tickless, CYC, T=300)
        self.assertGreater(r["avg_ma"], spec.pick(spec.I_STOP_MA, "nom"))
        self.assertLess(r["avg_ma"], spec.i_run_ma(170))

    def test_cost_ranges_ordered(self):
        for k, (lo, nom, hi) in spec.COST.items():
            self.assertLessEqual(lo, nom, k)
            self.assertLessEqual(nom, hi, k)
        for tup in (spec.I_RUN_170_MA, spec.I_SLEEP_170_MA, spec.I_STOP_MA, spec.ART_INFLATION):
            self.assertEqual(list(tup), sorted(tup))

    def test_blocking_sd_can_overrun_gps_ring_but_async_cannot(self):
        blocking = Arch("b", sd="blocking", consumer="main", gps="default", gps_ring=64)
        asyn = Arch("a", sd="async", consumer="main", gps="default", gps_ring=64)
        self.assertGreater(SC.ring_overflow(blocking, n=50000)["gps"]["p_per_write"], 0.0)
        self.assertEqual(SC.ring_overflow(asyn, n=50000)["gps"]["p_per_write"], 0.0)

    def test_battery_arithmetic(self):
        b = power.budget(1.0, 1.0)
        self.assertAlmostEqual(b["hours"], 2000 * 0.85 / b["total_ma"])
        self.assertLess(power.budget(0.25, 1.0)["total_ma"], b["total_ma"])


class FastMath(unittest.TestCase):
    def test_fastmath_accuracy_against_libm(self):
        src = r'''
#include <math.h>
#include <stdio.h>
#include "fastmath.h"
int main(void){ double el=0, ee=0;
  for (float x = 1.0f; x < 1.0e5f; x *= 1.0007f) { double d = fabs(fm_log10f(x) - log10f(x)); if (d > el) el = d; }
  for (float x = -20.0f; x < 20.0f; x += 0.01f) { double d = fabs(fm_expf(x) / expf(x) - 1.0); if (d > ee) ee = d; }
  printf("%g %g\n", el, ee); return 0; }
'''
        d = os.path.join(ROOT, "results", "mcu_build")
        os.makedirs(d, exist_ok=True)
        c = os.path.join(d, "t_fast.c")
        exe = os.path.join(d, "t_fast.exe")
        open(c, "w").write(src)
        subprocess.run(["gcc", "-O2", "-I", os.path.join(ROOT, "firmware_c"), c, "-o", exe, "-lm"], check=True, capture_output=True)
        el, ee = map(float, subprocess.run([exe], capture_output=True, text=True, check=True).stdout.split())
        self.assertLess(el, 3e-5)         # log10 absolute error
        self.assertLess(ee, 2e-6)         # exp relative error (float rounding of x*log2e dominates)


class Verification(unittest.TestCase):
    def test_ols_recovers_line(self):
        x = np.linspace(5, 100, 50)
        m = MT.ols_epa(x, 1.2 * x + 2)
        self.assertAlmostEqual(m["slope"], 1.2)
        self.assertAlmostEqual(m["intercept"], 2.0)
        self.assertAlmostEqual(m["r2"], 1.0)

    def test_epa_verdict_flags_bad_slope(self):
        x = np.linspace(5, 100, 50)
        v = MT.epa_verdict(MT.ols_epa(x, 2.0 * x), dict(sd=1.0, cv=0.1))
        self.assertFalse(v["slope"])
        v = MT.epa_verdict(MT.ols_epa(x, 1.05 * x + 1), dict(sd=1.0, cv=0.1))
        self.assertTrue(all(v.values()))

    def test_wilson_interval(self):
        lo, hi = MT.wilson(90, 100)
        self.assertLess(lo, 0.9)
        self.assertGreater(hi, 0.9)
        self.assertAlmostEqual(MT.n_needed_for_halfwidth(0.5, 0.05), 385, delta=2)

    def test_ece_perfect_and_overconfident(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(0.5, 1, 20000)
        correct = rng.random(20000) < conf
        self.assertLess(MT.ece(conf, correct)[0], 0.02)
        self.assertGreater(MT.ece(np.full(2000, 0.99), rng.random(2000) < 0.6)[0], 0.3)

    def test_source_correction_fits(self):
        rng = np.random.default_rng(1)
        pm = rng.uniform(20, 200, 400)
        sh = np.eye(5)[rng.integers(0, 5, 400)]
        gains = np.array([0.5, 0.7, 1.0, 2.0, 1.2])
        ref = (sh * gains).sum(1) * pm
        coef = MT.fit_source_correction(pm, np.zeros(400), sh, ref, alpha=1e-6)
        self.assertLess(np.abs(MT.apply_source_correction(coef, pm, np.zeros(400), sh) - ref).max(), 1.0)

    def test_block_bootstrap_interval_contains_point(self):
        blocks = [np.array([0.7 + 0.01 * i]) for i in range(10)]
        p, lo, hi = MT.block_bootstrap(lambda b: float(np.mean(np.concatenate(b))), blocks, 200, 0)
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)

    def test_domain_gap_detects_shift(self):
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, (1500, 25))
        same = domain_gap.compare(a, rng.normal(0, 1, (1500, 25)))["auc"]
        b = rng.normal(0, 1, (1500, 25))
        b[:, 3] += 1.5
        diff = domain_gap.compare(a, b)
        self.assertLess(same, 0.56)
        self.assertGreater(diff["auc"], 0.75)
        self.assertEqual(diff["features"][0]["feature"], "lfrac_2.5-5")


if __name__ == "__main__":
    unittest.main()
