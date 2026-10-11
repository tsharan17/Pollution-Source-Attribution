"""Unit tests for the digital twin and firmware twin:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polso_twin import params as P                                   # noqa: E402
from polso_twin.physics import (kappa_growth, abs_humidity, rel_humidity, mode_volume_um3,  # noqa: E402
                                subtype_number_per_pm25, mode_mass_frac_below, enclosure_loss)
from polso_twin.sensors import pms_frame, pms_frames_matrix, Unit     # noqa: E402
from polso_twin.firmware import PMSParser, decode_frames_fast, qmodel_forward, window_features  # noqa: E402
from polso_twin.environment import Scenario, Source                   # noqa: E402
from polso_twin.simulate import run_scenario, window_truth            # noqa: E402
from polso_twin.firmware import run_device                            # noqa: E402


class Physics(unittest.TestCase):
    def test_kappa_growth(self):
        self.assertAlmostEqual(float(kappa_growth(0.5, 0.0)), 1.0)
        self.assertGreater(float(kappa_growth(0.4, 90)), 1.5)           # sulfate at 90 % RH
        self.assertLess(float(kappa_growth(0.03, 90)), 1.1)             # dust barely grows

    def test_humidity_roundtrip(self):
        ah = abs_humidity(25.0, 60.0)
        self.assertAlmostEqual(float(ah), 13.8, delta=0.2)              # known value ~13.8 g/m3
        self.assertAlmostEqual(float(rel_humidity(25.0, ah)), 60.0, places=6)

    def test_pm25_mass_conservation(self):
        modes = P.SUBTYPES["road_dust"]["modes"]
        npp = subtype_number_per_pm25(modes)
        pm25 = sum(k * mode_volume_um3(m[0], m[1]) * m[3] * mode_mass_frac_below(m[0], m[1], 2.5)
                   for m, k in zip(modes, npp))
        self.assertAlmostEqual(pm25, 1.0, places=9)

    def test_inlet_loss(self):
        self.assertEqual(float(enclosure_loss(5.0, 0.0, 0, "hooded")), 0.0)   # no wind, no loss
        self.assertAlmostEqual(float(enclosure_loss(5.0, 4.0, 180, "baseline")), 0.132 * 1.71, places=3)
        self.assertGreater(float(enclosure_loss(5.0, 4.0, 180, "baseline")),
                           float(enclosure_loss(5.0, 4.0, 180, "hooded")))


class PMSFrames(unittest.TestCase):
    def setUp(self):
        self.cf1, self.atm, self.cum = [12, 25, 31], [12, 25, 31], [4000, 1500, 300, 20, 3, 1]
        self.frame = pms_frame(self.cf1, self.atm, self.cum)

    def test_frame_layout(self):
        self.assertEqual(len(self.frame), 32)
        self.assertEqual(self.frame[:4], bytes([0x42, 0x4D, 0x00, 0x1C]))
        self.assertEqual(sum(self.frame[:30]) & 0xFFFF, self.frame[30] << 8 | self.frame[31])

    def test_parser_roundtrip_and_resync(self):
        p = PMSParser()
        out = p.feed(b"\x00\x42\x13\x42" + self.frame + self.frame[:10])   # garbage, then a frame, then partial
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], (self.cf1, self.atm, self.cum))
        out = p.feed(self.frame[10:] + self.frame)                          # partial completes + one more
        self.assertEqual(len(out), 2)

    def test_parser_rejects_corruption(self):
        bad = bytearray(self.frame)
        bad[10] ^= 0x04
        p = PMSParser()
        self.assertEqual(p.feed(bytes(bad)), [])
        self.assertEqual(p.checksum_errors, 1)
        self.assertEqual(len(p.feed(self.frame)), 1)                       # recovers on the next frame

    def test_fast_decoder_matches_parser(self):
        rng = np.random.default_rng(0)
        cum = np.sort(rng.integers(0, 65535, (50, 6)), 1)[:, ::-1].astype(np.uint16)
        cf1 = rng.integers(0, 999, (50, 3)).astype(np.uint16)
        m = pms_frames_matrix(cf1, cf1, cum)
        present = rng.random(50) > 0.2
        f_cf1, f_cum, ok = decode_frames_fast(m, present)
        p = PMSParser()
        for i in range(50):
            fr = p.feed(m[i].tobytes()) if present[i] else []
            self.assertEqual(bool(fr), bool(ok[i]))
            if fr:
                self.assertEqual(list(f_cum[i].astype(int)), fr[0][2])
                self.assertEqual(list(f_cf1[i].astype(int)), fr[0][0])


class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = Scenario("t", 1500, seed=3, chamber=True, bg_pm25=15,
                          sources=[Source("road_dust", 100, pattern="chamber", t_on=360, t_off=1300)])
        cls.streams, cls.truth = run_scenario(cls.sc)

    def test_deterministic(self):
        s2, _ = run_scenario(self.sc)
        np.testing.assert_array_equal(s2["gas_r"], self.streams["gas_r"])
        np.testing.assert_array_equal(s2["uart"], self.streams["uart"])

    def test_counts_monotone_and_uint16(self):
        cum = self.truth["pms_cum"]
        self.assertTrue(np.all(np.diff(cum.astype(int), axis=1) <= 0))
        self.assertEqual(cum.dtype, np.uint16)

    def test_dust_raises_coarse_fraction(self):
        cum = self.truth["pms_cum"].astype(float)
        before, during = cum[100:300], cum[800:1200]
        self.assertGreater((during[:, 3] / during[:, 0]).mean(), 2 * (before[:, 3] / before[:, 0]).mean())

    def test_truth_labels(self):
        self.assertEqual(window_truth(self.truth, 1000)[0], "dust")
        self.assertEqual(window_truth(self.truth, 200)[0], "background")

    def test_device_windows_and_warmup(self):
        out = run_device(self.streams, None)
        self.assertTrue(np.all(out["label"][out["t_end"] < P.FW_WARMUP_S] == P.WARMUP))
        self.assertTrue(np.all(out["status"][out["t_end"] >= P.FW_WARMUP_S] == "ok"))
        self.assertTrue(np.all(np.isfinite(out["features"])))

    def test_dead_pms_is_flagged(self):
        sc = Scenario("t", 1200, seed=4, chamber=True, faults=dict(pms_dead_after=600))
        s, _ = run_scenario(sc)
        out = run_device(s, None)
        self.assertTrue(np.all(out["status"][out["t_end"] > 700] == P.FAULT))


class Quantisation(unittest.TestCase):
    def test_int8_close_to_float(self):
        from polso_twin.model import train_mlp, quantize, float_predict
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (1500, P.N_FEATURES))
        y = np.array(P.CLASSES, object)[np.argmax(X[:, :5] + 0.3 * rng.normal(size=(1500, 5)), 1)]
        m = train_mlp(X, y, seed=0)
        qm = quantize(m)
        pq, q0 = qmodel_forward(qm, X)
        pf = float_predict(m, X)
        self.assertGreater((pq.argmax(1) == pf.argmax(1)).mean(), 0.97)
        self.assertLessEqual(np.abs(q0).max(), 127)


if __name__ == "__main__":
    unittest.main()
