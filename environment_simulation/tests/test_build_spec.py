"""The Build tab must show the values the firmware really uses:  python -m unittest discover tests"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_spec                                                     # noqa: E402
from polso_twin import params as P                                    # noqa: E402

FW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "firmware_c")


def _row(b, sid, item):
    sec = next(s for s in b["sections"] if s["id"] == sid)
    return next(r for r in sec["rows"] if r[0] == item)


class BuildSheet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = build_spec.build()

    def test_heater_profile_matches_c_header(self):
        h = open(os.path.join(FW, "model_params.h"), encoding="utf-8").read()
        c = [int(x) for x in re.search(r"FW_HEATER_PROFILE\[10\] = \{([^}]*)\}", h).group(1).split(",")]
        shown = [int(x) for x in re.findall(r"\d+", _row(self.b, "sensors", "BME688 heater profile")[1])]
        self.assertEqual(c, list(P.HEATER_PROFILE))
        self.assertEqual(shown, c)

    def test_report_constants_match_c_header(self):
        h = open(os.path.join(FW, "polso_fw.h"), encoding="utf-8").read()
        n_pm = re.search(r"#define REP_PM_N (\d+)", h).group(1)
        n_lab = re.search(r"#define REP_LAB_N (\d+)", h).group(1)
        self.assertIn(n_pm, _row(self.b, "fw", "5-min PM2.5 average")[1])
        self.assertIn(n_lab, _row(self.b, "fw", "10-min source")[1])

    def test_every_feature_listed_and_tagged(self):
        model = next(s for s in self.b["sections"] if s["id"] == "model")
        self.assertEqual([r[1] for r in model["rows"]], list(P.FEATURE_NAMES))
        for s in self.b["sections"]:
            for r in s["rows"]:
                self.assertIn(r[-1], self.b["tags"], (s["id"], r))

    def test_pm_correction_has_all_classes(self):
        pm = next(s for s in self.b["sections"] if s["id"] == "pmcal")
        self.assertEqual({r[0] for r in pm["rows"]}, {k.replace("_", " ") for k in P.CLASSES + ["unknown"]})


if __name__ == "__main__":
    unittest.main()
