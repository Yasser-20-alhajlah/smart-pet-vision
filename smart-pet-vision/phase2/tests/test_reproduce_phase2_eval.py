"""tools/reproduce_phase2_eval.py: originals load from evidence; side-by-side comparison and tolerances."""
import sys
import unittest
from pathlib import Path

PHASE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2 / "tools"))

import reproduce_phase2_eval as rep  # noqa: E402


class ReproduceEvalTest(unittest.TestCase):
    def test_originals_match_reported_numbers(self):
        o = rep.load_originals(PHASE2 / "report" / "evidence")
        self.assertAlmostEqual(o["mAP50"], 0.984, places=3)
        self.assertAlmostEqual(o["mAP50_95"], 0.850, places=3)
        self.assertAlmostEqual(o["mask_mAP50_95"], 0.875, places=3)
        self.assertAlmostEqual(o["mIoU"], 0.923, places=3)
        self.assertAlmostEqual(o["mean_dice"], 0.959, places=3)

    def test_side_by_side_verdicts(self):
        o = {"mAP50": 0.9836, "mAP50_95": 0.8500, "mIoU": 0.9230}
        r = {"mAP50": 0.9830, "mAP50_95": 0.8560, "mIoU": 0.9229}
        c = rep.side_by_side(o, r)
        self.assertTrue(c["mAP50"]["within_tolerance"])
        self.assertFalse(c["mAP50_95"]["within_tolerance"])     # 0.006 > 0.005
        self.assertAlmostEqual(c["mAP50_95"]["difference"], 0.006, places=6)
        self.assertFalse(c["all_within_tolerance"])
        self.assertNotIn("mask_mAP50", c)                       # missing metrics are not invented

    def test_empty_comparison_has_no_verdict(self):
        self.assertIsNone(rep.side_by_side({}, {})["all_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
