"""Pure helpers of tools/phase2_gap_experiments.py (no model weights needed)."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import phase2_gap_experiments as gx


class Remap(unittest.TestCase):
    def test_labels_are_mapped_and_images_linked(self):
        try: import yaml
        except ImportError: self.skipTest("PyYAML unavailable")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "src"
            for split in ("val", "test"):
                (src / "labels" / split).mkdir(parents=True); (src / "images" / split).mkdir(parents=True)
                (src / "images" / split / "a.jpg").write_bytes(b"x")
                (src / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.2 0.3 0.4\n")
            out = gx.remap_yolo_dataset(src, d / "out", {15: "cat", 16: "dog"})
            self.assertEqual((d / "out/labels/val/a.txt").read_text().split("\n")[:2],
                             ["15 0.5 0.5 0.2 0.2", "16 0.1 0.2 0.3 0.4"])
            self.assertTrue((d / "out/images/test/a.jpg").exists())
            self.assertEqual(yaml.safe_load(out.read_text())["names"], {15: "cat", 16: "dog"})

    def test_unknown_class_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "src"; (src / "labels/val").mkdir(parents=True); (src / "images/val").mkdir(parents=True)
            (src / "labels/val/a.txt").write_text("7 0.5 0.5 0.1 0.1\n")
            with self.assertRaises(ValueError): gx.remap_yolo_dataset(src, d / "out", {})
            (d / "out2").mkdir()
            with self.assertRaises(FileExistsError): gx.remap_yolo_dataset(src, d / "out2", {})

    def test_empty_source_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): gx.remap_yolo_dataset(Path(d), Path(d) / "o", {})


class Selection(unittest.TestCase):
    def test_seg_conf_choice_prefers_higher_iou_then_higher_conf(self):
        t = pd.DataFrame({"conf": [0.1, 0.25, 0.4], "mIoU": [0.90, 0.92, 0.92]})
        self.assertEqual(gx.choose_seg_conf(t), 0.4)
        with self.assertRaises(ValueError): gx.choose_seg_conf(t.iloc[0:0])

    def test_largest_within_limit(self):
        t = pd.DataFrame({"min_mask_fraction": [0.01, 0.02, 0.05, 0.1], "share": [0.0, 0.002, 0.009, 0.03]})
        self.assertEqual(gx.largest_within(t, "min_mask_fraction", "share", 0.01), 0.05)
        with self.assertRaises(ValueError): gx.largest_within(t, "min_mask_fraction", "share", -1)


class Records(unittest.TestCase):
    def rec(self):
        return pd.DataFrame({"verdict": ["ready", "refer_to_staff", "refer_to_staff"],
                             "reasons": ["", "animal_too_small", "segmentation_unmatched;disagreement"],
                             "mask_fraction": [0.30, 0.03, 0.0], "segmentation_matched": [True, True, False],
                             "detect_ms": [10.0, 12.0, 30.0], "total_ms": [40.0, 41.0, 90.0]})

    def test_stage_percentiles(self):
        p = gx.stage_percentiles(self.rec()).set_index("stage")
        self.assertAlmostEqual(p.loc["detect_ms", "p50_ms"], 12.0)
        self.assertAlmostEqual(p.loc["total_ms", "p95_ms"], float(np.percentile([40, 41, 90], 95)))
        with self.assertRaises(ValueError): gx.stage_percentiles(pd.DataFrame({"a": [1]}))

    def test_min_mask_fraction_counts_matched_only(self):
        s = gx.min_mask_fraction_sweep(self.rec(), [0.02, 0.05]).set_index("min_mask_fraction")
        self.assertEqual(s.loc[0.02, "flagged_too_small"], 0)
        self.assertEqual(s.loc[0.05, "flagged_too_small"], 1)

    def test_referral_summary(self):
        r = gx.referral_summary(self.rec())
        self.assertEqual((r["n"], r["referred"], r["unmatched"]), (3, 2, 1))
        self.assertEqual(r["reason_counts"]["disagreement"], 1)

    def test_sweep_uses_project_scoring(self):
        gts = {"a": (np.array([[0, 0, 10, 10]]), np.array([0]))}
        preds = {"a": (np.array([[0, 0, 10, 10]]), np.array([0]), np.array([0.6]))}
        s = gx.sweep_from_predictions(preds, gts, [0.5, 0.7], 0.5).set_index("conf")
        self.assertEqual((s.loc[0.5, "tp"], s.loc[0.7, "fn"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
