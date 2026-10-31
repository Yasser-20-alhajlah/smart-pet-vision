"""Preflight and evidence-consistency checks (no model weights or GPU needed)."""
import hashlib, json, shutil, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import preflight_release as pf
import verify_submission_evidence as vse


class Preflight(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); (self.tmp / "runs").mkdir()
        self.w = {}
        for k in ("detector", "segmenter", "classifier"):
            p = self.tmp / "runs" / f"{k}.pt"; p.write_bytes(k.encode()); self.w[k] = hashlib.sha256(k.encode()).hexdigest()
        self.data = dict(detector="runs/detector.pt", segmenter="runs/segmenter.pt", classifier="runs/classifier.pt",
                         classifier_kind="finetune", temperature=.05, detector_conf=.4, detector_iou=.5,
                         verify_threshold=.9, min_mask_fraction=.05)
    def tearDown(self): shutil.rmtree(self.tmp)
    def manifest(self, **kw):
        d = dict(self.data, **kw); p = self.tmp / "runs" / "manifest.json"; p.write_text(json.dumps(d)); return p

    def test_valid_setup_passes(self):
        r = pf.run(self.manifest(), expected=self.w)
        self.assertEqual(r["weights_sha256"], self.w)
    def test_missing_manifest(self):
        with self.assertRaisesRegex(pf.PreflightError, "Manifest not found"): pf.run(self.tmp / "none.json")
    def test_invalid_manifest_value(self):
        with self.assertRaisesRegex(pf.PreflightError, "invalid"): pf.run(self.manifest(temperature=0), expected=self.w)
    def test_missing_weights_fail_before_loading(self):
        (self.tmp / "runs" / "segmenter.pt").unlink()
        with self.assertRaisesRegex(pf.PreflightError, "segmenter weights not found"): pf.run(self.manifest(), expected=self.w)
    def test_wrong_weights_hash(self):
        with self.assertRaisesRegex(pf.PreflightError, "differ from the evaluated"): pf.run(self.manifest())
    def test_onnx_detector_rejected(self):
        (self.tmp / "runs" / "d.onnx").write_bytes(b"x")
        with self.assertRaisesRegex(pf.PreflightError, "PyTorch detector"): pf.run(self.manifest(detector="runs/d.onnx"), expected=self.w)
    def test_operating_point_cannot_drift(self):
        with self.assertRaisesRegex(pf.PreflightError, "detector_conf"): pf.run(self.manifest(detector_conf=.3), expected=self.w)
    def test_review_mode_must_be_off(self):
        with self.assertRaisesRegex(pf.PreflightError, "review_mode"): pf.run(self.manifest(review_mode="audit"), expected=self.w)
    def test_existing_output_refused(self):
        with self.assertRaisesRegex(pf.PreflightError, "already exists"): pf.run(self.manifest(), output=self.tmp, expected=self.w)
    def test_empty_image_folder(self):
        (self.tmp / "img").mkdir()
        with self.assertRaisesRegex(pf.PreflightError, "No JPEG"): pf.run(self.manifest(), images=self.tmp / "img", expected=self.w)
    def test_label_mismatch(self):
        img = self.tmp / "img"; img.mkdir(); (img / "a.jpg").write_bytes(b"x")
        lab = self.tmp / "l.csv"; lab.write_text("image,species\nb,cat\n")
        try: import pandas  # noqa: F401
        except ImportError: self.skipTest("pandas unavailable")
        with self.assertRaisesRegex(pf.PreflightError, "Labels do not match"): pf.run(self.manifest(), images=img, labels=lab, expected=self.w)
    def test_version_check_reports_mismatch(self):
        req = self.tmp / "r.txt"; req.write_text("surely-not-installed-pkg==1.0\n")
        self.assertTrue(pf.check_requirements(req))


class EvidenceConsistency(unittest.TestCase):
    def copy(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp)
        shutil.copytree(ROOT / "report" / "evidence", tmp / "report" / "evidence")
        shutil.copy(ROOT / "report" / "main.tex", tmp / "report" / "main.tex"); return tmp
    def failed(self, root): return [n for n, ok, _ in vse.checks(root)[0] if not ok]

    def test_repository_evidence_is_consistent(self):
        self.assertEqual(self.failed(ROOT), [])
    def test_changed_latency_in_report_is_caught(self):
        r = self.copy(); t = r / "report" / "main.tex"; t.write_text(t.read_text().replace("430.8", "330.8"))
        self.assertIn("report shows CPU HTTP percentiles", self.failed(r))
    def test_relaxed_onnx_tolerance_is_caught(self):
        r = self.copy(); p = r / "report" / "evidence" / "M8_onnx_parity_recheck_variants.json"
        d = json.loads(p.read_text()); d["results"][0]["tolerance"] = 0.002; p.write_text(json.dumps(d))
        self.assertIn("A9b tolerance is 0.001 in every variant", self.failed(r))
    def test_claiming_onnx_pass_is_caught(self):
        r = self.copy(); t = r / "report" / "main.tex"; t.write_text(t.read_text() + "\nThe detector export passed, so A9b is met.\n")
        self.assertIn("report never states the ONNX detector passed", self.failed(r))
    def test_demo_included_in_timing_is_caught(self):
        r = self.copy(); p = r / "report" / "evidence" / "P2_cpu_http_validation_mu2tolkg.json"
        d = json.loads(p.read_text()); d["timing_includes_demo"] = True; p.write_text(json.dumps(d))
        self.assertIn("demo requests excluded from timing", self.failed(r))


class OnnxChannelSplit(unittest.TestCase):
    def test_box_and_score_groups_are_separated(self):
        import numpy as np, onnx_channel_diagnosis as d
        ref = np.zeros((1, 6, 10), np.float32); ref[:, :4] = 600.0; ref[:, 4:] = 0.5
        out = ref.copy(); out[0, 1, 3] += 0.002; out[0, 5, 2] += 1e-6
        r = d.group_stats(out, ref)
        self.assertEqual(r["argmax_group"], "box")
        self.assertAlmostEqual(r["max_abs_box"], 0.002, places=4)
        self.assertLess(r["max_abs_score"], 1e-5)
    def test_shape_mismatch_rejected(self):
        import numpy as np, onnx_channel_diagnosis as d
        with self.assertRaises(ValueError): d.group_stats(np.zeros((1, 6, 4)), np.zeros((1, 6, 5)))


if __name__ == "__main__":
    unittest.main()
