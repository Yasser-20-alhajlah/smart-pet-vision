"""Reject self-consistent but unsupported results and a failed release gate."""

import importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "submission/tools")]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


v = load("verify_review", TARGET / "tools/verify_v2_saved_results.py")
r = load("reproduce_review", TARGET / "tools/reproduce_v2_eval.py")
m = load("manifest_review", TARGET / "submission/tools/make_release_manifest.py")
b = load("benchmark_review", TARGET / "tools/benchmark_http_release.py")


class SavedOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.e = self.root / "report/evidence"
        self.e.mkdir(parents=True)
        shutil.copytree(
            ROOT / "report/evidence/v2_saved_review_20261114",
            self.e / "v2_saved_review_20261114",
        )
        self.hp = self.e / "P2_v2_headline_three_seeds_20261113.json"
        shutil.copy2(ROOT / "report/evidence" / self.hp.name, self.hp)

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self):
        return all(x["passed"] for x in v.verify(self.root))

    def test_current_records_pass(self):
        self.assertTrue(self.verify())

    def test_wrong_headline_miou_rejected_even_when_self_consistent(self):
        h = json.loads(self.hp.read_text())
        for s in h["seeds"].values():
            s["segmenter"]["mIoU"] = 0.99
        h["summary_over_seeds"]["mIoU"].update(mean=0.99, sd=0, min=0.99, max=0.99)
        self.hp.write_text(json.dumps(h))
        self.assertFalse(self.verify())

    def test_wrong_headline_f1_rejected(self):
        h = json.loads(self.hp.read_text())
        for s in h["seeds"].values():
            s["detector"]["test_at_operating_point"]["f1"] = 0.5
        h["summary_over_seeds"]["detector_f1_at_operating_point"].update(
            mean=0.5, sd=0, min=0.5, max=0.5
        )
        self.hp.write_text(json.dumps(h))
        self.assertFalse(self.verify())

    def test_wrong_record_accuracy_rejected(self):
        p = self.e / "v2_saved_review_20261114/seed44_record.json"
        a = json.loads(p.read_text())
        a["classifier"]["test"]["accuracy"] = 0.5
        p.write_text(json.dumps(a))
        self.assertFalse(self.verify())

    def test_wrong_summary_range_rejected(self):
        h = json.loads(self.hp.read_text())
        h["summary_over_seeds"]["mIoU"]["min"] = 0
        self.hp.write_text(json.dumps(h))
        self.assertFalse(self.verify())


class CohortTests(unittest.TestCase):
    def test_duplicate_image_stems_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "images/test"
            p.mkdir(parents=True)
            for n in ["pet.jpg", "pet.png"]:
                (p / n).write_bytes(b"")
            spec = {"path": d, "test": "images/test", "names": {0: "cat", 1: "dog"}}
            if not hasattr(r, "validate_yolo_test"):
                self.fail("No duplicate-safe cohort helper")
            with self.assertRaises(ValueError):
                r.validate_yolo_test(spec, Path(d) / "data.yaml", {"pet"}, [])

    def test_changed_detector_label_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            images = root / "images/test"
            images.mkdir(parents=True)
            (images / "pet.jpg").write_bytes(b"fixture")
            labels = root / "labels/test"
            labels.mkdir(parents=True)
            (labels / "pet.txt").write_text("1 0.5 0.5 1 1")
            row = {
                "image": "pet",
                "split": "test",
                "label": "0",
                "width": "10",
                "height": "10",
                "x1": "0",
                "y1": "0",
                "x2": "10",
                "y2": "10",
            }
            with self.assertRaisesRegex(ValueError, "Detector test label differs"):
                r.validate_yolo_test(
                    {"path": d, "test": "images/test", "names": ["cat", "dog"]},
                    root / "data.yaml",
                    {"pet"},
                    [row],
                    True,
                )


class HTTPContractTests(unittest.TestCase):
    def test_success_without_staff_guard_is_not_accepted(self):
        class Response:
            status_code = 200

            def json(self):
                return {"verdict": "ready"}

        self.assertFalse(b.valid_response(Response()))

    def test_complete_staff_guard_is_accepted(self):
        class Response:
            status_code = 200

            def json(self):
                return {
                    "verdict": "ready",
                    "staff_approval_required": True,
                    "automatic_publication_allowed": False,
                }

        self.assertTrue(b.valid_response(Response()))

    def test_non_json_is_not_accepted(self):
        class Response:
            status_code = 200

            def json(self):
                raise ValueError("invalid JSON")

        self.assertFalse(b.valid_response(Response()))


class ManifestTests(unittest.TestCase):
    def test_failed_checker_aborts_release(self):
        self.probe(False)

    def test_code_mismatch_aborts_release(self):
        self.probe(True)

    def probe(self, tamper):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "SmartPetVision_Phase2.pdf").write_bytes(b"test fixture")
            commit = m.git("rev-parse", "HEAD")
            tree = m.git("rev-parse", "HEAD^{tree}")
            idx = {
                "source_commit": commit,
                "source_tree": tree,
                "members": {},
                "excluded": {},
            }
            with zipfile.ZipFile(p / "reviewed_source.zip", "w") as z:
                z.writestr("README.md", b"fixture")
                z.writestr("CODE_ARCHIVE_INDEX.json", json.dumps(idx))
            with (
                zipfile.ZipFile(p / "reviewed_source.zip") as z,
                zipfile.ZipFile(p / "SmartPetVision_Phase2_source.zip", "w") as out,
            ):
                for name in z.namelist():
                    out.writestr("code/" + name, z.read(name))
            if tamper:
                s = p / "SmartPetVision_Phase2_source.zip"
                t = p / "new.zip"
                with zipfile.ZipFile(s) as z, zipfile.ZipFile(t, "w") as out:
                    for n in z.namelist():
                        out.writestr(
                            n, b"changed" if n == "code/README.md" else z.read(n)
                        )
                t.replace(s)
            real = subprocess.run

            def run(args, *a, **kw):
                if args == ["pdflatex", "--version"]:
                    return subprocess.CompletedProcess(
                        args, 0, stdout="stub pdfTeX\n", stderr=""
                    )
                return real(args, *a, **kw)

            checks = {
                "unit_tests": {"returncode": 0 if tamper else 1},
                "evidence_checks": {"returncode": 0},
                "saved_output_checks": {"returncode": 0},
            }
            with (
                patch.object(m, "SUB", p),
                patch.object(m, "PHASE2", ROOT),
                patch.object(m, "run_checks", return_value=checks),
                patch.object(subprocess, "run", side_effect=run),
                patch.object(sys, "argv", ["manifest"]),
            ):
                with self.assertRaises(SystemExit):
                    m.main()
            self.assertTrue((p / "RELEASE_MANIFEST.failed.json").exists())
            self.assertFalse((p / "RELEASE_MANIFEST.json").exists())


if __name__ == "__main__":
    unittest.main()
