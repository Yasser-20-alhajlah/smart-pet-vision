"""spv.provenance: commit/dirty state inside Git, archive index outside it, stable fingerprint."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from spv import provenance

PHASE2 = Path(__file__).resolve().parent.parent


class ProvenanceTest(unittest.TestCase):
    def test_record_in_git_checkout_names_commit(self):
        p = provenance.record(PHASE2)
        if p.get("code_source") != "git":
            self.skipTest("not a git checkout")
        self.assertEqual(len(p["commit"]), 40)
        self.assertIn("dirty", p)
        self.assertEqual(p["source_fingerprint"]["sha256"].__len__(), 64)
        self.assertGreater(p["source_fingerprint"]["files"], 10)
        json.dumps(p)  # JSON-serialisable

    def test_record_outside_git_uses_archive_index_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase2"
            (root / "src" / "spv").mkdir(parents=True)
            shutil.copy(PHASE2 / "src" / "spv" / "provenance.py", root / "src" / "spv" / "provenance.py")
            p = provenance.record(root)
            self.assertEqual(p["code_source"], "unversioned copy")
            self.assertEqual(p["source_fingerprint"]["files"], 1)
            (root / "CODE_ARCHIVE_INDEX.json").write_text(json.dumps({"source_commit": "a" * 40, "source_tree": "b" * 40}))
            p2 = provenance.record(root, extra={"split_version": "v1"})
            self.assertEqual(p2["code_source"], "archive")
            self.assertEqual(p2["commit"], "a" * 40)
            self.assertEqual(p2["split_version"], "v1")

    def test_nested_delivered_archive_records_published_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            root = code / "smart-pet-vision" / "phase2"
            (root / "src" / "spv").mkdir(parents=True)
            (root / "src" / "spv" / "example.py").write_text("value = 1\n")
            index = code / "CODE_ARCHIVE_INDEX.json"
            index.write_text(json.dumps({"source_commit": "c" * 40,
                                         "source_tree": "d" * 40}))
            result = provenance.record(root)
            self.assertEqual(result["code_source"], "archive")
            self.assertEqual(result["commit"], "c" * 40)
            self.assertEqual(result["tree"], "d" * 40)
            self.assertEqual(Path(result["archive_index"]), index)
            self.assertEqual(result["source_fingerprint"]["files"], 1)

    def test_fingerprint_changes_when_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "phase2"
            (root / "tools").mkdir(parents=True)
            (root / "tools" / "a.py").write_text("x = 1\n")
            f1 = provenance.source_fingerprint(root)["sha256"]
            self.assertEqual(f1, provenance.source_fingerprint(root)["sha256"])
            (root / "tools" / "a.py").write_text("x = 2\n")
            self.assertNotEqual(f1, provenance.source_fingerprint(root)["sha256"])

    def test_cli_prints_json(self):
        out = subprocess.run(["python3", str(PHASE2 / "src" / "spv" / "provenance.py")],
                             capture_output=True, text=True, cwd=str(PHASE2))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("source_fingerprint", json.loads(out.stdout))


if __name__ == "__main__":
    unittest.main()
