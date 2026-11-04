"""Release packaging: the code archive excludes PDFs, refuses private IDs, and the source ZIP holds the same code."""
import os, shutil, sys, tempfile, unittest, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission" / "tools"))
import make_code_archive as mca


class CodeArchive(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pdfs_are_excluded_and_listed(self):
        kept, excluded = mca.select({"a/run.py": b"x", "a/report.PDF": b"%PDF", "b/x.json": b"{}"})
        self.assertEqual(sorted(kept), ["a/run.py", "b/x.json"])
        self.assertIn("a/report.PDF", excluded)

    def test_private_ids_parsed_and_detected_as_whole_tokens_only(self):
        ids = self.tmp / "private_ids.tex"
        ids.write_text("\\newcommand{\\idA}{424242}\n\\newcommand{\\idB}{515151}\n")
        tokens = mca.private_tokens(ids)
        self.assertEqual(tokens, {"424242", "515151"})
        members = {"ok_hash.txt": b"sha f0a424242b1c", "ok_long.txt": b"1424242", "leak.tex": b"ID: 515151.",
                   "leak.json": b'{"v": 0.424242}'}
        self.assertEqual(mca.find_private(members, tokens), ["leak.json", "leak.tex"])
        self.assertEqual(mca.private_tokens(self.tmp / "missing.tex"), set())

    def test_source_zip_code_folder_is_identical_to_code_archive(self):
        kept = {"phase2/src/a.py": b"print(1)\n", "requirements.txt": b"torch==2.11.0\n"}
        code = self.tmp / "reviewed_source.zip"
        mca.write_zip(code, kept, mca.index(kept, {}, "c" * 40, "t" * 40))
        (self.tmp / "main.tex").write_text("\\documentclass{article}")
        src = self.tmp / "source.zip"
        cwd = os.getcwd(); os.chdir(self.tmp)
        try:
            mca.write_source_zip(src, code, ["main.tex"])
        finally:
            os.chdir(cwd)
        self.assertTrue(mca.code_members_identical(src, code))
        with zipfile.ZipFile(src) as z:
            self.assertIn("main.tex", z.namelist())
            self.assertIn("code/" + mca.INDEX_NAME, z.namelist())
        tampered = self.tmp / "tampered.zip"
        with zipfile.ZipFile(src) as s, zipfile.ZipFile(tampered, "w") as t:
            for n in s.namelist():
                t.writestr(n, b"changed" if n == "code/phase2/src/a.py" else s.read(n))
        self.assertFalse(mca.code_members_identical(tampered, code))


if __name__ == "__main__":
    unittest.main()
