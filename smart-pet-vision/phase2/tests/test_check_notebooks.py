"""The notebooks are valid nbformat JSON and every code cell parses (tools/check_notebooks.py)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PHASE2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE2 / "tools"))

import check_notebooks  # noqa: E402


class CheckNotebooksTest(unittest.TestCase):
    def test_strip_ipython_keeps_indentation(self):
        src = "if IN_COLAB:\n    !pip -q install -r requirements.txt\nx = 1\n%matplotlib inline\n"
        stripped = check_notebooks.strip_ipython(src)
        self.assertEqual(stripped, "if IN_COLAB:\n    pass\nx = 1\npass\n")
        compile(stripped, "<nb>", "exec")

    def test_syntax_error_is_reported(self):
        nb = {"nbformat": 4, "nbformat_minor": 5, "cells": [
            {"cell_type": "code", "source": ["def broken(:\n"], "execution_count": None}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.ipynb"
            p.write_text(json.dumps(nb))
            rec = check_notebooks.check_notebook(p)
        self.assertTrue(rec["valid_json"])
        self.assertEqual(len(rec["syntax_errors"]), 1)

    def test_all_project_notebooks_parse(self):
        records = [check_notebooks.check_notebook(p) for p in sorted((PHASE2 / "notebooks").glob("*.ipynb"))]
        self.assertGreaterEqual(len(records), 15)
        bad = [(r["notebook"], r["syntax_errors"]) for r in records if not r["valid_json"] or r["syntax_errors"]]
        self.assertEqual(bad, [])

    def test_cli_exit_status(self):
        out = subprocess.run([sys.executable, str(PHASE2 / "tools" / "check_notebooks.py")],
                             capture_output=True, text=True, cwd=str(PHASE2))
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("notebooks parse", out.stdout)


if __name__ == "__main__":
    unittest.main()
