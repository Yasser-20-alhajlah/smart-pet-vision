"""A failed preflight must never start a real-weight HTTP process."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spv_smoke_http", ROOT / "tools/smoke_test_http.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

class SmokePreflightFailure(unittest.TestCase):
    def check_failure(self, allow_drift):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            # Deliberately missing weights: preflight must get the first decision.
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({k: "missing.pt" for k in ("detector", "segmenter", "classifier")}))
            out = folder / "new-directory/result.json"
            args = ["smoke_test_http.py", "--manifest", str(manifest), "--image", str(folder / "cat.png"), "--out", str(out)]
            if allow_drift:
                args.append("--allow-version-drift")
            original = Path.cwd()
            try:
                with patch.object(sys, "argv", args), patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "PREFLIGHT FAILED: missing weights")) as preflight, patch.object(smoke.subprocess, "Popen") as launch:
                    with self.assertRaises(SystemExit) as raised:
                        smoke.main()
                    self.assertEqual(raised.exception.code, 2)
                    launch.assert_not_called()
                    self.assertEqual("--strict-versions" in preflight.call_args.args[0], not allow_drift)
                receipt = json.loads(out.read_text())
                self.assertEqual(receipt["requests_executed"], 0)
                self.assertEqual(receipt["status"], "PREFLIGHT_FAILED")
                self.assertEqual(receipt["allow_version_drift"], allow_drift)
            finally:
                os.chdir(original)

    def test_default_preflight_failure_stops_before_http(self):
        self.check_failure(False)

    def test_allow_drift_still_stops_on_other_preflight_failure(self):
        self.check_failure(True)
