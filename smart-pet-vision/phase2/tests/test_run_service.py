"""tools/run_service.py: refuses non-loopback hosts and missing prerequisites, and never starts uvicorn then."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import run_service as rs  # noqa: E402


def run(*argv):
    out = io.StringIO()
    with patch.object(sys, "argv", ["run_service.py", *argv]), patch.object(rs, "serve") as popen, \
            contextlib.redirect_stdout(out):
        code = rs.main()
    return code, out.getvalue(), popen


class Launcher(unittest.TestCase):
    def test_public_address_is_refused(self):
        for host in ("0.0.0.0", "192.168.1.10", "::"):
            code, out, popen = run("--host", host, "--manifest", "no_such_manifest.json")
            self.assertEqual(code, 1)
            self.assertIn("FAIL  loopback address only", out)
            popen.assert_not_called()

    def test_missing_manifest_stops_before_start(self):
        code, out, popen = run("--manifest", "no_such_manifest.json")
        self.assertEqual(code, 1)
        self.assertIn("FAIL  manifest present", out)
        self.assertIn("Not started", out)
        popen.assert_not_called()

    def test_never_installs_or_downloads(self):
        source = (ROOT / "tools" / "run_service.py").read_text()
        for forbidden in ("pip install", "urlopen", "requests.", "httpx.", "gdown", "wandb.login", "0.0.0.0\""):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
