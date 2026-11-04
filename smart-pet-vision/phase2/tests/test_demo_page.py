"""GET /demo served by the app (static checks, no browser). Behaviour: tests/test_demo_ui_browser.py."""
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from spv import serve


class DemoPage(unittest.TestCase):
    def get(self):
        # The page must not load the models; loading them here would fail the test.
        with patch.object(serve, "pipeline", side_effect=AssertionError("models loaded")):
            return TestClient(serve.app).get("/demo")

    def test_served_as_html_without_loading_models(self):
        response = self.get()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertIn('fetch("/analyze?images=true"', response.text)
        self.assertIn("Staff approval required", response.text)
        self.assertIn("Automatic publication disabled", response.text)

    def test_no_external_resources(self):
        # The demo must work on a presentation host without internet access.
        text = self.get().text
        self.assertIsNone(re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", text))
        self.assertNotIn("@import", text)

    def test_static_display_rules_present(self):
        # Static fallback for environments without a browser; the behaviour itself is checked in
        # tests/test_demo_ui_browser.py (Playwright), which this test does not replace.
        text = self.get().text
        self.assertIn('codes.indexOf("no_animal") !== -1', text)
        self.assertIn("Not shown: no animal was detected", text)
        self.assertIn("Ready for staff review \u2014 not approved for publication", text)
        self.assertIn("Staff review required \u2014 see reasons", text)
        self.assertNotIn("innerHTML", text)  # API messages and filenames are inserted as text only
        self.assertNotIn("localStorage", text)

    def test_api_contract_unchanged(self):
        paths = {route.path for route in serve.app.routes}
        self.assertTrue({"/analyze", "/health", "/demo"} <= paths)
        schema_paths = TestClient(serve.app).get("/openapi.json").json()["paths"]
        self.assertNotIn("/demo", schema_paths)


if __name__ == "__main__":
    unittest.main()
