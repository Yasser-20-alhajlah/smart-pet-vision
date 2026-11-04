"""GET /demo: the browser page used for the live demonstration (presentation layer over POST /analyze)."""
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

    def test_no_animal_identification_is_not_displayed(self):
        # Display rule: for a no_animal referral the page shows no species or confidence.
        text = self.get().text
        self.assertIn('codes.indexOf("no_animal") !== -1', text)
        self.assertIn("Not shown: no animal was detected", text)

    def test_api_contract_unchanged(self):
        paths = {route.path for route in serve.app.routes}
        self.assertTrue({"/analyze", "/health", "/demo"} <= paths)
        schema_paths = TestClient(serve.app).get("/openapi.json").json()["paths"]
        self.assertNotIn("/demo", schema_paths)


if __name__ == "__main__":
    unittest.main()
