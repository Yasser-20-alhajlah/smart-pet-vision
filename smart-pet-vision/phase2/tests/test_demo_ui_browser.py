"""Behavioural browser tests of the GET /demo page (headless Chromium through Playwright).

The real FastAPI app is served on 127.0.0.1 with the model pipeline replaced by a fake, so these
tests need no weights and check only what the page does with the service's answers. All images are
SYNTHETIC solid-colour PNGs made here; the colour selects the fake answer. Skipped, with the reason,
when Playwright or its Chromium is not available (for example in CI with ../requirements.txt only).
"""
import base64
import io
import socket
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from spv import serve
from spv.pipeline import REASONS, InvalidModelOutput

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - depends on the environment
    sync_playwright = None

import uvicorn

READY, NO_ANIMAL, MULTI, BROKEN, BAD_FLAGS, SLOW = (
    (200, 30, 30), (30, 30, 200), (30, 160, 30), (200, 200, 30), (90, 90, 90), (250, 120, 0))


def synthetic_png(rgb, size=(64, 48)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, rgb).save(b, "PNG")
    return b.getvalue()


def tiny_png_b64(rgb) -> str:
    return base64.b64encode(synthetic_png(rgb, (8, 8))).decode()


def reasons(*codes):
    return [{"code": c, "message": REASONS[c]} for c in codes]


class FakePipeline:
    """Returns schema-shaped answers chosen by the colour of the synthetic upload."""

    def __init__(self):
        self.calls = 0

    def analyze(self, im, with_images=True):
        self.calls += 1
        rgb = im.convert("RGB").getpixel((0, 0))
        base = {"staff_approval_required": True, "automatic_publication_allowed": False,
                "image_size": [im.width, im.height], "timings": {"total_ms": 12.0},
                "segmentation_animal_count": 1, "segmentation_species": "cat", "secondary_review": {}}
        if rgb == BROKEN:
            raise InvalidModelOutput("fake invalid output")
        if rgb in (READY, SLOW, BAD_FLAGS):
            if rgb == SLOW:
                time.sleep(1.5)
            r = {**base, "verdict": "ready", "reasons": [], "species": "cat", "confidence": 1.0,
                 "probabilities": {"cat": 1.0, "dog": 0.0},
                 "detections": [{"box_xyxy": [4.0, 4.0, 40.0, 40.0], "species": "cat", "confidence": 0.93}],
                 "mask_fraction": 0.31, "segmentation_matched": True}
            if rgb == BAD_FLAGS:
                r["automatic_publication_allowed"] = True
        elif rgb == NO_ANIMAL:
            r = {**base, "verdict": "refer_to_staff", "reasons": reasons("no_animal"), "species": "dog",
                 "confidence": 0.987, "probabilities": {"cat": 0.013, "dog": 0.987}, "detections": [],
                 "mask_fraction": 0.0, "segmentation_matched": False, "segmentation_animal_count": 0,
                 "segmentation_species": None}
        elif rgb == MULTI:
            r = {**base, "verdict": "refer_to_staff",
                 "reasons": reasons("multiple_animals", "segmentation_disagreement"),
                 "species": "cat", "confidence": 0.999, "probabilities": {"cat": 0.999, "dog": 0.001},
                 "detections": [{"box_xyxy": [1.0, 1.0, 30.0, 30.0], "species": "cat", "confidence": 0.9},
                                {"box_xyxy": [33.0, 5.0, 60.0, 40.0], "species": "dog", "confidence": 0.7}],
                 "mask_fraction": 0.2, "segmentation_matched": True, "segmentation_animal_count": 2,
                 "segmentation_species": "dog"}
        else:
            raise AssertionError(f"unexpected synthetic colour {rgb}")
        if with_images:
            r.update(annotated_png=tiny_png_b64((255, 140, 0)), mask_png=tiny_png_b64((255, 255, 255)),
                     cutout_png=tiny_png_b64((10, 10, 10)))
        return r


# Counts browser object URLs, so the tests can check that they are released.
COUNT_OBJECT_URLS = """
(() => { const c = URL.createObjectURL.bind(URL), r = URL.revokeObjectURL.bind(URL);
  window.__urls = {created: 0, revoked: 0};
  URL.createObjectURL = (o) => { window.__urls.created++; return c(o); };
  URL.revokeObjectURL = (u) => { window.__urls.revoked++; return r(u); }; })();
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipIf(sync_playwright is None, "Playwright is not installed; browser checks not executed")
class DemoPageInBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakePipeline()
        cls.patcher = patch.object(serve, "pipeline", lambda: cls.fake)
        cls.patcher.start()
        port = _free_port()
        cls.base = f"http://127.0.0.1:{port}"
        cls.server = uvicorn.Server(uvicorn.Config(serve.app, host="127.0.0.1", port=port, log_level="warning"))
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        t0 = time.time()
        while not cls.server.started and time.time() - t0 < 20:
            time.sleep(0.05)
        cls.pw = sync_playwright().start()
        try:
            cls.browser = cls.pw.chromium.launch()
        except Exception as exc:  # pragma: no cover - depends on the environment
            cls.pw.stop()
            cls.stop_server()
            raise unittest.SkipTest(f"Chromium could not be launched ({type(exc).__name__}); browser checks not executed")

    @classmethod
    def stop_server(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.patcher.stop()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.stop_server()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.page = self.context.new_page()
        self.page.set_default_timeout(15000)
        self.page.add_init_script(COUNT_OBJECT_URLS)
        self.requests = []
        self.page.on("request", lambda req: self.requests.append(req.url))
        self.page.goto(self.base + "/demo")

    def tearDown(self):
        self.context.close()

    # ---------------------------------------------------------------- helpers
    def choose(self, name, data, mime="image/png"):
        self.page.set_input_files("#file", files=[{"name": name, "mimeType": mime, "buffer": data}])

    def analyse(self, name, data, mime="image/png"):
        self.choose(name, data, mime)
        self.page.click("#analyze")

    def wait_done(self):
        self.page.wait_for_function("document.getElementById('status').textContent.indexOf('Done') === 0")

    def wait_error(self):
        self.page.wait_for_selector("#status.error")

    def text(self, selector):
        return self.page.inner_text(selector)

    def results_visible(self):
        return self.page.is_visible("#results")

    def urls(self):
        return self.page.evaluate("window.__urls")

    # ---------------------------------------------------------------- tests
    def test_page_loads_without_models_and_without_external_requests(self):
        calls_before_page = self.fake.calls
        self.assertIn("Staff approval required for every photo", self.text(".policy"))
        self.assertIn("Automatic publication disabled", self.text(".policy"))
        self.assertTrue(self.page.is_disabled("#analyze"))
        self.assertEqual(self.fake.calls, calls_before_page)  # loading the page called no model
        self.analyse("synthetic_ready.png", synthetic_png(READY))
        self.wait_done()
        external = [u for u in self.requests if not u.startswith((self.base, "data:", "blob:"))]
        self.assertEqual(external, [])
        self.assertEqual(self.page.evaluate("localStorage.length + sessionStorage.length"), 0)

    def test_ready_result(self):
        self.analyse("synthetic_ready.png", synthetic_png(READY))
        self.wait_done()
        self.assertEqual(self.text("#verdict h2"), "Ready for staff review — not approved for publication")
        facts = self.text("#facts")
        self.assertIn("Species (classifier)", facts)
        self.assertIn("score 1.000", facts)
        self.assertIn("even 1.000 is not proof", facts)
        self.assertIn("staff_approval_required = true", facts)
        self.assertIn("automatic_publication_allowed = false", facts)
        captions = self.page.locator("figcaption strong").all_inner_texts()
        self.assertEqual(captions, ["Original", "Detections and mask", "Mask", "Cutout"])
        self.assertEqual(self.page.locator("#dets tbody tr").count(), 1)
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "verdict")
        self.assertEqual(self.page.locator("button", has_text="Approve").count(), 0)  # no cosmetic approval

    def test_no_animal_hides_identification(self):
        self.analyse("synthetic_blank.png", synthetic_png(NO_ANIMAL))
        self.wait_done()
        self.assertEqual(self.text("#verdict h2"), "Staff review required — see reasons")
        self.assertIn("no_animal", self.text("#verdict"))
        facts = self.text("#facts")
        self.assertIn("Not shown: no animal was detected", facts)
        self.assertNotIn("Species", facts)
        self.assertNotIn("dog", facts)
        self.assertNotIn("0.987", facts)
        self.assertFalse(self.page.is_visible("#detcard"))
        self.assertEqual(self.page.locator("#images img").count(), 1)  # only the original
        self.page.click("details summary")  # the raw JSON keeps the API fields, with a note
        self.assertIn("do not identify an animal", self.text("#rawnote"))

    def test_referral_with_multiple_reasons_marks_species_provisional(self):
        self.analyse("synthetic_two_reasons.png", synthetic_png(MULTI))
        self.wait_done()
        self.assertEqual(self.text("#verdict h2"), "Staff review required — see reasons")
        items = self.page.locator("#verdict li").all_inner_texts()
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0].startswith("multiple_animals: "))
        self.assertTrue(items[1].startswith("segmentation_disagreement: "))
        facts = self.text("#facts")
        self.assertIn("Species (classifier, provisional)", facts)
        self.assertIn("Check the reasons above", facts)
        self.assertEqual(self.page.locator("#dets tbody tr").count(), 2)

    def test_invalid_upload_after_success_leaves_no_stale_result(self):
        self.analyse("synthetic_ready.png", synthetic_png(READY))
        self.wait_done()
        self.analyse("notes.png", b"this is a text file, not an image")
        self.wait_error()
        self.assertIn("HTTP 415", self.text("#status"))
        self.assertFalse(self.results_visible())
        self.assertEqual(self.page.locator("#images > *").count(), 0)
        self.assertEqual(self.text("#verdict").strip(), "")
        u = self.urls()
        self.assertEqual(u["created"], u["revoked"])

    def test_browser_side_limits_block_empty_and_oversized_files(self):
        before = self.fake.calls
        self.choose("empty.png", b"")
        self.assertIn("empty", self.text("#status"))
        self.assertTrue(self.page.is_disabled("#analyze"))
        self.choose("too_big.png", b"0" * (15 * 1024 * 1024 + 1))
        self.assertIn("larger than 15 MB", self.text("#status"))
        self.assertTrue(self.page.is_disabled("#analyze"))
        self.assertEqual(self.fake.calls, before)

    def test_over_dimensioned_image_is_refused_by_the_service(self):
        b = io.BytesIO()
        Image.new("L", (6000, 5000)).save(b, "PNG")  # 30 megapixels in a small file
        before = self.fake.calls
        self.analyse("synthetic_30_megapixels.png", b.getvalue())
        self.wait_error()
        self.assertIn("HTTP 413", self.text("#status"))
        self.assertIn("25 megapixels", self.text("#status"))
        self.assertEqual(self.fake.calls, before)
        self.assertFalse(self.results_visible())

    def test_invalid_model_output_is_shown_as_503_without_a_result(self):
        self.analyse("synthetic_invalid_output.png", synthetic_png(BROKEN))
        self.wait_error()
        status = self.text("#status")
        self.assertIn("HTTP 503", status)
        self.assertIn("Staff review required", status)
        self.assertFalse(self.results_visible())

    def test_service_unreachable_after_a_result_hides_the_old_result(self):
        self.analyse("synthetic_ready.png", synthetic_png(READY))
        self.wait_done()
        self.page.route("**/analyze*", lambda route: route.abort())  # simulated network failure
        self.page.click("#analyze")  # same file again
        self.wait_error()
        self.assertIn("could not be reached", self.text("#status"))
        self.assertFalse(self.results_visible())
        self.assertFalse(self.page.is_disabled("#analyze"))  # the user can retry

    def test_unexpected_policy_flags_raise_an_alert(self):
        self.analyse("synthetic_bad_flags.png", synthetic_png(BAD_FLAGS))
        self.wait_done()
        self.assertIn("Unexpected policy flags", self.text("#verdict h2"))
        self.assertNotIn("Ready for staff review", self.text("#verdict"))

    def test_duplicate_submission_sends_one_request(self):
        self.choose("synthetic_slow.png", synthetic_png(SLOW))
        before = self.fake.calls
        self.page.evaluate("() => { const b = document.getElementById('analyze'); b.click(); b.click(); b.click(); }")
        self.assertTrue(self.page.is_disabled("#analyze"))
        self.wait_done()
        self.assertEqual(self.fake.calls, before + 1)

    def test_new_image_during_a_request_discards_the_old_answer(self):
        self.analyse("synthetic_slow.png", synthetic_png(SLOW))
        self.choose("synthetic_blank.png", synthetic_png(NO_ANIMAL))
        self.page.wait_for_timeout(2500)  # longer than the slow answer
        self.assertFalse(self.results_visible())
        self.assertEqual(self.text("#status"), "Ready to analyse.")
        self.page.click("#analyze")
        self.wait_done()
        self.assertIn("no_animal", self.text("#verdict"))

    def test_reset_and_repeated_uploads(self):
        for name, rgb, heading in (("synthetic_ready.png", READY, "Ready for staff review"),
                                   ("synthetic_blank.png", NO_ANIMAL, "Staff review required"),
                                   ("synthetic_two_reasons.png", MULTI, "Staff review required"),
                                   ("synthetic_ready.png", READY, "Ready for staff review")):
            self.analyse(name, synthetic_png(rgb))
            self.wait_done()
            self.assertTrue(self.text("#verdict h2").startswith(heading))
            self.assertEqual(self.page.locator("#images figure").count(), 4)
            u = self.urls()
            self.assertEqual(u["created"] - u["revoked"], 1)  # only the visible original is held
        self.page.click("#clear")
        self.assertFalse(self.results_visible())
        self.assertEqual(self.page.evaluate("document.getElementById('file').value"), "")
        self.assertTrue(self.page.is_disabled("#analyze"))
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "file")
        u = self.urls()
        self.assertEqual(u["created"], u["revoked"])

    def test_keyboard_only_flow(self):
        self.page.keyboard.press("Tab")
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "file")
        # A file dialog cannot be driven headless; the file is set on the focused input instead.
        self.choose("synthetic_ready.png", synthetic_png(READY))
        self.page.keyboard.press("Tab")
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "analyze")
        self.page.keyboard.press("Enter")
        self.wait_done()
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "verdict")
        self.page.keyboard.press("Shift+Tab")
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "clear")
        self.page.keyboard.press("Enter")
        self.assertFalse(self.results_visible())
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "file")
        labelled = self.page.evaluate("""() => [...document.querySelectorAll('button, input')]
            .every(e => (e.labels && e.labels.length) || e.textContent.trim() || e.getAttribute('aria-label'))""")
        self.assertTrue(labelled)

    def test_narrow_screen_has_no_horizontal_scroll(self):
        self.page.set_viewport_size({"width": 360, "height": 740})
        self.analyse("synthetic_two_reasons.png", synthetic_png(MULTI))
        self.wait_done()
        widths = self.page.evaluate("[document.documentElement.scrollWidth, document.documentElement.clientWidth]")
        self.assertLessEqual(widths[0], widths[1])
        for sel in ("#analyze", "#clear", "#verdict h2"):
            box = self.page.locator(sel).bounding_box()
            self.assertLessEqual(box["x"] + box["width"], 360)


if __name__ == "__main__":
    unittest.main()
