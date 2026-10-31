"""Execution-review regressions. Synthetic inference outputs; no trained weights.

HTTP tests exercise the actual ASGI app with a mocked model, not model accuracy.
"""
import asyncio
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from spv import data, segment, serve
from spv.instance_alignment import pair_detection_boxes
from spv.pipeline import Manifest
from spv.runtime import environment_record
import importlib.util
_validator_spec = importlib.util.spec_from_file_location(
    'release_validator', Path(__file__).resolve().parents[1] / 'tools/validate_release.py')
_validator = importlib.util.module_from_spec(_validator_spec)
_validator_spec.loader.exec_module(_validator)


def image_bytes():
    b = io.BytesIO()
    Image.new('RGB', (8, 5)).save(b, format='PNG')
    return b.getvalue()


class DetectionParityAssociation(unittest.TestCase):
    def test_candidate_cannot_be_reused_for_two_references(self):
        a = np.array([[0, 0, 10, 10], [1, 0, 11, 10]])
        b = np.array([[0, 0, 10, 10], [100, 0, 110, 10]])
        j = pair_detection_boxes(a, b)
        self.assertEqual(len(set(j)), 2)
        self.assertGreater(np.abs(a - b[j]).max(), 90)

    def test_reordered_identical_boxes_have_zero_displacement(self):
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]])
        j = pair_detection_boxes(a, a[::-1])
        np.testing.assert_array_equal(a, a[::-1][j])

    def test_empty_sets(self):
        self.assertEqual(pair_detection_boxes(np.empty((0, 4)), np.empty((0, 4))).size, 0)

    def test_invalid_shapes_or_coordinates_are_not_parity(self):
        for a, b in [([[0, 0, 1, 1]], []), ([[0, 0, np.nan, 1]], [[0, 0, 1, 1]]),
                     ([[1, 0, 0, 1]], [[0, 0, 1, 1]])]:
            with self.assertRaises(ValueError):
                pair_detection_boxes(a, b)


class EvaluationIntegrity(unittest.TestCase):
    def test_release_cohort_must_cover_every_image_exactly_once(self):
        paths = [Path('a.jpg'), Path('b.jpg')]
        good = pd.DataFrame({'image': ['b', 'a'], 'species': ['dog', 'cat']})
        _validator.validate_label_coverage(paths, good)
        for invalid in [good.iloc[:1], good.assign(image=['a', 'c']),
                        good.assign(image=['a', 'a']), good.assign(image=[None, 'b'])]:
            with self.subTest(labels=invalid.to_dict('list')):
                with self.assertRaises(ValueError):
                    _validator.validate_label_coverage(paths, invalid)

    def test_release_cohort_rejects_invalid_truth_schema(self):
        for labels in [pd.DataFrame({'image': ['a']}),
                       pd.DataFrame({'image': ['a'], 'species': ['unknown']}),
                       pd.DataFrame({'image': ['a'], 'species': [None]})]:
            with self.assertRaises(ValueError):
                _validator.validate_label_coverage([Path('a.jpg')], labels)

    def test_mask_alignment_error_is_not_resized_into_a_score(self):
        frame = pd.DataFrame([dict(image='x', species='cat', breed='test')])
        with patch.object(segment, 'load_gt', return_value=(np.ones((2, 2), bool), np.zeros((2, 2), bool))):
            with self.assertRaisesRegex(ValueError, 'Unaligned'):
                segment.evaluate({'x': np.ones((1, 2), bool)}, frame, Path('.'))

    def test_missing_or_duplicate_records_fail_explicitly(self):
        frame = pd.DataFrame([dict(image='x', species='cat', breed='test')])
        with patch.object(segment, 'load_gt', return_value=(np.ones((2, 2), bool), np.zeros((2, 2), bool))):
            with self.assertRaisesRegex(ValueError, 'Missing'):
                segment.evaluate({}, frame, Path('.'))
            with self.assertRaises(ValueError):
                segment.evaluate({}, pd.concat([frame, frame]), Path('.'))

    def test_body_fraction_differs_from_box_area(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); tri = root/'annotations/trimaps'; tri.mkdir(parents=True)
            Image.fromarray(np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2]], np.uint8)).save(tri/'x.png')
            frame = pd.DataFrame([dict(image='x', width=3, height=3)])
            out = data.build_annotations(root, frame, root/'processed')
            self.assertAlmostEqual(out.iloc[0].mask_fraction, 3/9)
            self.assertAlmostEqual(out.iloc[0].box_fraction, 4/9)


class ManifestIntegrity(unittest.TestCase):
    def test_legacy_defaults_and_new_explicit_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); p = root/'manifest.json'
            values = dict(detector='d.pt', segmenter='s.pt', classifier='c.pt', classifier_kind='finetune',
                          temperature=.05, detector_conf=.4, detector_iou=.5,
                          verify_threshold=.9, min_mask_fraction=.05)
            p.write_text(json.dumps(values))
            m = Manifest.load(p)
            self.assertEqual((m.segmenter_conf, m.segmenter_iou, m.mask_match_iou), (.25, .7, .1))
            m.save(p)
            self.assertIn('segmenter_conf', json.loads(p.read_text()))
            for invalid in [dict(imgsz=0), dict(image_size=True), dict(mask_match_iou=0),
                            dict(segmenter_conf=float('nan')), dict(classifier_kind='unknown')]:
                p.write_text(json.dumps(values | invalid))
                with self.assertRaises(ValueError):
                    Manifest.load(p)


class HTTPIntegrity(unittest.TestCase):
    def test_actual_asgi_validation_statuses(self):
        client = TestClient(serve.app)
        with patch.object(serve, 'pipeline') as model:
            model.return_value.analyze.return_value = {'verdict': 'ready', 'staff_approval_required': True,
                                                      'automatic_publication_allowed': False}
            for content, status in [(b'', 400), (b'not an image', 415), (image_bytes(), 200)]:
                r = client.post('/analyze?images=false', files={'file': ('x.png', content, 'image/png')})
                self.assertEqual(r.status_code, status)
            self.assertEqual(model.return_value.analyze.call_count, 1)
            with patch.object(serve, 'MAX_PIXELS', 10):
                r = client.post('/analyze', files={'file': ('x.png', image_bytes())})
                self.assertEqual(r.status_code, 413)
            with patch.object(serve, 'MAX_BYTES', 4):
                r = client.post('/analyze', files={'file': ('x.png', b'12345')})
                self.assertEqual(r.status_code, 413)

    def test_decoding_preserves_exif_for_pipeline_orientation(self):
        b = io.BytesIO(); exif = Image.Exif(); exif[274] = 6
        Image.new('RGB', (8, 5)).save(b, format='JPEG', exif=exif)
        with patch.object(serve, 'pipeline') as model:
            serve._decode_and_analyze(b.getvalue(), False)
            im = model.return_value.analyze.call_args.args[0]
            self.assertEqual(im.getexif()[274], 6)

    def test_inference_does_not_block_event_loop_and_shared_models_are_serial(self):
        active = 0; peak = 0; lock = threading.Lock()
        def slow_model(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            time.sleep(.08)
            with lock:
                active -= 1
            return {'verdict': 'ready', 'staff_approval_required': True,
                    'automatic_publication_allowed': False}
        async def run():
            transport = httpx.ASGITransport(app=serve.app)
            async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
                tasks = [asyncio.create_task(client.post('/analyze', files={'file': ('x.png', image_bytes())}))
                         for _ in range(3)]
                await asyncio.sleep(.01)
                self.assertFalse(any(t.done() for t in tasks), 'A blocking endpoint starved the event loop')
                responses = await asyncio.gather(*tasks)
                self.assertTrue(all(r.status_code == 200 for r in responses))
        with patch.object(serve, 'pipeline') as model:
            model.return_value.analyze.side_effect = slow_model
            asyncio.run(run())
        self.assertEqual(peak, 1)

    def test_lock_released_after_model_failure(self):
        with patch.object(serve, 'pipeline') as model:
            model.return_value.analyze.side_effect = [RuntimeError('test failure'), {'verdict': 'refer_to_staff'}]
            with self.assertRaises(RuntimeError):
                serve._decode_and_analyze(image_bytes(), False)
            self.assertEqual(serve._decode_and_analyze(image_bytes(), False)['verdict'], 'refer_to_staff')


if __name__ == '__main__':
    unittest.main()
