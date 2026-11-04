"""Fault injection at the decision boundary; no trained models required."""
import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from spv import serve
from spv.pipeline import InvalidModelOutput, PetPipeline, crop_coordinates


class OutputSafety(unittest.TestCase):
    def setUp(self):
        self.pipe = PetPipeline.__new__(PetPipeline)
        self.pipe.m = MagicMock(verify_threshold=.9, min_mask_fraction=.05, review_mode='off')
        self.pipe.detect = MagicMock(return_value=(np.array([[1., 1., 9., 9.]]), np.array([0]), np.array([.99])))
        self.pipe.classify = MagicMock(return_value=np.array([.99, .01]))
        self.pipe.segment = MagicMock(return_value=(np.ones((10, 10), bool), True))
        self.image = Image.new('RGB', (10, 10))

    def test_invalid_probabilities_never_produce_ready(self):
        for probs in ([np.nan, np.nan], [np.inf, 0], [.8, .8], [-.1, 1.1], [.99]):
            with self.subTest(probs=probs):
                self.pipe.classify.return_value = np.array(probs)
                with self.assertRaises(InvalidModelOutput):
                    self.pipe.analyze(self.image, False)

    def test_invalid_detector_output_is_rejected(self):
        for box, cls, score in [([1,1,np.nan,9], 0, .9), ([5,1,4,9],0,.9),
                                ([-1,1,9,9],0,.9), ([1,1,11,9],0,.9),
                                ([1,1,9,9],2,.9), ([1,1,9,9],0,np.nan)]:
            self.pipe.detect.return_value = (np.array([box]), np.array([cls]), np.array([score]))
            with self.subTest(box=box, cls=cls, score=score):
                with self.assertRaises(InvalidModelOutput):
                    self.pipe.analyze(self.image, False)

    def test_invalid_masks_refer_instead_of_becoming_foreground(self):
        for mask in (np.full((10,10), np.nan), np.full((10,10), 2), np.ones((1,10))):
            self.pipe.segment.return_value = (mask, True)
            result = self.pipe.analyze(self.image, False)
            self.assertEqual(result['verdict'], 'refer_to_staff')
            self.assertIn('invalid_segmentation', [r['code'] for r in result['reasons']])

    def test_subpixel_crop_and_cutout_are_nonempty(self):
        self.pipe.detect.return_value = (np.array([[1.1,1.1,1.2,1.2]]), np.array([0]), np.array([.99]))
        self.assertEqual(crop_coordinates([1.1,1.1,1.2,1.2],10,10), (1,1,2,2))
        result = self.pipe.analyze(self.image, True)
        self.assertTrue(result['cutout_png'])

    def test_total_is_direct_wall_time_including_encoding(self):
        result = self.pipe.analyze(self.image, True)
        self.assertGreaterEqual(result['timings']['total_ms'], result['timings']['stages_sum_ms'])

    def test_http_invalid_output_returns_503_without_internal_message(self):
        data = io.BytesIO(); self.image.save(data, 'PNG')
        self.pipe.analyze = MagicMock(side_effect=InvalidModelOutput('private model path'))
        with patch.object(serve, 'pipeline', return_value=self.pipe):
            response = TestClient(serve.app).post('/analyze', files={'file': ('x.png', data.getvalue(), 'image/png')})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private model path', response.text)
