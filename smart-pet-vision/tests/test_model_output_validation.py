"""Reject unusable inference output; these fixtures do not measure model accuracy."""
import io
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from spv import serve
from spv.pipeline import ModelOutputError, PetPipeline, SegmentationEvidence


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class OutputValidation(unittest.TestCase):
    def setUp(self):
        self.p = PetPipeline.__new__(PetPipeline)
        self.p.m = SimpleNamespace(verify_threshold=.9, min_mask_fraction=.05,
                                  imgsz=640, segmenter_conf=.25, segmenter_iou=.7,
                                  mask_match_iou=.1)
        self.image = Image.new('RGB', (32, 32))
        self.boxes = np.array([[2., 2., 18., 18.]])
        self.p.detect = MagicMock(return_value=(self.boxes, np.array([0]), np.array([.99])))
        self.p.classify = MagicMock(return_value=np.array([.99, .01]))
        self.p.segment = MagicMock(return_value=SegmentationEvidence(np.ones((32, 32), bool), True, 1, 'cat'))

    def test_valid_output_preserves_readiness_and_probabilities(self):
        result = self.p.analyze(self.image, with_images=False)
        self.assertEqual(result['verdict'], 'ready')
        self.assertEqual(result['probabilities'], {'cat': .99, 'dog': .01})
        self.assertFalse(result['automatic_publication_allowed'])

    def test_nan_classifier_cannot_return_ready(self):
        self.p.classify.return_value = np.array([np.nan, .01])
        with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)

    def test_invalid_probability_shape_or_values_fail(self):
        for probs in [[], [[.99, .01]], [.99], [.8, .8], [-.1, 1.1], [np.inf, 0], ['cat', 'dog']]:
            with self.subTest(probs=probs):
                self.p.classify.return_value = np.asarray(probs)
                with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)

    def test_invalid_detector_classes_are_not_cast_into_valid_classes(self):
        for cls in [-1, 2, .5, np.nan, np.inf]:
            with self.subTest(cls=cls):
                self.p.detect.return_value = (self.boxes, np.array([cls]), np.array([.99]))
                with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)
        self.p.classify.assert_not_called()

    def test_invalid_detection_confidence_fails_before_classification(self):
        for confidence in [-.1, 1.01, np.nan, np.inf]:
            with self.subTest(confidence=confidence):
                self.p.detect.return_value = (self.boxes, np.array([0]), np.array([confidence]))
                with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)
        self.p.classify.assert_not_called()

    def test_inconsistent_detection_shapes_fail(self):
        for boxes, classes, confs in [([], [], []), (self.boxes, [], [.99]),
                                     (self.boxes, [0], []), (self.boxes, [[0]], [.99])]:
            with self.subTest(boxes=boxes, classes=classes, confs=confs):
                self.p.detect.return_value = (boxes, classes, confs)
                with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)

    def test_invalid_boxes_fail_before_crop(self):
        for box in [[2, 2, 2, 18], [18, 2, 2, 18], [-1, 2, 18, 18],
                    [2, 2, 33, 18], [2, 2, np.nan, 18], [2, 2, 18, np.inf]]:
            with self.subTest(box=box):
                self.p.detect.return_value = (np.array([box]), np.array([0]), np.array([.99]))
                with self.assertRaises(ModelOutputError): self.p.analyze(self.image, False)
        self.p.classify.assert_not_called()

    def test_subpixel_box_is_referred_with_nonempty_crop(self):
        self.p.detect.return_value = (np.array([[2.1, 2.1, 2.2, 2.2]]), np.array([0]), np.array([.99]))
        result = self.p.analyze(self.image, False)
        self.assertEqual(result['verdict'], 'refer_to_staff')
        self.assertIn('animal_too_small', [x['code'] for x in result['reasons']])
        self.assertEqual(self.p.classify.call_args.args[0].size, (1, 1))

    def test_empty_detector_still_returns_explicit_referral(self):
        self.p.detect.return_value = (np.empty((0, 4)), np.array([]), np.array([]))
        result = self.p.analyze(self.image, False)
        self.assertIn('no_animal', [x['code'] for x in result['reasons']])

    def test_subpixel_cutout_is_a_valid_png(self):
        self.p.detect.return_value = (np.array([[2.1, 2.1, 2.2, 2.2]]), np.array([0]), np.array([.99]))
        result = self.p.analyze(self.image, True)
        with Image.open(io.BytesIO(base64.b64decode(result['cutout_png']))) as cutout:
            self.assertEqual(cutout.size, (1, 1))
        self.assertEqual(result['verdict'], 'refer_to_staff')

    def test_nan_or_nonbinary_segmenter_mask_is_not_foreground(self):
        del self.p.segment
        for masks in [np.full((1, 32, 32), np.nan), np.full((1, 32, 32), .5)]:
            with self.subTest(value=masks[0, 0, 0]):
                result = SimpleNamespace(orig_shape=(32, 32), masks=SimpleNamespace(data=Tensor(masks)),
                                         boxes=SimpleNamespace(xyxy=Tensor(self.boxes), cls=Tensor([0])))
                self.p.segmenter = MagicMock()
                self.p.segmenter.predict.return_value = [result]
                output = self.p.analyze(self.image, False)
                self.assertEqual(output['verdict'], 'refer_to_staff')
                self.assertFalse(output['segmentation_matched'])

    def test_fractional_segmenter_class_is_referred(self):
        del self.p.segment
        result = SimpleNamespace(orig_shape=(32, 32), masks=SimpleNamespace(data=Tensor(np.ones((1, 32, 32)))),
                                 boxes=SimpleNamespace(xyxy=Tensor(self.boxes), cls=Tensor([.5])))
        self.p.segmenter = MagicMock()
        self.p.segmenter.predict.return_value = [result]
        self.assertFalse(self.p.analyze(self.image, False)['segmentation_matched'])

    def test_http_model_failure_is_503_without_internal_details(self):
        buffer = io.BytesIO()
        self.image.save(buffer, format='PNG')
        with patch.object(serve, 'pipeline') as pipeline:
            pipeline.return_value.analyze.side_effect = ModelOutputError('private model path')
            response = TestClient(serve.app).post('/analyze', files={'file': ('x.png', buffer.getvalue())})
        self.assertEqual(response.status_code, 503)
        self.assertIn('staff review', response.json()['detail'])
        self.assertNotIn('private model path', response.text)


if __name__ == '__main__': unittest.main()
