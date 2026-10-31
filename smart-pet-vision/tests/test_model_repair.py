"""Prevent original export replacement and silently cropped mask evaluations."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from spv import deploy, segment


class ModelRepair(unittest.TestCase):
    def test_classifier_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'original.onnx'
            p.write_bytes(b'original export')
            model = MagicMock()
            with self.assertRaises(FileExistsError):
                deploy.export_classifier_onnx(model, p)
            self.assertEqual(p.read_bytes(), b'original export')
            model.cpu.assert_not_called()

    def test_detector_existing_export_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'original.onnx'
            p.write_bytes(b'original export')
            with self.assertRaises(FileExistsError):
                deploy.export_yolo_onnx(Path(folder) / 'weights.pt', 640, p)
            self.assertEqual(p.read_bytes(), b'original export')

    def test_yolo_extra_mask_pixels_are_not_silently_cropped(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'x.jpg'
            Image.new('RGB', (4, 3)).save(p)
            tensor = MagicMock()
            tensor.cpu.return_value.numpy.return_value = np.ones((1, 4, 5), dtype=bool)
            result = SimpleNamespace(orig_shape=(3, 4), masks=SimpleNamespace(data=tensor))
            with patch('ultralytics.YOLO') as yolo:
                yolo.return_value.predict.return_value = [result]
                with self.assertRaisesRegex(ValueError, 'Unaligned'):
                    segment.yolo_masks(Path('weights.pt'), [p], .25, 640)

    def test_yolo_missing_batch_results_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'x.jpg'
            Image.new('RGB', (4, 3)).save(p)
            with patch('ultralytics.YOLO') as yolo:
                yolo.return_value.predict.return_value = []
                with self.assertRaisesRegex(ValueError, 'entire image batch'):
                    segment.yolo_masks(Path('weights.pt'), [p], .25, 640)


if __name__ == '__main__':
    unittest.main()
